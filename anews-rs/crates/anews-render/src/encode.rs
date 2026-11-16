use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{json, Value};

pub const ENCODER: &str = "h264_nvenc";
const MIN_AVAILABLE_MIB: u64 = 1536;
const MIN_GPU_FREE_MIB: u64 = 1024;
const MAX_SWAP_PAGES_PER_S: f64 = 2048.0;

pub struct Lock {
    _file: File,
}

pub fn lock() -> Result<Lock, String> {
    let dir = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let path = dir.join("anews-render.lock");
    let file = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    match file.try_lock() {
        Ok(()) => Ok(Lock { _file: file }),
        Err(std::fs::TryLockError::WouldBlock) => Err(format!("another anews render holds {}; run one render at a time", path.display())),
        Err(std::fs::TryLockError::Error(e)) => Err(format!("{}: {e}", path.display())),
    }
}

fn meminfo(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    text.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok())
}

fn swap_pages() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/vmstat").ok()?;
    Some(
        text.lines()
            .filter(|l| l.starts_with("pswpin ") || l.starts_with("pswpout "))
            .filter_map(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            .sum(),
    )
}

fn nvidia_smi(query: &str) -> Result<Vec<String>, String> {
    let out = Command::new("nvidia-smi")
        .args([&format!("--query-gpu={query}"), "--format=csv,noheader,nounits"])
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi failed: {}", String::from_utf8_lossy(&out.stderr).trim()));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Ok(text.lines().next().unwrap_or("").split(',').map(|s| s.trim().to_string()).collect())
}

pub fn preflight() -> Result<Value, String> {
    let available = meminfo("MemAvailable:").ok_or("cannot read /proc/meminfo")? / 1024;
    if available < MIN_AVAILABLE_MIB {
        return Err(format!("only {available} MiB RAM available (need {MIN_AVAILABLE_MIB}); free memory before rendering"));
    }
    let s0 = swap_pages().ok_or("cannot read /proc/vmstat")?;
    std::thread::sleep(Duration::from_secs(1));
    let swap_rate = (swap_pages().unwrap_or(s0) - s0) as f64;
    if swap_rate > MAX_SWAP_PAGES_PER_S {
        return Err(format!("host is swapping ({swap_rate:.0} pages/s); wait for it to settle before rendering"));
    }
    let gpu = nvidia_smi("name,memory.total,memory.used")?;
    let total: u64 = gpu.get(1).and_then(|v| v.parse().ok()).ok_or("nvidia-smi memory.total unreadable")?;
    let used: u64 = gpu.get(2).and_then(|v| v.parse().ok()).ok_or("nvidia-smi memory.used unreadable")?;
    if total - used < MIN_GPU_FREE_MIB {
        return Err(format!("GPU has {} MiB free (need {MIN_GPU_FREE_MIB})", total - used));
    }
    nvenc_probe()?;
    Ok(json!({
        "ram_available_mib": available,
        "swap_total_mib": meminfo("SwapTotal:").unwrap_or(0) / 1024,
        "swap_free_mib": meminfo("SwapFree:").unwrap_or(0) / 1024,
        "swap_pages_per_s": swap_rate,
        "gpu": gpu.first(),
        "gpu_memory_used_mib": used,
        "gpu_memory_total_mib": total,
        "nvenc_probe": "ok",
    }))
}

pub fn nvenc_probe() -> Result<(), String> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=256x256:r=25:d=0.4"])
        .args(["-c:v", ENCODER, "-profile:v", "main", "-pix_fmt", "yuv420p", "-f", "null", "-"])
        .output()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{ENCODER} test encode failed ({}); refusing to fall back to CPU encoding",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(())
}

pub struct Encoder {
    child: Child,
    tx: Option<SyncSender<Vec<u8>>>,
    writer: Option<JoinHandle<Result<(), String>>>,
}

impl Encoder {
    pub fn start(out: &Path, width: u32, height: u32, fps: u32, cq: u32) -> Result<Self, String> {
        let size = format!("{width}x{height}");
        let mut child = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-y"])
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", &size, "-r", &fps.to_string(), "-i", "-"])
            .args(["-c:v", ENCODER, "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", &cq.to_string(), "-b:v", "0"])
            .args(["-profile:v", "main", "-pix_fmt", "yuv420p"])
            .args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"])
            .args(["-movflags", "+faststart"])
            .arg(out)
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        let mut stdin = child.stdin.take().ok_or("ffmpeg stdin")?;
        let (tx, rx) = sync_channel::<Vec<u8>>(4);
        let writer = std::thread::spawn(move || {
            for frame in rx {
                stdin.write_all(&frame).map_err(|e| format!("ffmpeg stdin: {e}"))?;
            }
            Ok(())
        });
        Ok(Encoder { child, tx: Some(tx), writer: Some(writer) })
    }

    pub fn push(&self, frame: Vec<u8>) -> Result<(), String> {
        self.tx.as_ref().ok_or("encoder closed")?.send(frame).map_err(|_| "ffmpeg writer stopped".to_string())
    }

    pub fn finish(mut self) -> Result<(), String> {
        drop(self.tx.take());
        let written = self.writer.take().map(|w| w.join().map_err(|_| "writer panicked".to_string())).transpose()?;
        let mut err = String::new();
        if let Some(mut e) = self.child.stderr.take() {
            std::io::Read::read_to_string(&mut e, &mut err).ok();
        }
        let status = self.child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
        if !status.success() {
            return Err(format!("ffmpeg {ENCODER} failed: {}", err.trim()));
        }
        written.unwrap_or(Ok(()))
    }
}

pub fn probe_codec(path: &Path) -> Result<Value, String> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_frames:format=duration"])
        .args(["-of", "json"])
        .arg(path)
        .output()
        .map_err(|e| format!("ffprobe: {e}"))?;
    let v: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("ffprobe output: {e}"))?;
    Ok(json!({"stream": v["streams"][0], "duration_s": v["format"]["duration"]}))
}

pub struct GpuMonitor {
    child: Child,
    samples: Arc<Mutex<Vec<[f64; 3]>>>,
    reader: JoinHandle<()>,
}

impl GpuMonitor {
    pub fn start() -> Result<Self, String> {
        let mut child = Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu,utilization.encoder,memory.used", "--format=csv,noheader,nounits", "-lms", "500"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("nvidia-smi: {e}"))?;
        let stdout = child.stdout.take().ok_or("nvidia-smi stdout")?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let sink = samples.clone();
        let reader = std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let v: Vec<f64> = line.split(',').filter_map(|s| s.trim().parse().ok()).collect();
                if v.len() == 3 {
                    sink.lock().unwrap().push([v[0], v[1], v[2]]);
                }
            }
        });
        Ok(GpuMonitor { child, samples, reader })
    }

    pub fn stop(mut self) -> Value {
        self.child.kill().ok();
        self.child.wait().ok();
        self.reader.join().ok();
        let s = self.samples.lock().unwrap();
        let n = s.len().max(1) as f64;
        let mean = |k: usize| s.iter().map(|v| v[k]).sum::<f64>() / n;
        let max = |k: usize| s.iter().map(|v| v[k]).fold(0.0, f64::max);
        json!({
            "samples": s.len(),
            "gpu_util_mean_pct": mean(0),
            "gpu_util_max_pct": max(0),
            "encoder_util_mean_pct": mean(1),
            "gpu_memory_max_mib": max(2),
        })
    }
}

pub fn cpu_seconds() -> f64 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else { return 0.0 };
    let rest = stat.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let f: Vec<f64> = rest.split_whitespace().map(|v| v.parse().unwrap_or(0.0)).collect();
    let tick = 100.0;
    if f.len() < 15 {
        return 0.0;
    }
    (f[11] + f[12] + f[13] + f[14]) / tick
}

pub fn peak_rss_mib() -> f64 {
    meminfo_self("VmHWM:").unwrap_or(0) as f64 / 1024.0
}

fn meminfo_self(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    text.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok())
}
