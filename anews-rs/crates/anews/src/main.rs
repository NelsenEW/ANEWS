use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, Mutex};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message;

use anews_engine::sim::Simulation;

mod batch;
mod live;
mod offline;

#[derive(Parser)]
#[command(name = "anews", about = "Simulate a robot's restaurant shift and drive its real Nav2 stack through it")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    #[command(about = "Serve the 2D simulator over WebSocket")]
    Serve {
        #[arg(long, help = "Path to map YAML file")]
        map: PathBuf,

        #[arg(long, default_value = "8765", help = "WebSocket port")]
        port: u16,

        #[arg(long, help = "Optional scenario to load on startup")]
        scenario: Option<PathBuf>,

        #[arg(long, default_value = "config", help = "Config directory")]
        config_dir: PathBuf,
    },
    #[command(about = "Check, export and fit 3D venues")]
    Venue {
        #[command(subcommand)]
        command: offline::VenueCommand,
    },
    #[command(about = "Import recordings and rebuild their crowds")]
    Replay {
        #[command(subcommand)]
        command: offline::ReplayCommand,
    },
    #[command(about = "Compare simulated sensors with a recording")]
    Fidelity(offline::FidelityArgs),
    #[command(about = "Render a replay or MCAP recording to H.264 video on the GPU")]
    Render(offline::RenderArgs),
    #[command(about = "Run one simulated robot against a Nav2 stack over Zenoh")]
    Live(live::LiveArgs),
    #[command(about = "Run many robots on one simulation clock")]
    Sim {
        #[command(subcommand)]
        command: SimCommand,
    },
}

#[derive(clap::Subcommand)]
enum SimCommand {
    #[command(about = "Run a batch spec of missions and score each run")]
    Batch(batch::BatchArgs),
}

async fn until_ctrl_c<F>(run: F) -> Result<(), String>
where
    F: FnOnce(Arc<std::sync::atomic::AtomicBool>) -> Result<(), String> + Send + 'static,
{
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = stop.clone();
    let mut sim = tokio::task::spawn_blocking(move || run(flag));
    tokio::select! {
        done = &mut sim => done.map_err(|e| e.to_string())?,
        _ = tokio::signal::ctrl_c() => {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            sim.await.map_err(|e| e.to_string())?
        }
    }
}

type SimState = Arc<Mutex<Simulation>>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let command = match cli.command {
        Commands::Venue { command } => return offline::finish(offline::venue(command)),
        Commands::Replay { command } => return offline::finish(offline::replay(command)),
        Commands::Fidelity(args) => return offline::finish(offline::fidelity(args)),
        Commands::Render(args) => return offline::finish(offline::render(args)),
        Commands::Live(args) => return offline::finish(until_ctrl_c(move |stop| live::live(args, stop)).await),
        Commands::Sim { command: SimCommand::Batch(args) } => return offline::finish(until_ctrl_c(move |stop| batch::batch(args, stop)).await),
        serve => serve,
    };

    match command {
        Commands::Serve {
            map,
            port,
            scenario,
            config_dir,
        } => {
            let sim = Simulation::new(&map, scenario.as_deref(), &config_dir)?;

            let sim_state: SimState = Arc::new(Mutex::new(sim));

            let (tx, _) = broadcast::channel::<String>(4);

            let addr = SocketAddr::from(([0, 0, 0, 0], port));
            let listener = TcpListener::bind(&addr).await?;
            eprintln!("ANEWS Rust server listening on ws://0.0.0.0:{port}");

            let http_port = port + 1;
            let map_png_dir = std::env::temp_dir().join(format!("anews_map_{port}"));
            std::fs::create_dir_all(&map_png_dir).ok();
            {
                let sim = sim_state.lock().await;
                let init = sim.build_init();
                if let Some(image_b64) = init.get("map").and_then(|m| m.get("image_base64")).and_then(|v| v.as_str()) {
                    let raw_b64 = if let Some(idx) = image_b64.find(",") { &image_b64[idx+1..] } else { image_b64 };
                    if let Ok(bytes) = base64_decode(raw_b64) {
                        std::fs::write(map_png_dir.join("map.png"), &bytes).ok();
                    }
                }
            }
            tokio::spawn(start_http_server(http_port, map_png_dir));

            let tick_sim = sim_state.clone();
            let tick_tx = tx.clone();
            tokio::spawn(async move {
                let tick_dt = Duration::from_millis(50);
                loop {
                    let loop_start = Instant::now();

                    let state_json = {
                        let mut sim = tick_sim.lock().await;
                        sim.tick();
                        let state = sim.build_state();
                        sim.flush_dirty();
                        serde_json::to_string(&state).unwrap_or_default()
                    };

                    let _ = tick_tx.send(state_json);

                    let elapsed = loop_start.elapsed();
                    if elapsed < tick_dt {
                        tokio::time::sleep(tick_dt - elapsed).await;
                    }
                }
            });

            loop {
                let (stream, peer_addr) = listener.accept().await?;
                eprintln!("New connection from: {peer_addr}");
                let sim = sim_state.clone();
                let mut rx = tx.subscribe();
                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, sim, &mut rx).await {
                        eprintln!("Connection error from {peer_addr}: {e}");
                    }
                    eprintln!("Connection closed: {peer_addr}");
                });
            }
        }
        _ => unreachable!(),
    }
}

async fn start_http_server(port: u16, serve_dir: PathBuf) {
    use axum::Router;
    use tower_http::cors::CorsLayer;
    use tower_http::services::ServeDir;

    let cors = CorsLayer::very_permissive();
    let app = Router::new()
        .fallback_service(ServeDir::new(&serve_dir))
        .layer(cors);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    eprintln!("HTTP server listening on http://0.0.0.0:{port}");
    axum::serve(listener, app).await.unwrap();
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    STANDARD.decode(input).map_err(|e| e.to_string())
}

async fn handle_connection(
    stream: TcpStream,
    sim_state: SimState,
    rx: &mut broadcast::Receiver<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::sync::mpsc;

    let ws_stream = accept_async(stream).await?;
    let (mut write, mut read) = ws_stream.split();

    let (stream_tx, mut stream_rx) = mpsc::channel::<String>(256);

    let init_json = {
        let sim = sim_state.lock().await;
        serde_json::to_string(&sim.build_init())?
    };
    write.send(Message::Text(init_json)).await?;

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        let cmd: serde_json::Value = match serde_json::from_str(&text) {
                            Ok(v) => v,
                            Err(_) => {
                                let err = serde_json::json!({
                                    "type": "response",
                                    "status": "error",
                                    "error": "Invalid JSON"
                                });
                                write.send(Message::Text(
                                    serde_json::to_string(&err)?
                                )).await?;
                                continue;
                            }
                        };

                        let action = cmd.get("action").and_then(|v| v.as_str()).unwrap_or("");

                        match action {
                            "subscribe_events" => {
                                spawn_subscribe_events(sim_state.clone(), &cmd, stream_tx.clone()).await;
                                continue;
                            }
                            "run_benchmark_streaming" => {
                                spawn_benchmark_streaming(sim_state.clone(), &cmd, stream_tx.clone()).await;
                                continue;
                            }
                            "cancel" => {
                                let req_id = cmd
                                    .get("request_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                if !req_id.is_empty() {
                                    let sim = sim_state.lock().await;
                                    sim.cancellation.cancel(&req_id);
                                }
                                let ack = serde_json::json!({
                                    "type": "response",
                                    "status": "ok",
                                    "action": "cancel",
                                    "request_id": req_id,
                                });
                                write.send(Message::Text(serde_json::to_string(&ack)?)).await?;
                                continue;
                            }
                            _ => {}
                        }

                        let response = {
                            let mut sim = sim_state.lock().await;
                            sim.handle_command(&cmd)
                        };
                        if let Some(resp_type) = response.get("type").and_then(|v| v.as_str()) {
                            if resp_type == "response" || resp_type == "state" {
                                write.send(Message::Text(
                                    serde_json::to_string(&response)?
                                )).await?;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        eprintln!("WebSocket read error: {e}");
                        break;
                    }
                }
            }
            state = rx.recv() => {
                match state {
                    Ok(json) => {
                        if write.send(Message::Text(json)).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        eprintln!("Client lagged, dropped {n} state messages");
                    }
                    Err(_) => break,
                }
            }
            chunk = stream_rx.recv() => {
                match chunk {
                    Some(text) => {
                        if write.send(Message::Text(text)).await.is_err() {
                            break;
                        }
                    }
                    None => {}
                }
            }
        }
    }
    Ok(())
}

async fn spawn_benchmark_streaming(
    sim_state: SimState,
    cmd: &serde_json::Value,
    stream_tx: tokio::sync::mpsc::Sender<String>,
) {
    use anews_engine::events::SimEvent;
    use anews_engine::streaming;

    let request_id = cmd
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let params = cmd.get("params").cloned().unwrap_or(serde_json::json!({}));
    let scenario_path = params
        .get("scenario_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let timeout_secs = params
        .get("timeout_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(120);

    let resolved_path = {
        let p = std::path::PathBuf::from(&scenario_path);
        if p.is_absolute() {
            p
        } else {
            let stripped = p.strip_prefix("examples").unwrap_or(&p);
            let candidates = [
                std::path::PathBuf::from("/scenarios").join(stripped),
                std::path::PathBuf::from("/scenarios").join(&scenario_path),
                std::path::PathBuf::from(&scenario_path),
            ];
            candidates.into_iter()
                .find(|c| c.exists())
                .unwrap_or_else(|| std::path::PathBuf::from(&scenario_path))
        }
    };

    let load_result: Result<(_, _), String> = {
        let mut sim = sim_state.lock().await;
        match sim.load_scenario_file(&resolved_path) {
            Err(e) => Err(e.to_string()),
            Ok(()) => {
                sim.playing = true;
                let cancellation = sim.cancellation.clone();
                let rx = sim.event_bus.subscribe();
                Ok((cancellation, rx))
            }
        }
    };

    let (cancellation, mut event_rx) = match load_result {
        Err(msg) => {
            let chunk = streaming::chunk(
                &request_id,
                true,
                "error",
                serde_json::json!({
                    "code": "SCENARIO_LOAD_FAILED",
                    "message": msg,
                }),
            );
            let _ = stream_tx.send(chunk).await;
            return;
        }
        Ok(pair) => pair,
    };

    let req_id = request_id.clone();
    let sim_state_bg = sim_state.clone();

    tokio::spawn(async move {
        let start_time = std::time::Instant::now();
        let mut last_progress = std::time::Instant::now();
        let progress_interval = std::time::Duration::from_millis(2000);
        let mut terminated = false;
        let mut final_reason: Option<&'static str> = None;
        let mut flags: Vec<String> = Vec::new();
        let mut max_overshoot: f64 = 0.0;
        let mut min_clearance: f64 = f64::INFINITY;

        while !terminated {
            if cancellation.is_cancelled(&req_id) {
                let chunk = streaming::chunk(
                    &req_id,
                    true,
                    "cancelled",
                    serde_json::json!({"reason": "client_cancel"}),
                );
                let _ = stream_tx.send(chunk).await;
                cancellation.clear(&req_id);
                return;
            }

            let elapsed_s = start_time.elapsed().as_secs_f64();
            if elapsed_s >= timeout_secs as f64 {
                final_reason = Some("timeout");
                if !flags.contains(&"timeout".to_string()) {
                    flags.push("timeout".to_string());
                }
                break;
            }

            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                event_rx.recv(),
            )
            .await
            {
                Ok(Ok(event)) => {
                    match &event {
                        SimEvent::Overshoot { overshoot_m, .. } => {
                            if *overshoot_m > max_overshoot {
                                max_overshoot = *overshoot_m;
                            }
                            if !flags.contains(&"overshooting".to_string()) {
                                flags.push("overshooting".to_string());
                            }
                        }
                        SimEvent::Oscillation { .. } => {
                            if !flags.contains(&"oscillatory".to_string()) {
                                flags.push("oscillatory".to_string());
                            }
                        }
                        SimEvent::NearMiss { distance_m, .. } => {
                            if *distance_m < min_clearance {
                                min_clearance = *distance_m;
                            }
                            if !flags.contains(&"near_miss".to_string()) {
                                flags.push("near_miss".to_string());
                            }
                        }
                        _ => {}
                    }

                    let is_high_signal = matches!(
                        event,
                        SimEvent::Collision { .. }
                            | SimEvent::NearMiss { .. }
                            | SimEvent::GoalReached { .. }
                            | SimEvent::Timeout { .. }
                            | SimEvent::Stuck { .. }
                    );
                    if is_high_signal {
                        let payload =
                            serde_json::to_value(&event).unwrap_or(serde_json::json!({}));
                        let chunk = streaming::chunk(&req_id, false, "event", payload);
                        if stream_tx.send(chunk).await.is_err() {
                            return;
                        }
                    }

                    match event {
                        SimEvent::GoalReached { .. } => {
                            final_reason = Some("goal_reached");
                            terminated = true;
                        }
                        SimEvent::Collision { .. } => {
                            final_reason = Some("collision");
                            if !flags.contains(&"collision".to_string()) {
                                flags.push("collision".to_string());
                            }
                            terminated = true;
                        }
                        SimEvent::Timeout { .. } => {
                            final_reason = Some("timeout");
                            if !flags.contains(&"timeout".to_string()) {
                                flags.push("timeout".to_string());
                            }
                            terminated = true;
                        }
                        _ => {}
                    }
                }
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => {
                }
                Ok(Err(_)) => return,
                Err(_) => {}
            }

            if last_progress.elapsed() >= progress_interval {
                last_progress = std::time::Instant::now();
                let elapsed_s = start_time.elapsed().as_secs_f64();

                let (current_tick, rx, ry, rtheta, goal_dist) = {
                    let sim = sim_state_bg.lock().await;
                    let tick = sim.tick_count();
                    let (px, py, pt) = sim.drive.pose();
                    let goal_dist = sim
                        .goal
                        .map(|(gx, gy)| {
                            let dx = px - gx;
                            let dy = py - gy;
                            (dx * dx + dy * dy).sqrt()
                        })
                        .unwrap_or(0.0);
                    (tick, px, py, pt, goal_dist)
                };

                let progress_pct =
                    ((elapsed_s / timeout_secs as f64) * 100.0).min(100.0) as u32;
                let chunk = streaming::chunk(
                    &req_id,
                    false,
                    "progress",
                    serde_json::json!({
                        "progress_pct": progress_pct,
                        "tick": current_tick,
                        "elapsed_s": elapsed_s,
                        "robot_pose": {"x": rx, "y": ry, "theta": rtheta},
                        "goal_distance_m": goal_dist,
                    }),
                );
                if stream_tx.send(chunk).await.is_err() {
                    return;
                }
            }
        }

        let grade = if matches!(final_reason, Some("goal_reached"))
            && !flags.contains(&"collision".to_string())
            && !flags.contains(&"timeout".to_string())
        {
            "hard_pass"
        } else {
            "hard_fail"
        };

        let elapsed_s = start_time.elapsed().as_secs_f64();
        let metrics = serde_json::json!({
            "time_to_goal_s": if grade == "hard_pass" { serde_json::json!(elapsed_s) } else { serde_json::json!(null) },
            "max_overshoot_m": max_overshoot,
            "min_clearance_m": if min_clearance.is_finite() { serde_json::json!(min_clearance) } else { serde_json::json!(null) },
            "path_length_m": serde_json::json!(null),
        });

        let final_chunk = streaming::chunk(
            &req_id,
            true,
            "result",
            serde_json::json!({
                "grade": grade,
                "flags": flags,
                "metrics": metrics,
                "event_log_path": "",
                "reason": final_reason.unwrap_or("unknown"),
            }),
        );
        let _ = stream_tx.send(final_chunk).await;
    });
}

async fn spawn_subscribe_events(
    sim_state: SimState,
    cmd: &serde_json::Value,
    stream_tx: tokio::sync::mpsc::Sender<String>,
) {
    use anews_engine::streaming;

    let request_id = cmd
        .get("request_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let params = cmd.get("params").cloned().unwrap_or(serde_json::json!({}));

    let until: Vec<String> = params
        .get("until")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let max_ticks = params
        .get("max_ticks")
        .and_then(|v| v.as_u64())
        .unwrap_or(2000);
    let max_secs = params
        .get("max_secs")
        .and_then(|v| v.as_f64())
        .unwrap_or(120.0);

    let (mut rx, cancellation, start_tick) = {
        let sim = sim_state.lock().await;
        let rx = sim.event_bus.subscribe();
        let cancellation = sim.cancellation.clone();
        let start_tick = sim.tick_count();
        (rx, cancellation, start_tick)
    };

    let req_id = request_id.clone();
    let sim_state_bg = sim_state.clone();

    tokio::spawn(async move {
        let start_time = std::time::Instant::now();

        loop {
            if cancellation.is_cancelled(&req_id) {
                let final_chunk = streaming::chunk(
                    &req_id,
                    true,
                    "cancelled",
                    serde_json::json!({"reason": "client_cancel"}),
                );
                let _ = stream_tx.send(final_chunk).await;
                cancellation.clear(&req_id);
                return;
            }

            let current_tick = {
                let sim = sim_state_bg.lock().await;
                sim.tick_count()
            };
            let ticks_elapsed = current_tick.saturating_sub(start_tick);
            let secs_elapsed = start_time.elapsed().as_secs_f64();

            if ticks_elapsed >= max_ticks {
                let final_chunk = streaming::chunk(
                    &req_id,
                    true,
                    "result",
                    serde_json::json!({
                        "reason": "max_ticks",
                        "total_ticks": ticks_elapsed,
                        "total_secs": secs_elapsed,
                    }),
                );
                let _ = stream_tx.send(final_chunk).await;
                return;
            }
            if secs_elapsed >= max_secs {
                let final_chunk = streaming::chunk(
                    &req_id,
                    true,
                    "result",
                    serde_json::json!({
                        "reason": "max_secs",
                        "total_ticks": ticks_elapsed,
                        "total_secs": secs_elapsed,
                    }),
                );
                let _ = stream_tx.send(final_chunk).await;
                return;
            }

            match tokio::time::timeout(
                std::time::Duration::from_millis(100),
                rx.recv(),
            )
            .await
            {
                Ok(Ok(event)) => {
                    let kind = event.kind().to_string();
                    if until.contains(&kind) {
                        let final_chunk = streaming::chunk(
                            &req_id,
                            true,
                            "result",
                            serde_json::json!({
                                "reason": kind,
                                "total_ticks": ticks_elapsed,
                                "total_secs": secs_elapsed,
                            }),
                        );
                        let _ = stream_tx.send(final_chunk).await;
                        return;
                    }
                    let payload =
                        serde_json::to_value(&event).unwrap_or(serde_json::json!({}));
                    let chunk = streaming::chunk(&req_id, false, "event", payload);
                    if stream_tx.send(chunk).await.is_err() {
                        return;
                    }
                }
                Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => {
                }
                Ok(Err(_)) => {
                    return;
                }
                Err(_) => {
                }
            }
        }
    });
}
