use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct MapMeta {
    pub image: String,
    pub resolution: f64,
    pub origin: Vec<f64>,
    #[serde(default)]
    pub negate: u8,
    #[serde(default = "default_occupied_thresh")]
    pub occupied_thresh: f64,
    #[serde(default = "default_free_thresh")]
    pub free_thresh: f64,
}

fn default_occupied_thresh() -> f64 {
    0.65
}

fn default_free_thresh() -> f64 {
    0.196
}

#[derive(Debug, Clone)]
pub struct OccupancyMap {
    pub meta: MapMeta,
    pub width: usize,
    pub height: usize,
    pub occupied: Vec<bool>,
    pub free: Vec<bool>,
}

pub fn parse_map_meta(yaml: &str) -> Result<MapMeta, String> {
    serde_yaml::from_str(yaml).map_err(|e| e.to_string())
}

fn pgm_tokens(bytes: &[u8], count: usize) -> Result<(Vec<usize>, usize), String> {
    let mut out = Vec::new();
    let mut i = 2;
    while out.len() < count {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b'#') {
            if bytes[i] == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let text = std::str::from_utf8(&bytes[start..i]).map_err(|e| e.to_string())?;
        out.push(text.parse::<usize>().map_err(|_| "bad PGM header".to_string())?);
    }
    Ok((out, i + 1))
}

pub fn decode_pgm(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    if bytes.len() < 2 || bytes[0] != b'P' || bytes[1] != b'5' {
        return Err("only binary PGM (P5) maps are supported".into());
    }
    let (header, offset) = pgm_tokens(bytes, 3)?;
    let (width, height, maxval) = (header[0], header[1], header[2]);
    if maxval > 255 {
        return Err("16-bit PGM maps are not supported".into());
    }
    let pixels = bytes.get(offset..offset + width * height).ok_or("PGM image is truncated")?;
    Ok((width, height, pixels.to_vec()))
}

impl OccupancyMap {
    pub fn from_pgm(meta: MapMeta, pgm: &[u8]) -> Result<Self, String> {
        let (width, height, pixels) = decode_pgm(pgm)?;
        let mut occupied = vec![false; width * height];
        let mut free = vec![false; width * height];
        for py in 0..height {
            let row = height - 1 - py;
            for px in 0..width {
                let raw = pixels[py * width + px] as f64 / 255.0;
                let value = if meta.negate != 0 { raw } else { 1.0 - raw };
                occupied[row * width + px] = value >= meta.occupied_thresh;
                free[row * width + px] = value <= meta.free_thresh;
            }
        }
        Ok(OccupancyMap { meta, width, height, occupied, free })
    }
}
