use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapMetadata {
    pub resolution: f64,
    pub origin: [f64; 3],
    pub occupied_thresh: f64,
    pub free_thresh: f64,
    pub negate: bool,
    pub image_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct OccupancyGrid {
    pub cells: Vec<bool>,
    pub width: usize,
    pub height: usize,
    pub metadata: MapMetadata,
}

impl OccupancyGrid {
    pub fn is_occupied(&self, col: usize, row: usize) -> bool {
        if col >= self.width || row >= self.height {
            return false;
        }
        self.cells[row * self.width + col]
    }

    pub fn world_to_grid(&self, wx: f64, wy: f64) -> (i32, i32) {
        let col = ((wx - self.metadata.origin[0]) / self.metadata.resolution).floor() as i32;
        let row = ((wy - self.metadata.origin[1]) / self.metadata.resolution).floor() as i32;
        (col, row)
    }

    pub fn grid_to_world(&self, col: usize, row: usize) -> (f64, f64) {
        let wx = self.metadata.origin[0] + (col as f64 + 0.5) * self.metadata.resolution;
        let wy = self.metadata.origin[1] + (row as f64 + 0.5) * self.metadata.resolution;
        (wx, wy)
    }

    pub fn to_base64_png(&self) -> String {
        use base64::Engine as _;
        use image::{GrayImage, Luma};
        use std::io::Cursor;

        let mut img = GrayImage::new(self.width as u32, self.height as u32);
        for row in 0..self.height {
            let png_row = self.height - 1 - row;
            for col in 0..self.width {
                let val = if self.cells[row * self.width + col] {
                    0u8
                } else {
                    255u8
                };
                img.put_pixel(col as u32, png_row as u32, Luma([val]));
            }
        }
        let mut buf = Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png)
            .expect("PNG encode failed");
        let b64 = base64::engine::general_purpose::STANDARD.encode(buf.into_inner());
        format!("data:image/png;base64,{}", b64)
    }
}

#[derive(Deserialize)]
struct MapYaml {
    image: String,
    resolution: f64,
    origin: Vec<f64>,
    #[serde(default)]
    negate: u8,
    #[serde(default = "default_occupied_thresh")]
    occupied_thresh: f64,
    #[serde(default = "default_free_thresh")]
    free_thresh: f64,
}

fn default_occupied_thresh() -> f64 {
    0.65
}
fn default_free_thresh() -> f64 {
    0.196
}

pub fn load_map(yaml_path: &Path) -> Result<OccupancyGrid, Box<dyn std::error::Error>> {
    let yaml_str = std::fs::read_to_string(yaml_path)?;
    let cfg: MapYaml = serde_yaml::from_str(&yaml_str)?;

    let image_path = Path::new(&cfg.image);
    let resolved = if image_path.exists() {
        image_path.to_path_buf()
    } else {
        let fallback = yaml_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(image_path.file_name().unwrap_or(image_path.as_os_str()));
        if fallback.exists() {
            fallback
        } else {
            return Err(format!(
                "Map image not found at {} or {}",
                cfg.image,
                fallback.display()
            )
            .into());
        }
    };

    let img = image::open(&resolved)?.to_luma8();
    let (img_w, img_h) = (img.width() as usize, img.height() as usize);

    let negate = cfg.negate != 0;

    let mut cells = vec![false; img_w * img_h];
    for py in 0..img_h {
        let grid_row = img_h - 1 - py;
        for px in 0..img_w {
            let raw = img.get_pixel(px as u32, py as u32).0[0] as f64 / 255.0;
            let normalized = if negate { 1.0 - raw } else { raw };
            let occupancy_prob = 1.0 - normalized;
            cells[grid_row * img_w + px] = occupancy_prob >= cfg.occupied_thresh;
        }
    }

    let origin = if cfg.origin.len() >= 3 {
        [cfg.origin[0], cfg.origin[1], cfg.origin[2]]
    } else if cfg.origin.len() >= 2 {
        [cfg.origin[0], cfg.origin[1], 0.0]
    } else {
        [0.0, 0.0, 0.0]
    };

    Ok(OccupancyGrid {
        cells,
        width: img_w,
        height: img_h,
        metadata: MapMetadata {
            resolution: cfg.resolution,
            origin,
            occupied_thresh: cfg.occupied_thresh,
            free_thresh: cfg.free_thresh,
            negate,
            image_path: resolved,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_map(dir: &Path) -> PathBuf {
        let pgm_path = dir.join("test.pgm");
        let mut f = std::fs::File::create(&pgm_path).unwrap();
        write!(f, "P5\n4 3\n255\n").unwrap();
        f.write_all(&[255, 255, 255, 255]).unwrap();
        f.write_all(&[0, 0, 255, 255]).unwrap();
        f.write_all(&[255, 255, 255, 0]).unwrap();

        let yaml_path = dir.join("test.yaml");
        std::fs::write(
            &yaml_path,
            "image: test.pgm\nresolution: 0.05\norigin: [-1.0, -2.0, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
        ).unwrap();

        yaml_path
    }

    #[test]
    fn test_load_map_dimensions() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = create_test_map(dir.path());
        let grid = load_map(&yaml_path).unwrap();
        assert_eq!(grid.width, 4);
        assert_eq!(grid.height, 3);
    }

    #[test]
    fn test_load_map_occupancy() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = create_test_map(dir.path());
        let grid = load_map(&yaml_path).unwrap();

        assert!(!grid.is_occupied(0, 0));
        assert!(!grid.is_occupied(1, 0));
        assert!(!grid.is_occupied(2, 0));
        assert!(grid.is_occupied(3, 0));

        assert!(grid.is_occupied(0, 1));
        assert!(grid.is_occupied(1, 1));
        assert!(!grid.is_occupied(2, 1));
        assert!(!grid.is_occupied(3, 1));

        assert!(!grid.is_occupied(0, 2));
        assert!(!grid.is_occupied(1, 2));
        assert!(!grid.is_occupied(2, 2));
        assert!(!grid.is_occupied(3, 2));
    }

    #[test]
    fn test_load_map_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = create_test_map(dir.path());
        let grid = load_map(&yaml_path).unwrap();
        assert!((grid.metadata.resolution - 0.05).abs() < 1e-9);
        assert!((grid.metadata.origin[0] - (-1.0)).abs() < 1e-9);
        assert!((grid.metadata.origin[1] - (-2.0)).abs() < 1e-9);
        assert!(!grid.metadata.negate);
    }

    #[test]
    fn test_world_to_grid() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = create_test_map(dir.path());
        let grid = load_map(&yaml_path).unwrap();
        let (c, r) = grid.world_to_grid(-1.0, -2.0);
        assert_eq!(c, 0);
        assert_eq!(r, 0);
    }

    #[test]
    fn test_base64_png_is_data_url() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = create_test_map(dir.path());
        let grid = load_map(&yaml_path).unwrap();
        let b64 = grid.to_base64_png();
        assert!(b64.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_load_map_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let yaml_path = dir.path().join("missing.yaml");
        assert!(load_map(&yaml_path).is_err());
    }
}
