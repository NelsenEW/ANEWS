use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ab_glyph::{Font, FontVec, GlyphId, PxScale, ScaleFont};

pub const ATLAS: u32 = 1024;

const FONT_DIRS: [&str; 3] = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/usr/share/fonts/dejavu"];

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Weight {
    Regular,
    Bold,
}

#[derive(Clone, Copy)]
pub struct Placed {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub uv: [f32; 4],
}

#[derive(Clone, Copy)]
struct Cached {
    offset: [f32; 2],
    size: [f32; 2],
    uv: [f32; 4],
}

pub struct Fonts {
    regular: FontVec,
    bold: FontVec,
    cache: HashMap<(Weight, GlyphId, u32), Option<Cached>>,
    pub pixels: Vec<u8>,
    pub dirty: bool,
    shelf: (u32, u32, u32),
}

fn load(dir: &Path, file: &str) -> Result<FontVec, String> {
    let path = dir.join(file);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    FontVec::try_from_vec(bytes).map_err(|e| format!("{}: {e}", path.display()))
}

impl Fonts {
    pub fn new(dir: Option<&Path>) -> Result<Self, String> {
        let dir: PathBuf = match dir {
            Some(d) => d.to_path_buf(),
            None => FONT_DIRS
                .iter()
                .map(PathBuf::from)
                .find(|d| d.join("DejaVuSans.ttf").exists())
                .ok_or("DejaVuSans.ttf not found; pass --font-dir")?,
        };
        Ok(Fonts {
            regular: load(&dir, "DejaVuSans.ttf")?,
            bold: load(&dir, "DejaVuSans-Bold.ttf")?,
            cache: HashMap::new(),
            pixels: vec![0; (ATLAS * ATLAS) as usize],
            dirty: true,
            shelf: (1, 1, 0),
        })
    }

    fn font(&self, w: Weight) -> &FontVec {
        match w {
            Weight::Regular => &self.regular,
            Weight::Bold => &self.bold,
        }
    }

    pub fn width(&self, text: &str, px: f32, w: Weight) -> f32 {
        let f = self.font(w).as_scaled(PxScale::from(px));
        let mut x = 0.0;
        let mut prev: Option<GlyphId> = None;
        for ch in text.chars() {
            let id = f.glyph_id(ch);
            if let Some(p) = prev {
                x += f.kern(p, id);
            }
            x += f.h_advance(id);
            prev = Some(id);
        }
        x
    }

    pub fn ascent(&self, px: f32, w: Weight) -> f32 {
        self.font(w).as_scaled(PxScale::from(px)).ascent()
    }

    fn glyph(&mut self, w: Weight, id: GlyphId, px: f32) -> Result<Option<Cached>, String> {
        let key = (w, id, (px * 4.0).round() as u32);
        if let Some(c) = self.cache.get(&key) {
            return Ok(*c);
        }
        let glyph = id.with_scale_and_position(PxScale::from(px), ab_glyph::point(0.0, 0.0));
        let cached = match self.font(w).outline_glyph(glyph) {
            None => None,
            Some(outline) => {
                let b = outline.px_bounds();
                let (gw, gh) = (b.width().ceil() as u32, b.height().ceil() as u32);
                let (mut sx, mut sy, mut sh) = self.shelf;
                if sx + gw + 1 >= ATLAS {
                    sx = 1;
                    sy += sh + 1;
                    sh = 0;
                }
                if sy + gh + 1 >= ATLAS {
                    return Err("glyph atlas is full".into());
                }
                let pixels = &mut self.pixels;
                outline.draw(|x, y, c| {
                    let i = ((sy + y) * ATLAS + sx + x) as usize;
                    pixels[i] = (c.clamp(0.0, 1.0) * 255.0) as u8;
                });
                self.shelf = (sx + gw + 1, sy, sh.max(gh));
                self.dirty = true;
                let a = ATLAS as f32;
                Some(Cached {
                    offset: [b.min.x, b.min.y],
                    size: [gw as f32, gh as f32],
                    uv: [sx as f32 / a, sy as f32 / a, (sx + gw) as f32 / a, (sy + gh) as f32 / a],
                })
            }
        };
        self.cache.insert(key, cached);
        Ok(cached)
    }

    pub fn layout(&mut self, text: &str, x: f32, top: f32, px: f32, w: Weight) -> Result<Vec<Placed>, String> {
        let baseline = (top + self.ascent(px, w)).round();
        let mut pen = x;
        let mut prev: Option<GlyphId> = None;
        let mut out = Vec::new();
        for ch in text.chars() {
            let (id, adv, kern) = {
                let f = self.font(w).as_scaled(PxScale::from(px));
                let id = f.glyph_id(ch);
                (id, f.h_advance(id), prev.map(|p| f.kern(p, id)).unwrap_or(0.0))
            };
            pen += kern;
            if let Some(g) = self.glyph(w, id, px)? {
                let x0 = (pen + g.offset[0]).round();
                let y0 = baseline + g.offset[1];
                out.push(Placed { x0, y0, x1: x0 + g.size[0], y1: y0 + g.size[1], uv: g.uv });
            }
            pen += adv;
            prev = Some(id);
        }
        Ok(out)
    }
}
