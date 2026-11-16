use crate::world3d::{World3d, V3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: V3,
    pub dir: V3,
    pub t_max: f64,
}

pub trait RayCaster: Send + Sync {
    fn cast_batch(&self, rays: &[Ray]) -> Vec<f64>;
}

impl RayCaster for World3d {
    fn cast_batch(&self, rays: &[Ray]) -> Vec<f64> {
        rays.iter().map(|r| self.cast(r.origin, r.dir, r.t_max)).collect()
    }
}
