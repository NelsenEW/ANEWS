use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffDrive {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
}

impl DiffDrive {
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }
    pub fn update(&mut self, linear: f64, angular: f64, dt: f64) {
        if angular.abs() < 1e-6 {
            self.x += linear * self.theta.cos() * dt;
            self.y += linear * self.theta.sin() * dt;
        } else {
            let r = linear / angular;
            self.x += r * ((self.theta + angular * dt).sin() - self.theta.sin());
            self.y += r * (-(self.theta + angular * dt).cos() + self.theta.cos());
        }
        self.theta += angular * dt;
        self.theta = self.theta.sin().atan2(self.theta.cos());
    }

    pub fn reset(&mut self, x: f64, y: f64, theta: f64) {
        self.x = x;
        self.y = y;
        self.theta = theta;
    }

    pub fn pose(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_straight_line() {
        let mut drive = DiffDrive::new(0.0, 0.0, 0.0);
        drive.update(1.0, 0.0, 1.0);
        assert!((drive.x - 1.0).abs() < 1e-6);
        assert!(drive.y.abs() < 1e-6);
        assert!(drive.theta.abs() < 1e-6);
    }

    #[test]
    fn test_pure_rotation() {
        let mut drive = DiffDrive::new(0.0, 0.0, 0.0);
        drive.update(0.0, 1.0, 1.0);
        assert!(drive.x.abs() < 1e-6);
        assert!(drive.y.abs() < 1e-6);
        assert!((drive.theta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_arc_motion() {
        let mut drive = DiffDrive::new(0.0, 0.0, 0.0);
        drive.update(1.0, PI, 1.0);
        let expected_x = (1.0 / PI) * (PI.sin() - 0.0);
        let expected_y = (1.0 / PI) * (-PI.cos() + 1.0);
        assert!((drive.x - expected_x).abs() < 1e-6, "x={}", drive.x);
        assert!((drive.y - expected_y).abs() < 1e-6, "y={}", drive.y);
    }

    #[test]
    fn test_theta_normalization() {
        let mut drive = DiffDrive::new(0.0, 0.0, 3.0);
        drive.update(0.0, 1.0, 1.0);
        assert!(
            drive.theta >= -PI && drive.theta <= PI,
            "theta={} should be in [-PI, PI]",
            drive.theta
        );
    }

    #[test]
    fn test_reset() {
        let mut drive = DiffDrive::new(0.0, 0.0, 0.0);
        drive.update(1.0, 0.5, 2.0);
        drive.reset(10.0, 20.0, 1.5);
        assert!((drive.x - 10.0).abs() < 1e-9);
        assert!((drive.y - 20.0).abs() < 1e-9);
        assert!((drive.theta - 1.5).abs() < 1e-9);
    }
}
