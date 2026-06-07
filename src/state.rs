//! State space representation for simulated annealing.

use crate::rng::SimpleRng;

/// Trait for state spaces in simulated annealing.
pub trait AnnealingState: Clone {
    fn energy(&self) -> f64;
    fn neighbor(&self, rng: &mut SimpleRng) -> Self;
}

/// A simple scalar state for 1D optimization.
#[derive(Clone, Debug)]
pub struct ScalarState {
    pub value: f64,
    pub step_size: f64,
    pub energy_fn: fn(f64) -> f64,
}

impl ScalarState {
    pub fn new(value: f64, step_size: f64, energy_fn: fn(f64) -> f64) -> Self {
        Self { value, step_size, energy_fn }
    }
}

impl AnnealingState for ScalarState {
    fn energy(&self) -> f64 {
        (self.energy_fn)(self.value)
    }

    fn neighbor(&self, rng: &mut SimpleRng) -> Self {
        Self {
            value: self.value + rng.gen_range(-self.step_size..self.step_size),
            step_size: self.step_size,
            energy_fn: self.energy_fn,
        }
    }
}

/// A 2D vector state for 2D optimization problems.
#[derive(Clone, Debug)]
pub struct VectorState2D {
    pub x: f64,
    pub y: f64,
    pub step_size: f64,
    pub energy_fn: fn(f64, f64) -> f64,
}

impl VectorState2D {
    pub fn new(x: f64, y: f64, step_size: f64, energy_fn: fn(f64, f64) -> f64) -> Self {
        Self { x, y, step_size, energy_fn }
    }
}

impl AnnealingState for VectorState2D {
    fn energy(&self) -> f64 {
        (self.energy_fn)(self.x, self.y)
    }

    fn neighbor(&self, rng: &mut SimpleRng) -> Self {
        Self {
            x: self.x + rng.gen_range(-self.step_size..self.step_size),
            y: self.y + rng.gen_range(-self.step_size..self.step_size),
            step_size: self.step_size,
            energy_fn: self.energy_fn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_energy() {
        let s = ScalarState::new(3.0, 0.1, |x| x * x);
        assert!((s.energy() - 9.0).abs() < 1e-10);
    }

    #[test]
    fn scalar_neighbor() {
        let mut rng = SimpleRng::new(42);
        let s = ScalarState::new(5.0, 10.0, |_| 0.0);
        let n = s.neighbor(&mut rng);
        assert!((n.value - 5.0).abs() < 10.0);
    }

    #[test]
    fn vector_energy() {
        let s = VectorState2D::new(3.0, 4.0, 0.1, |x, y| x * x + y * y);
        assert!((s.energy() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn vector_neighbor_preserves_fn() {
        let mut rng = SimpleRng::new(42);
        let s = VectorState2D::new(1.0, 2.0, 1.0, |x, y| x + y);
        let n = s.neighbor(&mut rng);
        assert!((n.energy() - (n.x + n.y)).abs() < 1e-10);
    }
}
