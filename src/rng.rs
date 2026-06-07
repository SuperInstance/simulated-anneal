//! Simple deterministic PRNG (LCG) to avoid external dependencies.

/// A simple Linear Congruential Generator for reproducible randomness.
#[derive(Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    /// Generate the next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a random f64 in [0, 1).
    pub fn gen_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a random f64 in [low, high).
    pub fn gen_range(&mut self, range: std::ops::Range<f64>) -> f64 {
        let t = self.gen_f64();
        range.start + t * (range.end - range.start)
    }

    /// Generate a random bool.
    pub fn gen_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_produces_different_values() {
        let mut rng = SimpleRng::new(42);
        let a = rng.gen_f64();
        let b = rng.gen_f64();
        assert_ne!(a, b);
    }

    #[test]
    fn rng_in_range() {
        let mut rng = SimpleRng::new(42);
        for _ in 0..100 {
            let v = rng.gen_range(-1.0..1.0);
            assert!(v >= -1.0 && v < 1.0);
        }
    }

    #[test]
    fn rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);
        for _ in 0..10 {
            assert_eq!(rng1.gen_f64(), rng2.gen_f64());
        }
    }

    #[test]
    fn rng_seed_zero_does_not_repeat() {
        let mut rng = SimpleRng::new(0);
        let a = rng.gen_f64();
        assert_ne!(a, 0.0);
    }
}
