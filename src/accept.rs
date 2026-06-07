//! Acceptance criteria for simulated annealing.

use crate::rng::SimpleRng;

/// Trait for acceptance criteria.
pub trait AcceptanceCriterion {
    /// Decide whether to accept a transition.
    fn accept(&self, current_energy: f64, candidate_energy: f64, temp: f64, rng: &mut SimpleRng) -> bool;
}

/// Standard Metropolis-Hastings acceptance criterion.
#[derive(Clone)]
pub struct MetropolisAccept;

impl AcceptanceCriterion for MetropolisAccept {
    fn accept(&self, current_energy: f64, candidate_energy: f64, temp: f64, rng: &mut SimpleRng) -> bool {
        if candidate_energy <= current_energy {
            return true;
        }
        if temp <= 0.0 {
            return false;
        }
        let delta = candidate_energy - current_energy;
        let prob = (-delta / temp).exp();
        rng.gen_f64() < prob
    }
}

/// Threshold acceptance criterion.
#[derive(Clone)]
pub struct ThresholdAccept {
    pub threshold: f64,
}

impl AcceptanceCriterion for ThresholdAccept {
    fn accept(&self, current_energy: f64, candidate_energy: f64, _temp: f64, _rng: &mut SimpleRng) -> bool {
        let delta = candidate_energy - current_energy;
        delta <= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metropolis_always_accepts_improvement() {
        let mut rng = SimpleRng::new(42);
        assert!(MetropolisAccept.accept(10.0, 5.0, 1.0, &mut rng));
    }

    #[test]
    fn metropolis_rejects_at_zero_temp() {
        let mut rng = SimpleRng::new(42);
        assert!(!MetropolisAccept.accept(5.0, 10.0, 0.0, &mut rng));
    }

    #[test]
    fn threshold_accepts_below() {
        let mut rng = SimpleRng::new(42);
        assert!(ThresholdAccept { threshold: 1.0 }.accept(5.0, 5.5, 0.0, &mut rng));
    }

    #[test]
    fn threshold_rejects_above() {
        let mut rng = SimpleRng::new(42);
        assert!(!ThresholdAccept { threshold: 1.0 }.accept(5.0, 7.0, 0.0, &mut rng));
    }

    #[test]
    fn threshold_accepts_improvement() {
        let mut rng = SimpleRng::new(42);
        assert!(ThresholdAccept { threshold: 1.0 }.accept(5.0, 3.0, 0.0, &mut rng));
    }
}
