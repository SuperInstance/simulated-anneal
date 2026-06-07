//! # Simulated Annealing Framework
//!
//! A comprehensive simulated annealing optimization library providing multiple
//! temperature schedules, acceptance criteria, multi-start variants, and reheating strategies.
//!
//! # Example
//! ```
//! use simulated_anneal::{Annealer, LinearSchedule, MetropolisAccept, SimpleRng};
//!
//! let schedule = LinearSchedule::new(100.0, 0.1, 1000);
//! let mut rng = SimpleRng::new(42);
//! let (best, energy) = Annealer::new(0.0_f64, schedule, MetropolisAccept, &mut rng)
//!     .with_energy(|x: &f64| x * x)
//!     .with_neighbor(|x: &f64, rng| *x + rng.gen_range(-0.5..0.5))
//!     .run();
//! ```

pub mod schedule;
pub mod accept;
pub mod state;
pub mod multistart;
pub mod reheat;
mod rng;

pub use schedule::{LinearSchedule, ExponentialSchedule, AdaptiveSchedule, TemperatureSchedule};
pub use accept::{MetropolisAccept, AcceptanceCriterion, ThresholdAccept};
pub use state::AnnealingState;
pub use multistart::MultiStartAnnealer;
pub use reheat::ReheatingAnnealer;
pub use rng::SimpleRng;

/// Core annealer that drives the simulated annealing process.
#[allow(clippy::type_complexity)]
pub struct Annealer<S, Sch, Acc> {
    state: S,
    schedule: Sch,
    accept: Acc,
    rng: SimpleRng,
    energy_fn: Option<Box<dyn Fn(&S) -> f64>>,
    neighbor_fn: Option<Box<dyn FnMut(&S, &mut SimpleRng) -> S>>,
}

impl<S, Sch, Acc> Annealer<S, Sch, Acc> {
    /// Create a new annealer.
    pub fn new(state: S, schedule: Sch, accept: Acc, rng: &mut SimpleRng) -> Self {
        Self {
            state,
            schedule,
            accept,
            rng: rng.clone(),
            energy_fn: None,
            neighbor_fn: None,
        }
    }

    /// Set the energy function.
    pub fn with_energy(mut self, f: impl Fn(&S) -> f64 + 'static) -> Self {
        self.energy_fn = Some(Box::new(f));
        self
    }

    /// Set the neighbor generation function.
    pub fn with_neighbor(mut self, f: impl FnMut(&S, &mut SimpleRng) -> S + 'static) -> Self {
        self.neighbor_fn = Some(Box::new(f));
        self
    }

    /// Run the annealing process, returning (final_state, best_energy).
    pub fn run(mut self) -> (S, f64)
    where
        Sch: TemperatureSchedule,
        Acc: AcceptanceCriterion,
    {
        let energy_fn = self.energy_fn.expect("energy function required");
        let mut neighbor_fn = self.neighbor_fn.expect("neighbor function required");

        let mut best_energy = energy_fn(&self.state);
        let mut current_energy = best_energy;

        let steps = self.schedule.steps();
        for i in 0..steps {
            let temp = self.schedule.temperature(i);
            let candidate = neighbor_fn(&self.state, &mut self.rng);
            let candidate_energy = energy_fn(&candidate);

            if self.accept.accept(current_energy, candidate_energy, temp, &mut self.rng) {
                self.state = candidate;
                current_energy = candidate_energy;

                if candidate_energy < best_energy {
                    best_energy = candidate_energy;
                }
            }
        }

        (self.state, best_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metropolis_accepts_improvement() {
        let mut rng = SimpleRng::new(42);
        assert!(MetropolisAccept.accept(10.0, 5.0, 1.0, &mut rng));
    }

    #[test]
    fn test_metropolis_rejects_at_zero_temp() {
        let mut rng = SimpleRng::new(42);
        assert!(!MetropolisAccept.accept(5.0, 10.0, 0.0, &mut rng));
    }

    #[test]
    fn test_linear_schedule_start() {
        let s = LinearSchedule::new(100.0, 0.1, 1000);
        assert!((s.temperature(0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_decay() {
        let s = ExponentialSchedule::new(100.0, 0.99, 1000);
        assert!(s.temperature(1) < 100.0);
        assert!(s.temperature(1) >= 99.0);
    }

    #[test]
    fn test_simple_optimization() {
        let schedule = LinearSchedule::new(100.0, 0.01, 5000);
        let mut rng = SimpleRng::new(42);
        let (_best, energy) = Annealer::new(10.0_f64, schedule, MetropolisAccept, &mut rng)
            .with_energy(|x| x * x)
            .with_neighbor(|x, rng| *x + rng.gen_range(-0.5..0.5))
            .run();
        assert!(energy < 5.0, "energy should be small, got {}", energy);
    }

    #[test]
    fn test_sphere_optimization() {
        let schedule = ExponentialSchedule::new(50.0, 0.995, 5000);
        let mut rng = SimpleRng::new(123);
        let (_best, energy) = Annealer::new(0.0_f64, schedule, MetropolisAccept, &mut rng)
            .with_energy(|x| (x - 3.0).powi(2))
            .with_neighbor(|x, rng| *x + rng.gen_range(-1.0..1.0))
            .run();
        assert!(energy < 5.0, "should find near optimum, got energy={}", energy);
    }
}
