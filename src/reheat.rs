//! Reheating strategies for simulated annealing.

use crate::schedule::TemperatureSchedule;
use crate::accept::AcceptanceCriterion;
use crate::Annealer;
use crate::SimpleRng;

/// Annealer with periodic reheating.
pub struct ReheatingAnnealer {
    pub cycles: usize,
    pub reheat_factor: f64,
}

impl ReheatingAnnealer {
    pub fn new(cycles: usize, reheat_factor: f64) -> Self {
        assert!(cycles > 0, "must have at least one cycle");
        assert!(reheat_factor > 0.0, "reheat factor must be positive");
        Self { cycles, reheat_factor }
    }

    /// Run annealing with reheating.
    pub fn run<Sch, Acc>(
        &self,
        initial: f64,
        schedule: Sch,
        accept: Acc,
        rng: &mut SimpleRng,
        energy_fn: impl Fn(&f64) -> f64 + Clone + 'static,
        neighbor_fn: impl FnMut(&f64, &mut SimpleRng) -> f64 + Clone + 'static,
    ) -> (f64, f64)
    where
        Sch: TemperatureSchedule + Clone,
        Acc: AcceptanceCriterion + Clone,
    {
        let mut current_state = initial;
        let mut best_state = initial;
        let mut best_energy = energy_fn(&initial);

        for cycle in 0..self.cycles {
            let sched = schedule.clone();
            let acc = accept.clone();
            let neighbor = neighbor_fn.clone();
            let efn = energy_fn.clone();

            let (state, energy) = Annealer::new(current_state, sched, acc, rng)
                .with_energy(move |x: &f64| efn(x))
                .with_neighbor(neighbor)
                .run();

            if energy < best_energy {
                best_energy = energy;
                best_state = state;
            }

            if cycle < self.cycles - 1 {
                let perturbation = rng.gen_range(-self.reheat_factor..self.reheat_factor);
                current_state = best_state + perturbation;
            }
        }

        (best_state, best_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExponentialSchedule, MetropolisAccept};

    #[test]
    fn reheating_improves() {
        let mut rng = SimpleRng::new(42);
        let schedule = ExponentialSchedule::new(30.0, 0.995, 1000);
        let rh = ReheatingAnnealer::new(3, 2.0);
        let (_best, energy) = rh.run(
            10.0, schedule, MetropolisAccept, &mut rng,
            |x: &f64| x * x,
            |x, rng| *x + rng.gen_range(-0.5..0.5),
        );
        assert!(energy < 10.0, "got {}", energy);
    }

    #[test]
    #[should_panic(expected = "must have at least one cycle")]
    fn reheating_panics_zero() {
        ReheatingAnnealer::new(0, 1.0);
    }

    #[test]
    #[should_panic(expected = "reheat factor must be positive")]
    fn reheating_panics_negative() {
        ReheatingAnnealer::new(3, -1.0);
    }

    #[test]
    fn reheating_finds_optimum() {
        let mut rng = SimpleRng::new(123);
        let schedule = ExponentialSchedule::new(50.0, 0.99, 2000);
        let rh = ReheatingAnnealer::new(5, 1.0);
        let (_best, energy) = rh.run(
            -5.0, schedule, MetropolisAccept, &mut rng,
            |x: &f64| (x - 2.0).powi(2),
            |x, rng| *x + rng.gen_range(-1.0..1.0),
        );
        assert!(energy < 5.0, "got {}", energy);
    }
}
