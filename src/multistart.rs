//! Multi-start simulated annealing variant.

use crate::schedule::TemperatureSchedule;
use crate::accept::AcceptanceCriterion;
use crate::Annealer;
use crate::SimpleRng;

/// Multi-start annealer.
pub struct MultiStartAnnealer {
    pub num_starts: usize,
}

impl MultiStartAnnealer {
    pub fn new(num_starts: usize) -> Self {
        assert!(num_starts > 0, "must have at least one start");
        Self { num_starts }
    }

    /// Run multi-start annealing.
    pub fn run<Sch, Acc>(
        &self,
        initial_states: Vec<f64>,
        schedule_factory: impl Fn(usize) -> Sch,
        accept: Acc,
        rng: &mut SimpleRng,
        energy_fn: impl Fn(&f64) -> f64 + Clone + 'static,
        neighbor_fn: impl FnMut(&f64, &mut SimpleRng) -> f64 + Clone + 'static,
    ) -> (f64, f64)
    where
        Sch: TemperatureSchedule,
        Acc: AcceptanceCriterion + Clone,
    {
        assert_eq!(initial_states.len(), self.num_starts, "state count mismatch");

        let mut best_state = initial_states[0];
        let mut best_energy = f64::MAX;

        for (i, init) in initial_states.into_iter().enumerate() {
            let schedule = schedule_factory(i);
            let acc = accept.clone();
            let neighbor = neighbor_fn.clone();
            let efn = energy_fn.clone();
            let (state, energy) = Annealer::new(init, schedule, acc, rng)
                .with_energy(move |x: &f64| efn(x))
                .with_neighbor(neighbor)
                .run();

            if energy < best_energy {
                best_energy = energy;
                best_state = state;
            }
        }

        (best_state, best_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearSchedule, MetropolisAccept};

    #[test]
    fn multistart_finds_solution() {
        let mut rng = SimpleRng::new(42);
        let ms = MultiStartAnnealer::new(3);
        let (_best, energy) = ms.run(
            vec![-10.0, 10.0, 0.0],
            |_| LinearSchedule::new(50.0, 0.01, 2000),
            MetropolisAccept,
            &mut rng,
            |x: &f64| x * x,
            |x, rng| *x + rng.gen_range(-1.0..1.0),
        );
        assert!(energy < 10.0, "got {}", energy);
    }

    #[test]
    #[should_panic(expected = "must have at least one start")]
    fn multistart_panics_on_zero() {
        MultiStartAnnealer::new(0);
    }

    #[test]
    fn multistart_single() {
        let mut rng = SimpleRng::new(42);
        let ms = MultiStartAnnealer::new(1);
        let (_best, energy) = ms.run(
            vec![5.0],
            |_| LinearSchedule::new(20.0, 0.01, 500),
            MetropolisAccept,
            &mut rng,
            |x: &f64| (x - 3.0).powi(2),
            |x, rng| *x + rng.gen_range(-0.5..0.5),
        );
        assert!(energy < 10.0);
    }
}
