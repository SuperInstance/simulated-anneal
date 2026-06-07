# simulated-anneal

A comprehensive simulated annealing optimization framework in Rust with zero external dependencies.

## Features

- **Temperature Schedules**: Linear, exponential, and adaptive cooling strategies
- **Acceptance Criteria**: Metropolis-Hastings and threshold acceptance
- **Multi-Start**: Run multiple independent annealing processes from different initial states
- **Reheating**: Periodic reheating to escape local minima
- **State Space**: Built-in scalar and 2D vector state types with custom energy functions

## Usage

```rust
use simulated_anneal::{Annealer, LinearSchedule, MetropolisAccept, SimpleRng};

let schedule = LinearSchedule::new(100.0, 0.01, 5000);
let mut rng = SimpleRng::new(42);

let (best, energy) = Annealer::new(10.0_f64, schedule, MetropolisAccept, &mut rng)
    .with_energy(|x| x * x)  // minimize x^2
    .with_neighbor(|x, rng| *x + rng.gen_range(-0.5..0.5))
    .run();

println!("Best: {}, Energy: {}", best, energy);
```

## Modules

- `schedule` — Temperature schedules (linear, exponential, adaptive)
- `accept` — Acceptance criteria (Metropolis, threshold)
- `state` — State space representations (scalar, 2D vector)
- `multistart` — Multi-start annealing variant
- `reheat` — Reheating strategies

## License

MIT
