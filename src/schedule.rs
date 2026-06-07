//! Temperature schedule strategies for simulated annealing.

/// Trait for temperature schedule strategies.
pub trait TemperatureSchedule: Clone {
    fn temperature(&self, step: usize) -> f64;
    fn steps(&self) -> usize;
}

/// Linear cooling schedule.
#[derive(Clone)]
pub struct LinearSchedule {
    t_start: f64,
    t_end: f64,
    steps: usize,
}

impl LinearSchedule {
    pub fn new(t_start: f64, t_end: f64, steps: usize) -> Self {
        assert!(steps > 0, "steps must be positive");
        Self { t_start, t_end, steps }
    }
}

impl TemperatureSchedule for LinearSchedule {
    fn temperature(&self, step: usize) -> f64 {
        if step >= self.steps {
            return self.t_end;
        }
        let frac = step as f64 / (self.steps - 1) as f64;
        self.t_start + (self.t_end - self.t_start) * frac
    }

    fn steps(&self) -> usize {
        self.steps
    }
}

/// Exponential cooling schedule: `T(i) = t_start * alpha^i`.
#[derive(Clone)]
pub struct ExponentialSchedule {
    t_start: f64,
    alpha: f64,
    steps: usize,
}

impl ExponentialSchedule {
    pub fn new(t_start: f64, alpha: f64, steps: usize) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        assert!(steps > 0, "steps must be positive");
        Self { t_start, alpha, steps }
    }
}

impl TemperatureSchedule for ExponentialSchedule {
    fn temperature(&self, step: usize) -> f64 {
        self.t_start * self.alpha.powi(step as i32)
    }

    fn steps(&self) -> usize {
        self.steps
    }
}

/// Adaptive cooling schedule.
#[derive(Clone)]
pub struct AdaptiveSchedule {
    t_start: f64,
    alpha: f64,
    steps: usize,
    target_rate: f64,
    current_temp: f64,
}

impl AdaptiveSchedule {
    pub fn new(t_start: f64, alpha: f64, steps: usize, target_rate: f64) -> Self {
        assert!(steps > 0, "steps must be positive");
        Self {
            t_start,
            alpha,
            steps,
            target_rate: target_rate.clamp(0.0, 1.0),
            current_temp: t_start,
        }
    }

    pub fn update(&mut self, acceptance_rate: f64) {
        if acceptance_rate > self.target_rate {
            self.current_temp *= self.alpha;
        } else {
            self.current_temp /= self.alpha;
            if self.current_temp > self.t_start {
                self.current_temp = self.t_start;
            }
        }
    }
}

impl TemperatureSchedule for AdaptiveSchedule {
    fn temperature(&self, step: usize) -> f64 {
        let base_temp = self.t_start * self.alpha.powi(step as i32);
        base_temp.max(self.current_temp.min(self.t_start))
    }

    fn steps(&self) -> usize {
        self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_decreases() {
        let s = LinearSchedule::new(100.0, 0.0, 100);
        for i in 1..100 {
            assert!(s.temperature(i) <= s.temperature(i - 1) + 1e-10);
        }
    }

    #[test]
    fn exponential_positive() {
        let s = ExponentialSchedule::new(100.0, 0.99, 10000);
        for i in 0..10000 {
            assert!(s.temperature(i) > 0.0);
        }
    }

    #[test]
    fn adaptive_decreases_on_high_rate() {
        let mut s = AdaptiveSchedule::new(100.0, 0.95, 100, 0.5);
        let before = s.current_temp;
        s.update(0.8);
        assert!(s.current_temp < before);
    }

    #[test]
    fn adaptive_increases_on_low_rate() {
        let mut s = AdaptiveSchedule::new(100.0, 0.95, 100, 0.5);
        s.current_temp = 10.0;
        let before = s.current_temp;
        s.update(0.1);
        assert!(s.current_temp > before);
    }

    #[test]
    #[should_panic(expected = "steps must be positive")]
    fn linear_panics_on_zero() {
        LinearSchedule::new(100.0, 0.0, 0);
    }

    #[test]
    #[should_panic(expected = "alpha must be in (0, 1)")]
    fn exponential_panics_on_bad_alpha() {
        ExponentialSchedule::new(100.0, 1.5, 100);
    }
}
