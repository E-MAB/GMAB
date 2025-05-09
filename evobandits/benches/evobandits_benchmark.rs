use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use evobandits::evobandits::EvoBandits;
use rand::rng;
use rand_distr::{Distribution, Normal};

pub fn noisy_rosenbrock(x: &[i32]) -> f64 {
    let x_f64 = x[0] as f64 / 10.0;
    let y_f64 = x[1] as f64 / 10.0;

    // Rosenbrock function
    let term1 = (1.0 - x_f64).powi(2);
    let term2 = 100.0 * (y_f64 - x_f64.powi(2)).powi(2);
    let base_value = term1 + term2;

    // Add Gaussian noise
    let mut rng = rng();
    let normal = Normal::new(0.0, 5.0).unwrap();
    let noise = normal.sample(&mut rng);

    base_value + noise
}

fn benchmark_evobandits(c: &mut Criterion) {
    let mut group = c.benchmark_group("Rosenbrock Optimization");

    group.measurement_time(std::time::Duration::from_secs(60));

    // Simulate different budgets
    for budget in [10_000, 100_000].iter() {
        group.bench_with_input(BenchmarkId::new("Noisy", budget), budget, |b, &budget| {
            b.iter(|| {
                let mut evobandits = EvoBandits::new(Default::default());
                let bounds = vec![(-50, 50), (-50, 50)];

                // Run the optimization
                let result = evobandits.optimize(
                    black_box(noisy_rosenbrock),
                    black_box(bounds),
                    black_box(budget),
                    1,
                    Default::default(),
                );

                result
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_evobandits);
criterion_main!(benches);
