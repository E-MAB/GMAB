use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::arm::{Arm, OptimizationFn};

pub(crate) struct GeneticAlgorithm<F: OptimizationFn> {
    mutation_rate: f64,
    crossover_rate: f64,
    mutation_span: f64,
    pub(crate) population_size: usize,
    pub(crate) opti_function: F,
    dimension: usize,
    lower_bound: Vec<i32>,
    upper_bound: Vec<i32>,
}

impl<F: OptimizationFn> GeneticAlgorithm<F> {
    pub(crate) fn new(
        opti_function: F,
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        mutation_span: f64,
        dimension: usize,
        lower_bound: Vec<i32>,
        upper_bound: Vec<i32>,
    ) -> Self {
        Self {
            mutation_rate,
            crossover_rate,
            mutation_span,
            population_size,
            opti_function,
            dimension,
            lower_bound,
            upper_bound,
        }
    }

    pub(crate) fn generate_new_population(&self, seed: u64) -> Vec<Arm> {
        let mut individuals: Vec<Arm> = Vec::new();
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

        while individuals.len() < self.population_size {
            let candidate_solution: Vec<i32> = (0..self.dimension)
                .map(|j| rng.random_range(self.lower_bound[j]..=self.upper_bound[j]))
                .collect();

            let candidate_arm = Arm::new(&candidate_solution);

            if !individuals.contains(&candidate_arm) {
                individuals.push(candidate_arm);
            }
        }
        individuals
    }

    pub(crate) fn crossover(&self, seed: u64, population: &[Arm]) -> Vec<Arm> {
        let mut crossover_pop: Vec<Arm> = Vec::new();
        let population_size = self.population_size;
        let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

        for i in (0..population_size).step_by(2) {
            if rng.random::<f64>() < self.crossover_rate {
                // Crossover
                let max_dim_index = self.dimension - 1;
                let swap_rv = rng.random_range(1..=max_dim_index);

                for j in 1..=max_dim_index {
                    if swap_rv == j {
                        let mut cross_vec_1: Vec<i32> =
                            population[i].get_action_vector()[0..j].to_vec();
                        cross_vec_1.extend_from_slice(
                            &population[i + 1].get_action_vector()[j..=max_dim_index],
                        );

                        let mut cross_vec_2: Vec<i32> =
                            population[i + 1].get_action_vector()[0..j].to_vec();
                        cross_vec_2.extend_from_slice(
                            &population[i].get_action_vector()[j..=max_dim_index],
                        );

                        let new_individual_1 = Arm::new(&cross_vec_1);
                        let new_individual_2 = Arm::new(&cross_vec_2);

                        crossover_pop.push(new_individual_1);
                        crossover_pop.push(new_individual_2);
                    }
                }
            } else {
                // No Crossover
                crossover_pop.push(population[i].clone());
                crossover_pop.push(population[i + 1].clone());
            }
        }

        crossover_pop
    }

    pub(crate) fn mutate(&self, seed: u64, population: &[Arm]) -> Vec<Arm> {
        let mut mutated_population = Vec::new();
        let mut seen = HashSet::new();
        let mut rng = StdRng::seed_from_u64(seed);

        for individual in population.iter() {
            // Clone the action vector
            let mut new_action_vector = individual.get_action_vector().to_vec(); // Here I assumed `get_action_vector` returns a slice or Vec

            for (i, value) in new_action_vector.iter_mut().enumerate() {
                if rng.random::<f64>() < self.mutation_rate {
                    let adjustment = Normal::new(
                        0.0,
                        self.mutation_span * (self.upper_bound[i] - self.lower_bound[i]) as f64,
                    )
                    .unwrap()
                    .sample(&mut rng);

                    *value = (*value as f64 + adjustment)
                        .max(self.lower_bound[i] as f64)
                        .min(self.upper_bound[i] as f64) as i32;
                }
            }

            let new_individual = Arm::new(new_action_vector.as_slice());

            if seen.insert(new_individual.clone()) {
                mutated_population.push(new_individual);
            }
        }

        mutated_population
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SEED: u64 = 42;

    // Mock optimization function for testing
    fn mock_opti_function(_vec: &[i32]) -> f64 {
        0.0
    }

    #[test]
    fn test_get_population_size() {
        let ga = GeneticAlgorithm::new(
            mock_opti_function,
            10,
            0.1,
            0.9,
            0.5,
            2,
            vec![0, 0],
            vec![10, 10],
        );
        assert_eq!(ga.population_size, 10);
    }

    #[test]
    fn test_mutate() {
        let ga = GeneticAlgorithm::new(
            mock_opti_function,
            2,   // Two individuals in population
            1.0, // 100% mutation rate for demonstration
            0.9,
            1.0,
            2,
            vec![0, 0],
            vec![10, 10],
        );

        let initial_population = vec![Arm::new(&vec![1, 1]), Arm::new(&vec![2, 2])];

        let mutated_population = ga.mutate(SEED, &initial_population);

        // Assuming the mutation is deterministic and in the expected bounds, you'd check like this:
        for (i, individual) in mutated_population.iter().enumerate() {
            let init_vector = initial_population[i].get_action_vector();
            let mut_vector = individual.get_action_vector();

            for j in 0..ga.dimension {
                assert!(mut_vector[j] >= ga.lower_bound[j]);
                assert!(mut_vector[j] <= ga.upper_bound[j]);
            }

            assert_ne!(mut_vector, init_vector); // since mutation rate is 100%
        }
    }

    #[test]
    fn test_crossover() {
        let ga = GeneticAlgorithm::new(
            mock_opti_function,
            2, // Two individuals for simplicity
            0.1,
            1.0, // 100% crossover rate for demonstration
            0.5,
            10, // higher dimension for demonstration so low probability of crossover leading to identical individuals
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            vec![10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        );

        let initial_population = vec![
            Arm::new(&vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            Arm::new(&vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
        ];

        let crossover_population = ga.crossover(SEED, &initial_population);

        // Since the crossover rate is 100%, the two individuals should not be identical to the original individuals
        assert_ne!(
            crossover_population[0].get_action_vector(),
            initial_population[0].get_action_vector()
        );
        assert_ne!(
            crossover_population[1].get_action_vector(),
            initial_population[1].get_action_vector()
        );
    }

    #[test]
    fn test_reproduction_with_seeding() {
        // Helper function that generates and modifies a population using a seed.
        fn generate_population(seed: u64) -> Vec<Arm> {
            let ga = GeneticAlgorithm::new(
                mock_opti_function,
                10,
                0.1,
                0.9,
                0.5,
                2,
                vec![0, 0],
                vec![10, 10],
            );

            let mut population = ga.generate_new_population(seed);
            population = ga.crossover(seed, &population);
            population = ga.mutate(seed, &population);

            return population;
        }

        // The same seed should lead to the same population
        assert_eq!(generate_population(SEED), generate_population(SEED));

        // A different seed should not lead to the same population
        assert_ne!(generate_population(SEED), generate_population(SEED + 1));
    }
}
