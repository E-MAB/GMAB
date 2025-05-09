use std::hash::{Hash, Hasher};

pub trait OptimizationFn {
    fn evaluate(&self, action_vector: &[i32]) -> f64;
}

impl<F: Fn(&[i32]) -> f64> OptimizationFn for F {
    fn evaluate(&self, action_vector: &[i32]) -> f64 {
        self(action_vector)
    }
}

#[derive(Debug)]
pub struct Arm {
    action_vector: Vec<i32>,
    reward: f64,
    num_pulls: i32,
}

impl Arm {
    pub fn new(action_vector: &[i32]) -> Self {
        Self {
            reward: 0.0,
            num_pulls: 0,
            action_vector: action_vector.to_vec(),
        }
    }

    pub(crate) fn pull<F: OptimizationFn>(&mut self, opt_fn: &F) -> f64 {
        let g = opt_fn.evaluate(&self.action_vector);

        self.reward += g;
        self.num_pulls += 1;

        g
    }

    pub fn get_num_pulls(&self) -> i32 {
        self.num_pulls
    }

    pub(crate) fn get_function_value<F: OptimizationFn>(&self, opt_fn: &F) -> f64 {
        opt_fn.evaluate(&self.action_vector)
    }

    pub fn get_action_vector(&self) -> &[i32] {
        &self.action_vector
    }

    pub fn get_mean_reward(&self) -> f64 {
        if self.num_pulls == 0 {
            return 0.0;
        }
        self.reward / self.num_pulls as f64
    }
}

impl Clone for Arm {
    fn clone(&self) -> Self {
        Self {
            action_vector: self.action_vector.clone(),
            reward: self.reward,
            num_pulls: self.num_pulls,
        }
    }
}

impl PartialEq for Arm {
    fn eq(&self, other: &Self) -> bool {
        self.action_vector == other.action_vector
    }
}

impl Eq for Arm {}

impl Hash for Arm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.action_vector.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock optimization function for testing
    fn mock_opti_function(_vec: &[i32]) -> f64 {
        5.0
    }

    #[test]
    fn test_arm_new() {
        let arm = Arm::new(&vec![1, 2]);
        assert_eq!(arm.get_num_pulls(), 0);
        assert_eq!(arm.get_function_value(&mock_opti_function), 5.0);
    }

    #[test]
    fn test_arm_pull() {
        let mut arm = Arm::new(&vec![1, 2]);
        let reward = arm.pull(&mock_opti_function);

        assert_eq!(reward, 5.0);
        assert_eq!(arm.get_num_pulls(), 1);
        assert_eq!(arm.get_mean_reward(), 5.0);
    }

    #[test]
    fn test_arm_pull_multiple() {
        let mut arm = Arm::new(&vec![1, 2]);
        arm.pull(&mock_opti_function);
        arm.pull(&mock_opti_function);

        assert_eq!(arm.get_num_pulls(), 2);
        assert_eq!(arm.get_mean_reward(), 5.0); // Since reward is always 5.0
    }

    #[test]
    fn test_arm_clone() {
        let arm = Arm::new(&vec![1, 2]);
        let cloned_arm = arm.clone();

        assert_eq!(arm.get_num_pulls(), cloned_arm.get_num_pulls());
        assert_eq!(
            arm.get_function_value(&mock_opti_function),
            cloned_arm.get_function_value(&mock_opti_function)
        );
        assert_eq!(arm.get_action_vector(), cloned_arm.get_action_vector());
    }

    #[test]
    fn test_initial_reward_is_zero() {
        let arm = Arm::new(&vec![1, 2]);
        assert_eq!(arm.get_mean_reward(), 0.0);
    }

    #[test]
    fn test_mean_reward_with_zero_pulls() {
        let arm = Arm::new(&vec![1, 2]);
        assert_eq!(arm.get_mean_reward(), 0.0);
    }

    #[test]
    fn test_clone_after_pulls() {
        let mut arm = Arm::new(&vec![1, 2]);
        arm.pull(&mock_opti_function);
        let cloned_arm = arm.clone();
        assert_eq!(arm.get_num_pulls(), cloned_arm.get_num_pulls());
        assert_eq!(arm.get_mean_reward(), cloned_arm.get_mean_reward());
    }
}
