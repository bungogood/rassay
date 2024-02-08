use crate::probabilities::Probabilities;
use bkgm::{Dice, State};
use burn::tensor::backend::Backend;
use std::path::Path;

pub trait PartialEvaluator<G: State>: Sized {
    /// Returns a cubeless evaluation of a position.
    /// Implementing types will calculate the probabilities with different strategies.
    /// Examples of such strategies are a rollout or 1-ply inference of a neural net.
    fn try_eval(&self, pos: &G) -> f32;

    fn best_position(&self, pos: &G, dice: &Dice) -> G {
        *pos.possible_positions(dice)
            .iter()
            .map(|pos| (pos, self.try_eval(&pos)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

pub trait Evaluator<G: State>: PartialEvaluator<G> + Sized {
    /// Returns a cubeless evaluation of a position.
    /// Implementing types will calculate the probabilities with different strategies.
    /// Examples of such strategies are a rollout or 1-ply inference of a neural net.
    fn eval(&self, pos: &G) -> Probabilities;
}

pub trait ONNEvaluator<B: Backend, G: State>: Evaluator<G> + Sized {
    const MODEL_PATH: &'static str;
    const NUM_INTPUTS: usize;
    const NUM_OUTPUTS: usize;

    fn with_default_model() -> Option<Self> {
        Self::from_file_path(Self::MODEL_PATH)
    }

    fn from_file_path(file_path: impl AsRef<Path>) -> Option<Self>;

    fn input_labels(&self) -> Vec<String> {
        (0..Self::NUM_INTPUTS)
            .map(|i| format!("input_{}", i))
            .collect()
    }

    fn output_labels(&self) -> Vec<String> {
        (0..Self::NUM_OUTPUTS)
            .map(|i| format!("output_{}", i))
            .collect()
    }

    fn input_vec(&self, position: &G) -> Vec<f32>;
    fn output_vec(&self, position: &G) -> Vec<f32>;
}

pub struct RandomEvaluator;

impl<G: State> PartialEvaluator<G> for RandomEvaluator {
    fn try_eval(&self, pos: &G) -> f32 {
        let probs = self.eval(pos);
        probs.equity()
    }
}

impl<G: State> Evaluator<G> for RandomEvaluator {
    #[allow(dead_code)]
    /// Returns random probabilities. Each call will return different values.
    fn eval(&self, _pos: &G) -> Probabilities {
        let win_n = fastrand::f32();
        let win_g = fastrand::f32();
        let win_b = fastrand::f32();
        let lose_n = fastrand::f32();
        let lose_g = fastrand::f32();
        let lose_b = fastrand::f32();

        // Now we like to make sure that the different probabilities add up to 1
        let sum = win_n + win_g + win_b + lose_n + lose_g + lose_b;
        Probabilities {
            win_n: win_n / sum,
            win_g: win_g / sum,
            win_b: win_b / sum,
            lose_n: lose_n / sum,
            lose_g: lose_g / sum,
            lose_b: lose_b / sum,
        }
    }
}

impl RandomEvaluator {
    pub fn new() -> RandomEvaluator {
        #[allow(dead_code)]
        RandomEvaluator {}
    }
}
