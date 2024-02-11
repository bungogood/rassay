use std::marker::PhantomData;

use crate::probabilities::Probabilities;
use bkgm::State;

use super::{Evaluator, PartialEvaluator};

pub struct GreedyEvaluator<G: State, E: Evaluator<G>> {
    phantom: PhantomData<G>,
    evaluator: E,
    epsilon: f32,
}

impl<G: State, E: Evaluator<G>> PartialEvaluator<G> for GreedyEvaluator<G, E> {
    fn try_eval(&self, pos: &G) -> f32 {
        self.evaluator.try_eval(pos)
    }

    fn best_position(&self, pos: &G, dice: &bkgm::Dice) -> G {
        if fastrand::f32() > self.epsilon {
            self.evaluator.best_position(pos, dice)
        } else {
            let positions = pos.possible_positions(dice);
            positions[fastrand::usize(..positions.len())]
        }
    }
}

impl<G: State, E: Evaluator<G>> Evaluator<G> for GreedyEvaluator<G, E> {
    fn eval(&self, pos: &G) -> Probabilities {
        self.evaluator.eval(pos)
    }
}

impl<G: State, E: Evaluator<G>> GreedyEvaluator<G, E> {
    pub fn new(evaluator: E, epsilon: f32) -> Self {
        Self {
            phantom: PhantomData,
            evaluator,
            epsilon,
        }
    }
}
