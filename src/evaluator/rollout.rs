use std::marker::PhantomData;

use crate::probabilities::{Probabilities, ResultCounter};
use bkgm::dice_gen::{DiceGen, FastrandDice};
use bkgm::position::GameState::{GameOver, Ongoing};
use bkgm::{GameResult, State};

use super::{Evaluator, PartialEvaluator};

pub struct RolloutEvaluator<G: State, E: Evaluator<G>> {
    phantom: PhantomData<G>,
    evaluator: E,
    num_rollouts: usize,
}

impl<G: State, E: Evaluator<G>> PartialEvaluator<G> for RolloutEvaluator<G, E> {
    fn try_eval(&self, pos: &G) -> f32 {
        self.eval(pos).equity()
    }
}

impl<G: State, E: Evaluator<G>> Evaluator<G> for RolloutEvaluator<G, E> {
    fn eval(&self, pos: &G) -> Probabilities {
        self.rollout(pos)
    }
}

impl<G: State, E: Evaluator<G>> RolloutEvaluator<G, E> {
    pub fn new(evaluator: E, num_rollouts: usize) -> Self {
        Self {
            phantom: PhantomData,
            evaluator,
            num_rollouts,
        }
    }

    fn rollout(&self, pos: &G) -> Probabilities {
        let mut dice_gen = FastrandDice::new();
        let mut counter = ResultCounter::default();
        for _ in 0..self.num_rollouts {
            let result = self.single_rollout(&mut dice_gen, pos);
            counter.add(result);
        }
        counter.probabilities()
    }

    fn single_rollout<V: DiceGen>(&self, dice_gen: &mut V, pos: &G) -> GameResult {
        let mut pos = pos.clone();
        let mut depth = 0;
        loop {
            match pos.game_state() {
                Ongoing => {
                    let dice = dice_gen.roll();
                    pos = self.evaluator.best_position(&pos, &dice);
                    depth += 1;
                }
                GameOver(result) => {
                    return if depth % 2 == 0 {
                        result
                    } else {
                        result.reverse()
                    };
                }
            }
        }
    }
}
