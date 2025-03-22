use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::dicegen::{DiceGen, FastrandDice};
use crate::evaluator::PartialEvaluator;
use crate::probabilities::{Probabilities, ResultCounter};
use bkgm::GameState::{GameOver, Ongoing};
use bkgm::State;
use indicatif::ProgressIterator;

pub struct Duel<T: PartialEvaluator<G>, U: PartialEvaluator<G>, G: State> {
    evaluator1: T,
    evaluator2: U,
    phantom: PhantomData<G>,
}

pub fn duel<G: State>(
    evaluator1: impl PartialEvaluator<G>,
    evaluator2: impl PartialEvaluator<G>,
    rounds: usize,
) -> Probabilities {
    let duel = Duel::new(evaluator1, evaluator2);
    let mut results = ResultCounter::default();
    let mut unique = std::collections::HashSet::new();
    let mut game_length = std::collections::HashMap::new();
    let mut captures = 0;
    for round in (0..rounds).progress() {
        let outcome = duel.single_duel(
            &mut FastrandDice::new(),
            &mut unique,
            &mut game_length,
            &mut captures,
        );
        results = results.combine(&outcome);
        let probs = results.probabilities();
    }
    results.probabilities()
}

/// Let two `PartialEvaluator`s duel each other. A bit quick and dirty.
impl<T: PartialEvaluator<G>, U: PartialEvaluator<G>, G: State> Duel<T, U, G> {
    #[allow(clippy::new_without_default)]
    pub fn new(evaluator1: T, evaluator2: U) -> Self {
        Duel {
            evaluator1,
            evaluator2,
            phantom: PhantomData,
        }
    }

    // pub fn

    /// The two `PartialEvaluator`s will play twice each against each other.
    /// Either `PartialEvaluator` will start once and play with the same dice as vice versa.
    pub fn single_duel<V: DiceGen>(
        &self,
        dice_gen: &mut V,
        unique: &mut HashSet<usize>,
        game_length: &mut HashMap<usize, usize>,
        captures: &mut u32,
    ) -> ResultCounter {
        let mut pos1 = G::new();
        let mut pos2 = G::new();
        let mut iteration = 1;
        let mut pos1_finished = false;
        let mut pos2_finished = false;
        let mut counter = ResultCounter::default();
        while !(pos1_finished && pos2_finished) {
            let dice = dice_gen.roll();
            match pos1.game_state() {
                Ongoing => {
                    let new_pos = if iteration % 2 == 0 {
                        self.evaluator1.best_position(&pos1, &dice)
                    } else {
                        self.evaluator2.best_position(&pos1, &dice)
                    };

                    if pos1.o_bar() == 0 && new_pos.x_bar() > 0 {
                        *captures += new_pos.x_bar() as u32;
                    }
                    pos1 = new_pos;
                    // pos1 = if iteration % 2 == 0 {
                    //     self.evaluator1.best_position(&pos1, &dice)
                    // } else {
                    //     self.evaluator2.best_position(&pos1, &dice)
                    // };
                    // unique.insert(pos1.dbhash());
                }
                GameOver(result) => {
                    if !pos1_finished {
                        let cur = game_length.get(&iteration).unwrap_or(&0);
                        game_length.insert(iteration, cur + 1);
                        pos1_finished = true;
                        let result = if iteration % 2 == 0 {
                            result
                        } else {
                            result.reverse()
                        };
                        counter.add(result);
                    }
                }
            }
            match pos2.game_state() {
                Ongoing => {
                    let new_pos = if iteration % 2 == 0 {
                        self.evaluator2.best_position(&pos2, &dice)
                    } else {
                        self.evaluator1.best_position(&pos2, &dice)
                    };

                    if pos2.o_bar() == 0 && new_pos.x_bar() > 0 {
                        *captures += new_pos.x_bar() as u32;
                    }
                    pos2 = new_pos;
                    // pos2 = if iteration % 2 == 0 {
                    //     self.evaluator2.best_position(&pos2, &dice)
                    // } else {
                    //     self.evaluator1.best_position(&pos2, &dice)
                    // };
                    // unique.insert(pos2.dbhash());
                }
                GameOver(result) => {
                    if !pos2_finished {
                        let cur = game_length.get(&iteration).unwrap_or(&0);
                        game_length.insert(iteration, cur + 1);
                        pos2_finished = true;
                        let result = if iteration % 2 == 0 {
                            result.reverse()
                        } else {
                            result
                        };
                        counter.add(result);
                    }
                }
            }
            iteration += 1;
        }
        debug_assert!(counter.sum() == 2, "Each duel should have two game results");
        counter
    }
}
