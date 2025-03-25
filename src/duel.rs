use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

use crate::evaluator::PartialEvaluator;
use crate::probabilities::{Probabilities, ResultCounter};
use bkgm::dice_gen::{DiceGen, FastrandDice};
use bkgm::position::{GamePhase, OngoingPhase};
use bkgm::GameState::{self, GameOver, Ongoing};
use bkgm::State;
use indicatif::ProgressIterator;

pub struct Duel<T: PartialEvaluator<G>, U: PartialEvaluator<G>, G: State> {
    evaluator1: T,
    evaluator2: U,
    phantom: PhantomData<G>,
}

pub fn duel<G: State>(
    state: &G,
    evaluator1: impl PartialEvaluator<G>,
    evaluator2: impl PartialEvaluator<G>,
    rounds: usize,
) -> Probabilities {
    let duel = Duel::new(evaluator1, evaluator2);
    let mut results = ResultCounter::default();
    let mut unique = std::collections::HashSet::new();
    let mut game_length = std::collections::HashMap::new();
    let mut phases = std::collections::HashMap::new();
    let mut captures = 0;
    let mut possible = 0;
    for round in (0..rounds).progress() {
        let outcome = duel.single_duel(
            state,
            &mut FastrandDice::new(),
            &mut unique,
            &mut game_length,
            &mut phases,
            &mut captures,
            &mut possible,
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
        state: &G,
        dice_gen: &mut V,
        unique: &mut HashSet<G>,
        game_length: &mut HashMap<usize, usize>,
        phases: &mut HashMap<OngoingPhase, usize>,
        captures: &mut u32,
        possible: &mut u32,
    ) -> ResultCounter {
        let mut pos1 = state.clone();
        let mut pos2 = state.clone();
        let mut iteration = 1;
        let mut pos1_finished = false;
        let mut pos2_finished = false;
        let mut counter = ResultCounter::default();
        let mut dice = dice_gen.roll_mixed();
        while !(pos1_finished && pos2_finished) {
            match pos1.game_state() {
                GameState::Ongoing => {
                    // let cur = phases.get(&phase).unwrap_or(&0);
                    // phases.insert(phase, cur + 1);
                    // *possible += pos1.possible_positions(&dice).len() as u32;
                    let new_pos = if iteration % 2 == 0 {
                        self.evaluator1.best_position(&pos1, &dice)
                    } else {
                        self.evaluator2.best_position(&pos1, &dice)
                    };

                    // *captures += new_pos.x_bar().saturating_sub(pos1.o_bar()) as u32;
                    // unique.insert(new_pos);
                    pos1 = new_pos;
                }
                GameState::GameOver(result) => {
                    if !pos1_finished {
                        // let cur = game_length.get(&iteration).unwrap_or(&0);
                        // game_length.insert(iteration, cur + 1);
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
                GameState::Ongoing => {
                    // let cur = phases.get(&phase).unwrap_or(&0);
                    // phases.insert(phase, cur + 1);
                    // *possible += pos2.possible_positions(&dice).len() as u32;
                    let new_pos = if iteration % 2 == 0 {
                        self.evaluator2.best_position(&pos2, &dice)
                    } else {
                        self.evaluator1.best_position(&pos2, &dice)
                    };

                    // *captures += new_pos.x_bar().saturating_sub(pos2.o_bar()) as u32;
                    // unique.insert(new_pos);
                    pos2 = new_pos;
                }
                GameState::GameOver(result) => {
                    if !pos2_finished {
                        // let cur = game_length.get(&iteration).unwrap_or(&0);
                        // game_length.insert(iteration, cur + 1);
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
            dice = dice_gen.roll();
            iteration += 1;
        }
        debug_assert!(counter.sum() == 2, "Each duel should have two game results");
        counter
    }
}
