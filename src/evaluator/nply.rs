use std::marker::PhantomData;

use bkgm::{dice::ALL_21, GameState::GameOver, State};

use super::{Evaluator, PartialEvaluator};

pub struct PlyEvaluator<G: State, E: Evaluator<G>> {
    phantom: PhantomData<G>,
    evaluator: E,
    depth: usize,
    top_k: Option<usize>,
}

impl<G: State, E: Evaluator<G>> PartialEvaluator<G> for PlyEvaluator<G, E> {
    fn try_eval(&self, pos: &G) -> f32 {
        self.ply(pos, self.depth)
    }

    fn best_position(&self, pos: &G, dice: &bkgm::Dice) -> G {
        *pos.possible_positions(dice)
            .iter()
            .map(|pos| (pos, self.ply(pos, self.depth - 1)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

// impl<G: State, E: Evaluator<G>> Evaluator<G> for PlyEvaluator<G, E> {
//     fn eval(&self, pos: &G) -> Probabilities {
//         self.evaluator.eval(pos)
//         for _ in 0..self.num_rollouts {
//             let mut pos = pos.clone();
//             let mut dice = bkgm::Dice::new();
//             while !pos.is_game_over() {
//                 let best_pos = self.best_position(&pos, &dice);
//                 pos = best_pos;
//                 dice = bkgm::Dice::new();
//             }
//             let winner = pos.winner();
//             if winner == bkgm::Player::White {
//                 white_wins += 1;
//             } else {
//                 black_wins += 1;
//             }
//         }
//     }
// }

impl<G: State, E: Evaluator<G>> PlyEvaluator<G, E> {
    pub fn new(evaluator: E, depth: usize) -> Self {
        assert!(depth > 0, "depth must be greater than 0");
        Self {
            phantom: PhantomData,
            evaluator,
            depth,
            top_k: None,
        }
    }

    fn ply(&self, pos: &G, depth: usize) -> f32 {
        if depth == 0 {
            return self.evaluator.try_eval(pos);
        } else if let GameOver(result) = pos.game_state() {
            return result.value();
        }
        let mut result = 0.0;
        for (dice, prob) in ALL_21 {
            let mut best_value = f32::NEG_INFINITY;
            for pos in pos.possible_positions(&dice) {
                let value = -self.ply(&pos, depth - 1);
                if value > best_value {
                    best_value = value;
                }
            }
            result += prob * best_value;
        }
        result / 36.0
    }
}
