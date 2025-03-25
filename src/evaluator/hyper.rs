use crate::evaluator::Evaluator;
use crate::probabilities::Probabilities;
use bkgm::{utils::mcomb, State};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::PartialEvaluator;

const POSSIBLE: usize = mcomb(26, 3 as usize).pow(2);

#[derive(Clone)]
pub struct HyperEvaluator {
    probs: Vec<Probabilities>,
}

impl<G: State> PartialEvaluator<G> for HyperEvaluator {
    fn try_eval(&self, pos: &G) -> f32 {
        let probs = self.eval(pos);
        probs.equity()
    }
}

impl<G: State> Evaluator<G> for HyperEvaluator {
    fn eval(&self, position: &G) -> Probabilities {
        self.probs[position.dbhash()]
    }
}

impl HyperEvaluator {
    pub fn new() -> Option<Self> {
        Self::from_file("data/hyper.db")
    }

    pub fn from_file(file_path: impl AsRef<Path>) -> Option<Self> {
        let file = File::open(file_path).expect("File not found");

        let mut reader = BufReader::new(file);

        let mut buffer = [0u8; 20];
        let mut probs = Vec::new();

        while reader.read_exact(&mut buffer).is_ok() {
            let wgbgb = [
                f32::from_le_bytes(buffer[0..4].try_into().unwrap()),
                f32::from_le_bytes(buffer[4..8].try_into().unwrap()),
                f32::from_le_bytes(buffer[8..12].try_into().unwrap()),
                f32::from_le_bytes(buffer[12..16].try_into().unwrap()),
                f32::from_le_bytes(buffer[16..20].try_into().unwrap()),
            ];

            probs.push(Probabilities {
                win_n: wgbgb[0] - wgbgb[1],
                win_g: wgbgb[1] - wgbgb[2],
                win_b: wgbgb[2],
                lose_n: 1.0 - wgbgb[0] - wgbgb[3],
                lose_g: wgbgb[3] - wgbgb[4],
                lose_b: wgbgb[4],
            });
        }

        if probs.len() == POSSIBLE {
            Some(Self { probs })
        } else {
            None
        }
    }
}
