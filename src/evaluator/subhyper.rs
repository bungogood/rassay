use bkgm::{utils::mcomb, Position, State};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use super::PartialEvaluator;

const POSSIBLE: usize = mcomb(26, 3).pow(2);

#[derive(Clone)]
pub struct SubHyperEvaluator {
    values: Vec<f32>,
}

impl PartialEvaluator<Position<3>> for SubHyperEvaluator {
    fn try_eval(&self, pos: &Position<3>) -> f32 {
        self.values[pos.dbhash()]
    }
}

impl SubHyperEvaluator {
    pub fn from_file(file_path: impl AsRef<Path>) -> Option<Self> {
        let file = File::open(file_path).expect("File not found");

        let mut reader = BufReader::new(file);

        let mut buffer = [0; 4];
        let mut values = Vec::new();

        while reader.read_exact(&mut buffer).is_ok() {
            let value = f32::from_le_bytes(buffer.try_into().unwrap());
            values.push(value);
        }

        if values.len() == POSSIBLE {
            Some(Self { values })
        } else {
            None
        }
    }
}
