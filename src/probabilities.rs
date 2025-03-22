use std::fmt;

use bkgm::GameResult;

/// Sum of all six fields will always be 1.0
#[derive(PartialEq, Clone, Copy)]
pub struct Probabilities {
    pub win_n: f32,
    pub win_g: f32,
    pub win_b: f32,
    pub lose_n: f32,
    pub lose_g: f32,
    pub lose_b: f32,
}

// impl<'de> Deserialize<'de> for Probabilities {
//     fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         struct ProbabilitiesVisitor;

//         impl<'de> Visitor<'de> for ProbabilitiesVisitor {
//             type Value = Probabilities;

//             fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
//                 formatter.write_str("a struct representing probabilities")
//             }

//             fn visit_seq<V>(self, mut seq: V) -> Result<Probabilities, V::Error>
//             where
//                 V: serde::de::SeqAccess<'de>,
//             {
//                 let win: f32 = seq
//                     .next_element()?
//                     .ok_or_else(|| de::Error::invalid_length(0, &self))?;
//                 let win_gb: f32 = seq
//                     .next_element()?
//                     .ok_or_else(|| de::Error::invalid_length(1, &self))?;
//                 let win_b: f32 = seq
//                     .next_element()?
//                     .ok_or_else(|| de::Error::invalid_length(2, &self))?;
//                 let lose_gb: f32 = seq
//                     .next_element()?
//                     .ok_or_else(|| de::Error::invalid_length(3, &self))?;
//                 let lose_b: f32 = seq
//                     .next_element()?
//                     .ok_or_else(|| de::Error::invalid_length(4, &self))?;

//                 if win < win_gb {
//                     return Err(de::Error::custom(
//                         "The probability of winning a normal game must be greater than the probability of winning a gammon",
//                     ));
//                 }

//                 if win_gb < win_b {
//                     return Err(de::Error::custom(
//                         "The probability of winning a gammon must be greater than the probability of winning a backgammon",
//                     ));
//                 }

//                 if lose_gb < lose_b {
//                     return Err(de::Error::custom(
//                         "The probability of losing a gammon must be greater than the probability of losing a backgammon",
//                     ));
//                 }

//                 if win < 0.0 || win_gb < 0.0 || win_b < 0.0 || lose_gb < 0.0 || lose_b < 0.0 {
//                     return Err(de::Error::custom(
//                         "None of the probabilities can be negative",
//                     ));
//                 }

//                 if win + lose_gb > 1.0 + f32::EPSILON {
//                     return Err(de::Error::custom(
//                         "The sum of all probabilities must equal 1.0",
//                     ));
//                 }

//                 Ok(Probabilities {
//                     win_n: win - win_gb,
//                     win_g: win_gb - win_b,
//                     win_b,
//                     lose_n: 1.0 - win - lose_gb,
//                     lose_g: lose_gb - lose_b,
//                     lose_b,
//                 })
//             }
//         }

//         const FIELDS: &'static [&'static str] = &["win", "win_g", "win_b", "lose_g", "lose_b"];
//         deserializer.deserialize_struct("Probabilities", FIELDS, ProbabilitiesVisitor)
//     }
// }

impl fmt::Debug for Probabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Probabilities: wn {:.2}%; wg {:.2}%; wb {:.2}%; ln {:.2}%; lg {:.2}%; lb {:.2}%",
            100.0 * self.win_n,
            100.0 * self.win_g,
            100.0 * self.win_b,
            100.0 * self.lose_n,
            100.0 * self.lose_g,
            100.0 * self.lose_b
        )
    }
}

impl Probabilities {
    /// Typically used from rollouts.
    /// The index within the array has to correspond to the discriminant of the `Probabilities` enum.
    /// Input integer values will be normalized so that the sum in the return value is 1.0
    pub fn new(results: &[u32; 6]) -> Self {
        let sum = results.iter().sum::<u32>() as f32;
        Probabilities {
            win_n: results[GameResult::WinNormal as usize] as f32 / sum,
            win_g: results[GameResult::WinGammon as usize] as f32 / sum,
            win_b: results[GameResult::WinBackgammon as usize] as f32 / sum,
            lose_n: results[GameResult::LoseNormal as usize] as f32 / sum,
            lose_g: results[GameResult::LoseGammon as usize] as f32 / sum,
            lose_b: results[GameResult::LoseBackgammon as usize] as f32 / sum,
        }
    }

    pub fn empty() -> Self {
        Probabilities {
            win_n: 0.0,
            win_g: 0.0,
            win_b: 0.0,
            lose_n: 0.0,
            lose_g: 0.0,
            lose_b: 0.0,
        }
    }

    pub fn win_prob(&self) -> f32 {
        self.win_n + self.win_g + self.win_b
    }

    pub fn from_result(results: &GameResult) -> Self {
        match results {
            GameResult::WinNormal => Self {
                win_n: 1.0,
                win_g: 0.0,
                win_b: 0.0,
                lose_n: 0.0,
                lose_g: 0.0,
                lose_b: 0.0,
            },
            GameResult::WinGammon => Self {
                win_n: 0.0,
                win_g: 1.0,
                win_b: 0.0,
                lose_n: 0.0,
                lose_g: 0.0,
                lose_b: 0.0,
            },
            GameResult::WinBackgammon => Self {
                win_n: 0.0,
                win_g: 0.0,
                win_b: 1.0,
                lose_n: 0.0,
                lose_g: 0.0,
                lose_b: 0.0,
            },
            GameResult::LoseNormal => Self {
                win_n: 0.0,
                win_g: 0.0,
                win_b: 0.0,
                lose_n: 1.0,
                lose_g: 0.0,
                lose_b: 0.0,
            },
            GameResult::LoseGammon => Self {
                win_n: 0.0,
                win_g: 0.0,
                win_b: 0.0,
                lose_n: 0.0,
                lose_g: 1.0,
                lose_b: 0.0,
            },
            GameResult::LoseBackgammon => Self {
                win_n: 0.0,
                win_g: 0.0,
                win_b: 0.0,
                lose_n: 0.0,
                lose_g: 0.0,
                lose_b: 1.0,
            },
        }
    }

    pub fn normalized(&self) -> Self {
        let sum = self.to_vec().iter().sum::<f32>();
        Probabilities {
            win_n: self.win_n / sum,
            win_g: self.win_g / sum,
            win_b: self.win_b / sum,
            lose_n: self.lose_n / sum,
            lose_g: self.lose_g / sum,
            lose_b: self.lose_b / sum,
        }
    }

    pub fn flip(&self) -> Self {
        Self {
            win_n: self.lose_n,
            win_g: self.lose_g,
            win_b: self.lose_b,
            lose_n: self.win_n,
            lose_g: self.win_g,
            lose_b: self.win_b,
        }
    }

    /// Cubeless equity
    pub fn equity(&self) -> f32 {
        self.win_n - self.lose_n
            + 2.0 * (self.win_g - self.lose_g)
            + 3.0 * (self.win_b - self.lose_b)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        Vec::from(self.to_slice())
    }

    pub fn to_slice(&self) -> [f32; 6] {
        [
            self.win_n,
            self.win_g,
            self.win_b,
            self.lose_n,
            self.lose_g,
            self.lose_b,
        ]
    }

    pub fn to_gnu(&self) -> [f32; 5] {
        let win_g = self.win_g + self.win_b;
        let lose_g = self.lose_g + self.lose_b;
        [self.win_n + win_g, win_g, self.win_b, lose_g, self.lose_b]
    }
}

impl From<&[f32; 5]> for Probabilities {
    /// Typically used from rollouts.
    fn from(value: &[f32; 5]) -> Self {
        let win_b = value[2];
        let lose_b = value[4];
        let win_g = value[1] - win_b;
        let lose_g = value[3] - lose_b;
        let win_n = value[0] - value[1];
        let lose_n = 1.0 - value[0] - value[3];

        Probabilities {
            win_n,
            win_g,
            win_b,
            lose_n,
            lose_g,
            lose_b,
        }
    }
}

#[derive(Default)]
pub struct ResultCounter {
    results: [u32; 6],
}

impl ResultCounter {
    /// Convenience method, mainly for tests
    pub fn new(win_n: u32, win_g: u32, win_b: u32, lose_n: u32, lose_g: u32, lose_b: u32) -> Self {
        let results = [win_n, win_g, win_b, lose_n, lose_g, lose_b];
        Self { results }
    }
    pub fn add(&mut self, result: GameResult) {
        self.results[result as usize] += 1;
    }

    pub fn add_results(&mut self, result: GameResult, amount: u32) {
        self.results[result as usize] += amount;
    }

    pub fn sum(&self) -> u32 {
        self.results.iter().sum::<u32>()
    }

    pub fn num_of(&self, result: GameResult) -> u32 {
        // This works because the enum has associated integer values (discriminant), starting with zero.
        self.results[result as usize]
    }

    pub fn combine(self, counter: &ResultCounter) -> Self {
        let mut results = self.results;
        for (self_value, counter_value) in results.iter_mut().zip(counter.results) {
            *self_value += counter_value;
        }
        Self { results }
    }

    pub fn probabilities(&self) -> Probabilities {
        Probabilities::new(&self.results)
    }
}

#[cfg(test)]
mod probabilities_tests {
    use crate::probabilities::Probabilities;

    #[test]
    fn new() {
        // sum of `results is 32, a power of 2. Makes fractions easier to handle.
        let results = [0_u32, 1, 3, 4, 8, 16];
        let probabilities = Probabilities::new(&results);
        assert_eq!(probabilities.win_n, 0.0);
        assert_eq!(probabilities.win_g, 0.03125);
        assert_eq!(probabilities.win_b, 0.09375);
        assert_eq!(probabilities.lose_n, 0.125);
        assert_eq!(probabilities.lose_g, 0.25);
        assert_eq!(probabilities.lose_b, 0.5);
    }

    #[test]
    fn equity_win_n() {
        let probabilities = Probabilities {
            win_n: 1.0,
            win_g: 0.0,
            win_b: 0.0,
            lose_n: 0.0,
            lose_g: 0.0,
            lose_b: 0.0,
        };
        assert_eq!(probabilities.equity(), 1.0);
    }

    #[test]
    fn equity_win_g() {
        let probabilities = Probabilities {
            win_n: 0.0,
            win_g: 1.0,
            win_b: 0.0,
            lose_n: 0.0,
            lose_g: 0.0,
            lose_b: 0.0,
        };
        assert_eq!(probabilities.equity(), 2.0);
    }

    #[test]
    fn equity_win_b() {
        let probabilities = Probabilities {
            win_n: 0.0,
            win_g: 0.0,
            win_b: 1.0,
            lose_n: 0.0,
            lose_g: 0.0,
            lose_b: 0.0,
        };
        assert_eq!(probabilities.equity(), 3.0);
    }

    #[test]
    fn equity_lose_n() {
        let probabilities = Probabilities {
            win_n: 0.0,
            win_g: 0.0,
            win_b: 0.0,
            lose_n: 1.0,
            lose_g: 0.0,
            lose_b: 0.0,
        };
        assert_eq!(probabilities.equity(), -1.0);
    }

    #[test]
    fn equity_lose_g() {
        let probabilities = Probabilities {
            win_n: 0.0,
            win_g: 0.0,
            win_b: 0.0,
            lose_n: 0.0,
            lose_g: 1.0,
            lose_b: 0.0,
        };
        assert_eq!(probabilities.equity(), -2.0);
    }

    #[test]
    fn equity_lose_b() {
        let probabilities = Probabilities {
            win_n: 0.0,
            win_g: 0.0,
            win_b: 0.0,
            lose_n: 0.0,
            lose_g: 0.0,
            lose_b: 1.0,
        };
        assert_eq!(probabilities.equity(), -3.0);
    }

    #[test]
    fn equity_balanced() {
        let probabilities = Probabilities {
            win_n: 0.3,
            win_g: 0.1,
            win_b: 0.1,
            lose_n: 0.3,
            lose_g: 0.1,
            lose_b: 0.1,
        };
        assert_eq!(probabilities.equity(), 0.0);
    }

    #[test]
    fn to_gnu() {
        let probabilities = Probabilities {
            win_n: 0.3,
            win_g: 0.1,
            win_b: 0.1,
            lose_n: 0.3,
            lose_g: 0.1,
            lose_b: 0.1,
        };
        let gv = probabilities.to_gnu();
        assert_eq!(probabilities, Probabilities::from(&gv));
    }
}
