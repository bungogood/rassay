use std::path::PathBuf;

use crate::{
    evaluator::{Evaluator, PartialEvaluator},
    inputs::{self, Inputs},
};
use bincode::de;
use bkgm::{dice::ALL_21, position, GameResult, State};
use burn::{
    // module::Module,
    module::Module,
    nn::{self, LinearConfig},
    optim::GradientsParams,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::{
        self,
        activation::sigmoid,
        backend::{AutodiffBackend, Backend},
        Tensor, TensorData,
    },
};

use super::EquityModel;

#[derive(Module, Debug)]
pub struct TDModel<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
}

impl<B: Backend> Default for TDModel<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device, 40)
    }
}

impl<B: Backend> EquityModel<B> for TDModel<B> {
    const INPUT_SIZE: usize = 202;

    fn init_with(device: B::Device, model_path: &PathBuf, size: usize) -> Self {
        let record = NoStdTrainingRecorder::new()
            .load(model_path.into(), &device)
            .expect("Failed to load model");
        Self::new_from(&device, record, size)
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        // let x = sigmoid(x);
        // let x = self.fc2.forward(x);
        let x = sigmoid(x);
        let x = self.fc3.forward(x);
        sigmoid(x)
    }

    fn inputs<G: State>(&self, position: &G) -> Vec<f32> {
        // Data::<f32, 1>::from(Inputs::from_position(position).to_vec().as_slice())
        Inputs::from_position(position).to_vec()
        // position_to_vec(position)
    }
}

impl<B: Backend> TDModel<B> {
    pub fn new(device: &B::Device, size: usize) -> Self {
        Self {
            fc1: LinearConfig::new(202, size).init(device),
            fc2: LinearConfig::new(size, size).init(device),
            fc3: LinearConfig::new(size, 1).init(device),
        }
    }

    fn new_from(device: &B::Device, record: TDModelRecord<B>, size: usize) -> Self {
        let model = Self::new(device, size);
        model.load_record(record)
    }

    pub fn from_result(&self, result: GameResult) -> f32 {
        match result {
            GameResult::WinNormal => 1.0,
            GameResult::WinGammon => 1.0,
            GameResult::WinBackgammon => 1.0,
            GameResult::LoseNormal => 0.0,
            GameResult::LoseGammon => 0.0,
            GameResult::LoseBackgammon => 0.0,
        }
    }

    pub fn forward_pos<G: bkgm::State>(&self, position: G, device: &B::Device) -> f32 {
        let inputs = self.input_tensor(device, &vec![position]);
        let output = self.forward(inputs);

        let value: TensorData = output.into_data();
        let value: &[f32] = value.as_slice().unwrap();
        value[0]
    }
}

impl<B: AutodiffBackend> TDModel<B> {
    pub fn forward_grads_pos<G: bkgm::State>(
        &self,
        position: &G,
        device: &B::Device,
    ) -> (f32, GradientsParams) {
        let inputs = self.input_tensor(device, &vec![*position]);
        let output = self.forward(inputs);

        let grads = GradientsParams::from_grads(output.backward(), self);

        let value: TensorData = output.into_data();
        let value: &[f32] = value.as_slice().unwrap();
        (value[0], grads)
    }
}

// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub struct FState<G: State> {
//     pub state: G,
//     pub turn: bool,
// }

// impl<G: State> State for FState<G> {
//     const NUM_CHECKERS: u8 = G::NUM_CHECKERS;

//     fn new() -> Self {
//         Self {
//             state: G::new(),
//             turn: true,
//         }
//     }

//     fn position(&self) -> position::Position {
//         self.state.position()
//     }

//     fn flip(&self) -> Self {
//         Self {
//             state: self.state.flip(),
//             turn: !self.turn,
//         }
//     }

//     fn game_state(&self) -> bkgm::GameState {
//         self.state.game_state()
//     }

//     fn possible_positions(&self, dice: &bkgm::Dice) -> Vec<Self> {
//         self.state
//             .possible_positions(dice)
//             .iter()
//             .map(|pos| FState {
//                 state: *pos,
//                 turn: !self.turn,
//             })
//             .collect()
//     }

//     fn from_position(position: bkgm::Position) -> Self {
//         Self {
//             state: G::from_position(position),
//             turn: true,
//         }
//     }

//     fn x_bar(&self) -> u8 {
//         self.state.x_bar()
//     }

//     fn o_bar(&self) -> u8 {
//         self.state.o_bar()
//     }

//     fn x_off(&self) -> u8 {
//         self.state.x_off()
//     }

//     fn o_off(&self) -> u8 {
//         self.state.o_off()
//     }

//     fn pip(&self, pip: usize) -> i8 {
//         self.state.pip(pip)
//     }

//     fn board(&self) -> [i8; 24] {
//         self.state.board()
//     }

//     fn dbhash(&self) -> usize {
//         self.state.dbhash()
//     }
// }

// impl<G: State> FState<G> {
//     pub fn f_game_state(&self) -> bkgm::GameState {
//         if self.turn {
//             self.state.game_state()
//         } else {
//             self.state.flip().game_state()
//         }
//     }

//     pub fn f_state(&self) -> G {
//         if self.turn {
//             self.state
//         } else {
//             self.state.flip()
//         }
//     }
// }

impl<G: State, B: Backend> PartialEvaluator<G> for TDModel<B> {
    fn try_eval(&self, pos: &G) -> f32 {
        let device = B::Device::default();

        if pos.turn() {
            let inputs = self.input_tensor(&device, &vec![*pos]);
            let output = self.forward(inputs);
            let output = output.reshape([1]);

            let value: TensorData = output.into_data();
            let value: &[f32] = value.as_slice().unwrap();
            value[0]
        } else {
            let inputs = self.input_tensor(&device, &vec![pos.flip()]);
            let output = self.forward(inputs);
            let output = output.reshape([1]);

            let value: TensorData = output.into_data();
            let value: &[f32] = value.as_slice().unwrap();
            1.0 - value[0]
        }
    }

    fn best_position(&self, position: &G, dice: &bkgm::Dice) -> G {
        let positions = position.possible_positions(dice);

        if position.turn() {
            let inputs = self.input_tensor(&B::Device::default(), &positions);

            let output = self.forward(inputs);

            let value: TensorData = output.into_data();
            let value: &[f32] = value.as_slice().unwrap();

            *positions
                .iter()
                .zip(value)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .to_owned()
                .0
        } else {
            let inputs = self.input_tensor(
                &B::Device::default(),
                &positions.iter().map(|pos| pos.flip()).collect(),
            );

            let output = self.forward(inputs);

            let value: TensorData = output.into_data();
            let value: &[f32] = value.as_slice().unwrap();

            *positions
                .iter()
                .zip(value)
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .to_owned()
                .0
        }
    }
}

pub(crate) const NUM_INPUTS: usize = 202;

pub fn position_to_vec<G: State>(pos: &G) -> Vec<f32> {
    let mut vec = vec![0.0; NUM_INPUTS];

    // Off-board checkers
    vec[0] = pos.x_off() as f32;
    vec[1] = pos.o_off() as f32;

    // Bar checkers (index 2-5 for X, 6-9 for O)
    encode_pip(&mut vec[2..6], pos.x_bar());
    encode_pip(&mut vec[6..10], pos.o_bar());

    // Board checkers
    for i in 1..=24 {
        let pip = pos.pip(i);
        if pip > 0 {
            encode_pip(&mut vec[(i * 4 + 6)..(i * 4 + 10)], pip as u8);
        } else if pip < 0 {
            encode_pip(&mut vec[(i * 4 + 6 + 96)..(i * 4 + 10 + 96)], -pip as u8);
        }
    }

    vec
}

/// Encode a pip count into four values as per the TD-Gammon style representation.
fn encode_pip(target: &mut [f32], pip: u8) {
    match pip {
        0 => target.copy_from_slice(&[0.0, 0.0, 0.0, 0.0]),
        1 => target.copy_from_slice(&[1.0, 0.0, 0.0, 0.0]),
        2 => target.copy_from_slice(&[0.0, 1.0, 0.0, 0.0]),
        p => target.copy_from_slice(&[0.0, 0.0, 1.0, (p - 3) as f32]),
    }
}
