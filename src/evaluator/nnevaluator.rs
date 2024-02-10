use std::marker::PhantomData;

use bkgm::State;
use burn::tensor::{backend::Backend, Data, Tensor};

use super::{Evaluator, PartialEvaluator};
use crate::{inputs::Inputs, model::EquityModel, probabilities::Probabilities};

pub struct NNEvaluator<B: Backend, G: State, E: EquityModel<B>> {
    model: E,
    phantom: PhantomData<G>,
    device: B::Device,
}

impl<B: Backend, G: State, E: EquityModel<B>> NNEvaluator<B, G, E> {
    pub fn new(device: B::Device, model: E) -> Self {
        Self {
            model,
            phantom: PhantomData,
            device,
        }
    }
}

impl<B: Backend, G: State, E: EquityModel<B>> PartialEvaluator<G> for NNEvaluator<B, G, E> {
    fn try_eval(&self, position: &G) -> f32 {
        let probs = self.eval(position);
        probs.equity()
    }

    fn best_position(&self, position: &G, dice: &bkgm::Dice) -> G {
        let positions = position.possible_positions(dice);

        let inputs = self.model.input_tensor(
            &self.device,
            positions.iter().map(|pos| pos.position()).collect(),
        );

        let output = self.model.forward(inputs);

        let data: Data<f32, 2> = output.into_data().convert();

        *positions
            .iter()
            .enumerate()
            .map(|(i, pos)| {
                let row = i * 5;
                let win = data.value[row];
                let win_g = data.value[row + 1];
                let win_b = data.value[row + 2];
                let lose_g = data.value[row + 3];
                let lose_b = data.value[row + 4];
                let equity = 2.0 * win - 1.0 + win_g - lose_g + win_b - lose_b;
                (pos, equity)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0
    }
}

impl<B: Backend, G: State, E: EquityModel<B>> Evaluator<G> for NNEvaluator<B, G, E> {
    #[allow(dead_code)]
    fn eval(&self, position: &G) -> Probabilities {
        let data = Data::<f32, 1>::from(
            Inputs::from_position(&position.position())
                .to_vec()
                .as_slice(),
        );
        let input = Tensor::<B, 1>::from_data(data.convert(), &self.device);
        let input = input.reshape([1, 202]);

        let output = self.model.forward(input);
        let output = output.reshape([5]);
        let data: Data<f32, 1> = output.into_data().convert();

        Probabilities {
            win_n: data.value[0] - data.value[1],
            win_g: data.value[1] - data.value[2],
            win_b: data.value[2],
            lose_n: 1.0 - data.value[0] - data.value[3],
            lose_g: data.value[3] - data.value[4],
            lose_b: data.value[4],
        }
    }
}
