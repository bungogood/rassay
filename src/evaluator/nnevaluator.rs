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
