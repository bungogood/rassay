use std::path::PathBuf;

use crate::{data::PositionBatch, inputs::Inputs};
use burn::{
    module::Module,
    nn::{
        self,
        loss::{MSELoss, Reduction::Mean},
    },
    record::{NoStdTrainingRecorder, Recorder},
    tensor::{
        self,
        backend::{AutodiffBackend, Backend},
        Data, Tensor,
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use super::EquityModel;

#[derive(Module, Debug)]
pub struct RassayModel<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
    output: nn::Linear<B>,
    activation: nn::ReLU,
}

impl<B: Backend> Default for RassayModel<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}

impl<B: Backend> EquityModel<B> for RassayModel<B> {
    const INPUT_SIZE: usize = 202;

    fn init_with(device: B::Device, model_path: &PathBuf) -> Self {
        let record = NoStdTrainingRecorder::new()
            .load(model_path.into(), &device)
            .expect("Failed to load model");
        Self::new_from(record)
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);
        let x = self.fc3.forward(x);
        let x = self.activation.forward(x);
        let x = self.output.forward(x);
        tensor::activation::softmax(x, 1)
    }

    fn inputs(&self, position: &bkgm::Position) -> Data<f32, 1> {
        Data::<f32, 1>::from(Inputs::from_position(position).to_vec().as_slice())
    }

    fn forward_step(&self, item: PositionBatch<B>) -> RegressionOutput<B> {
        let targets = item.probs;
        let output = self.forward(item.positions);
        let loss = MSELoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: Backend> RassayModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(202, 300).init(device),
            fc2: nn::LinearConfig::new(300, 250).init(device),
            fc3: nn::LinearConfig::new(250, 200).init(device),
            output: nn::LinearConfig::new(200, 5).init(device),
            activation: nn::ReLU::new(),
        }
    }

    fn new_from(record: RassayModelRecord<B>) -> Self {
        Self {
            fc1: nn::LinearConfig::new(202, 300).init_with(record.fc1),
            fc2: nn::LinearConfig::new(300, 250).init_with(record.fc2),
            fc3: nn::LinearConfig::new(250, 200).init_with(record.fc3),
            output: nn::LinearConfig::new(200, 5).init_with(record.output),
            activation: nn::ReLU::new(),
        }
    }
}

impl<B: AutodiffBackend> TrainStep<PositionBatch<B>, RegressionOutput<B>> for RassayModel<B> {
    fn step(&self, item: PositionBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<PositionBatch<B>, RegressionOutput<B>> for RassayModel<B> {
    fn step(&self, item: PositionBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
