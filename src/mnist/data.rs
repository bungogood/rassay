use bkgm::Position;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

use crate::{inputs::Inputs, probabilities::Probabilities};

pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // normalize: make between [0,1] and make the mean =  0 and std = 1
            // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    Data::from([(item.label as i64).elem()]),
                    &self.device,
                )
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MNISTBatch { images, targets }
    }
}

pub struct PositionItem {
    pub position: Position,
    pub probs: Probabilities,
}

pub struct PositionBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct PositionBatch<B: Backend> {
    pub positions: Tensor<B, 2>,
    pub probs: Tensor<B, 2>,
}

impl<B: Backend> PositionBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<PositionItem, PositionBatch<B>> for PositionBatcher<B> {
    fn batch(&self, items: Vec<PositionItem>) -> PositionBatch<B> {
        let positions = items
            .iter()
            .map(|item| {
                Data::<f32, 1>::from(Inputs::from_position(&item.position).to_vec().as_slice())
            })
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 202]))
            .collect();

        let targets = items
            .iter()
            .map(|item| Data::<f32, 1>::from(item.probs.to_slice()))
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([1, 6]))
            .collect();

        let positions = Tensor::cat(positions, 0);
        let probs = Tensor::cat(targets, 0);

        PositionBatch { positions, probs }
    }
}
