use bkgm::Position;
use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::inputs::Inputs;

fn serialize_position_id<S>(position: &Position, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&position.position_id())
}

fn deserialize_position_id<'de, D>(deserializer: D) -> Result<Position, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(Position::from_id(&s, 15).unwrap())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionItem {
    #[serde(
        serialize_with = "serialize_position_id",
        deserialize_with = "deserialize_position_id"
    )]
    pub position: Position,
    #[serde(flatten)]
    pub probs: OutcomeProb,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutcomeProb {
    pub win: f32,
    pub win_g: f32,
    pub win_b: f32,
    pub lose_g: f32,
    pub lose_b: f32,
}

impl OutcomeProb {
    pub fn to_slice(&self) -> [f32; 5] {
        [self.win, self.win_g, self.win_b, self.lose_g, self.lose_b]
    }
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
            .map(|tensor| tensor.reshape([1, 5]))
            .collect();

        let positions = Tensor::cat(positions, 0);
        let probs = Tensor::cat(targets, 0);

        PositionBatch { positions, probs }
    }
}
