mod large;
mod rassay;

use std::path::PathBuf;

use bkgm::Position;
use burn::tensor::{backend::Backend, Data, Tensor};
pub use large::LargeModel;
pub use rassay::RassayModel;

pub trait EquityModel<B: Backend>: Sized {
    const INPUT_SIZE: usize;
    fn init_with(device: B::Device, model_path: &PathBuf) -> Self;
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
    fn inputs(&self, position: &Position) -> Data<f32, 1>;

    fn input_tensor(&self, device: &B::Device, positions: Vec<Position>) -> Tensor<B, 2> {
        let tensor_pos = positions
            .iter()
            .map(|item| self.inputs(&item))
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), device))
            .map(|tensor| tensor.reshape([1, Self::INPUT_SIZE]))
            .collect();

        Tensor::cat(tensor_pos, 0)
    }
}
