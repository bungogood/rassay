mod large;
mod rassay;

use std::path::PathBuf;

use burn::tensor::{backend::Backend, Tensor};
pub use large::LargeModel;
pub use rassay::RassayModel;

pub trait EquityModel<B: Backend>: Sized {
    fn init_with(device: B::Device, model_path: &PathBuf) -> Self;
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}
