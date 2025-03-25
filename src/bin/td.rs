use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Model path
    #[arg(short = 'm', long = "model")]
    model_path: Option<PathBuf>,

    /// Verbose
    #[arg(short = 'v', long = "verbose", default_value = "false")]
    verbose: bool,

    /// Use CPU only
    #[arg(short = 'c', long = "cpu", default_value = "false")]
    cpu_only: bool,
}

use bkgm::{BACKGAMMON, HYPERGAMMON};
use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::backend::Autodiff;
use burn::module::Module;
use burn::record::NoStdTrainingRecorder;
use rassay::model::{EquityModel, TDModel};
use rassay::training::td_learning::{TDConfig, TDTrainer};

fn get_device(cup_only: bool) -> LibTorchDevice {
    if cup_only {
        LibTorchDevice::Cpu
    } else {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        // MacOs Mps too slow
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Cpu;
        // let device = LibTorchDevice::Mps;
        device
    }
}

pub fn run(args: &Args) {
    let device = get_device(args.cpu_only);

    let model = match &args.model_path {
        Some(path) => TDModel::<Autodiff<LibTorch>>::init_with(device, path, 160),
        None => TDModel::<Autodiff<LibTorch>>::new(&device, 160),
    };

    let td_config = TDConfig::new(0.1, 0.7);

    let mut td: TDTrainer<Autodiff<LibTorch>> = TDTrainer::new(device.clone(), td_config);

    let model2 = td.train(&BACKGAMMON, model, 1_000_000);

    model2
        .save_file(format!("model/td"), &NoStdTrainingRecorder::new())
        .expect("Failed to save trained model");
}

fn main() {
    let args = Args::parse();
    run(&args);
}
