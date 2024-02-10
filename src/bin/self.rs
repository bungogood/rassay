use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Test Csv file
    #[arg(short = 't', long = "test-csv", default_value = "data/test.csv")]
    test_csv: PathBuf,

    /// Train Csv file
    #[arg(short = 'r', long = "train-csv", default_value = "data/train.csv")]
    train_csv: PathBuf,

    /// Model Path
    #[arg(short = 'm', long = "model-path", default_value = "model/model")]
    model_path: PathBuf,

    /// Load model
    #[arg(short = 'l', long = "load", default_value = "false")]
    load_model: bool,

    /// Verbose
    #[arg(short = 'v', long = "verbose", default_value = "false")]
    verbose: bool,

    /// Use CPU only
    #[arg(short = 'c', long = "cpu", default_value = "false")]
    cpu_only: bool,
}

#[cfg(feature = "tch")]
mod tch {
    use crate::Args;
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use burn::backend::Autodiff;
    use rassay::self_play;

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

        self_play::run::<Autodiff<LibTorch>>(
            device,
            args.load_model,
            &args.model_path,
            &args.test_csv,
            &args.train_csv,
        );
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::Args;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use burn::backend::Autodiff;
    use rassay::self_play;

    pub fn run(args: &Args) {
        let device = WgpuDevice::default();
        self_play::run::<Autodiff<Wgpu>>(device, &args.model_path, &args.test_csv, &args.train_csv);
    }
}

fn main() {
    let args = Args::parse();
    #[cfg(feature = "tch")]
    tch::run(&args);
    #[cfg(feature = "wgpu")]
    wgpu::run(&args);
}
