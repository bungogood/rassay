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

#[cfg(feature = "tch")]
mod tch {
    use crate::Args;
    use bkgm::{Backgammon, Hypergammon};
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use burn::backend::Autodiff;
    use rassay::model::{self, EquityModel, RassayModel};
    use rassay::training::self_play::SelfPlay;

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
            Some(path) => RassayModel::<Autodiff<LibTorch>>::init_with(device, path),
            None => RassayModel::<Autodiff<LibTorch>>::new(&device),
        };

        let sp: SelfPlay<Autodiff<LibTorch>, Hypergammon, RassayModel<Autodiff<LibTorch>>> =
            SelfPlay::new(device.clone(), model.clone());

        sp.learn(model)
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
