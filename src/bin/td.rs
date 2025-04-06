use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Variant {
    BACKGAMMON,
    HYPERGAMMON,
    HYPERGAMMON2,
    HYPERGAMMON4,
    HYPERGAMMON5,
    LONGGAMMON,
    NACKGAMMON,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose the backgammon variant to play
    #[arg(short = 'v', long = "variant")]
    variant: Variant,
}

use bkgm::{
    Position, BACKGAMMON, HYPERGAMMON, HYPERGAMMON2, HYPERGAMMON4, HYPERGAMMON5, LONGGAMMON,
    NACKGAMMON,
};
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

pub fn run<const N: u8>(position: Position<N>, path: String) {
    let device = get_device(true);

    // let model = match &args.model_path {
    //     Some(path) => TDModel::<Autodiff<LibTorch>>::init_with(device, path, 160),
    //     None => TDModel::<Autodiff<LibTorch>>::new(&device, 160),
    // };

    let model = TDModel::<Autodiff<LibTorch>>::new(&device, 160);

    let td_config = TDConfig::new(0.1, 0.7, 0.05);

    let mut td: TDTrainer<Autodiff<LibTorch>> = TDTrainer::new(device.clone(), td_config);

    let start = std::time::Instant::now();

    let model2 = td.train(&position, model, 1_000_000, path);

    let elapsed = start.elapsed();
    println!("Elapsed time: {:?}", elapsed);

    // model2
    //     .save_file(format!("model/td"), &NoStdTrainingRecorder::new())
    //     .expect("Failed to save trained model");
}

fn main() {
    let args = Args::parse();

    let base = "model/exp005";

    let position = match args.variant {
        Variant::BACKGAMMON => run(BACKGAMMON, format!("{}/backgammon", base)),
        Variant::HYPERGAMMON => run(HYPERGAMMON, format!("{}/hypergammon2", base)),
        Variant::HYPERGAMMON2 => run(HYPERGAMMON2, format!("{}/hypergammon3", base)),
        Variant::HYPERGAMMON4 => run(HYPERGAMMON4, format!("{}/hypergammon4", base)),
        Variant::HYPERGAMMON5 => run(HYPERGAMMON5, format!("{}/hypergammon5", base)),
        Variant::LONGGAMMON => run(LONGGAMMON, format!("{}/longgammon", base)),
        Variant::NACKGAMMON => run(NACKGAMMON, format!("{}/nackgammon", base)),
    };
}
