use bkgm::dice_gen::FastrandDice;
use bkgm::{
    Position, State, BACKGAMMON, HYPERGAMMON, HYPERGAMMON2, HYPERGAMMON4, HYPERGAMMON5, LONGGAMMON,
    NACKGAMMON,
};
use burn::backend::libtorch::LibTorchDevice;
use burn::backend::LibTorch;
use burn::record::{NoStdTrainingRecorder, Recorder};
use clap::Parser;
use rassay::duel::Duel;
use rassay::evaluator::{
    self, Evaluator, GreedyEvaluator, HyperEvaluator, PartialEvaluator, PlyEvaluator, PubEval,
    RandomEvaluator, RolloutEvaluator, SubHyperEvaluator,
};
use rassay::model::{EquityModel, TDModel};
use rassay::probabilities::ResultCounter;
use std::{
    io::{stdout, Write},
    path::PathBuf,
};

// TODO: improve argument names & allow for rollouts, random, etc.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model 1
    model1: PathBuf,

    /// Model 2
    model2: PathBuf,

    /// Matches
    #[arg(short = 'm', long = "matches", default_value = "10000")]
    matches: usize,

    /// Use CPU only
    #[arg(short = 'c', long = "cpu", default_value = "false")]
    cpu_only: bool,
}

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

fn run(args: &Args) {
    let device = get_device(args.cpu_only);
    // let model1 = RassayModel::<LibTorch>::init_with(device, &args.model1);
    // let model2 = RassayModel::<LibTorch>::init_with(device, &args.model2);

    let evaluator1 = TDModel::<LibTorch>::init_with(device, &args.model1, 160);
    let evaluator2 = TDModel::<LibTorch>::init_with(device, &args.model2, 160);

    // let evaluator1 = PubEval::new();
    // let evaluator2 = PubEval::new();
    // let evaluator1 = HyperEvaluator::new().unwrap();
    // let evaluator2 = HyperEvaluator::new().unwrap();
    // let evaluator2 = SubHyperEvaluator::from_file("../diss/data/hyper/data/hyper-win.db").unwrap();
    // let evaluator1 = RandomEvaluator::new();
    // let evaluator2 = RandomEvaluator::new();
    // let evaluator1 = GreedyEvaluator::new(evaluator1, 0.4);
    // let evaluator1 = PlyEvaluator::new(evaluator1, 2);
    // let evaluator2 = PlyEvaluator::new(evaluator2, 2);
    // let evaluator1 = RolloutEvaluator::new(evaluator1, 100);
    // duel(&HYPERGAMMON2, evaluator1, evaluator2, args.matches / 2);
    // duel(&HYPERGAMMON, evaluator1, evaluator2, args.matches / 2);
    duel(&HYPERGAMMON4, evaluator1, evaluator2, args.matches / 2);
    // duel(&HYPERGAMMON5, evaluator1, evaluator2, args.matches / 2);
    // duel(&BACKGAMMON, evaluator1, evaluator2, args.matches / 2);
    // duel(&NACKGAMMON, evaluator1, evaluator2, args.matches / 2);
    // duel(&LONGGAMMON, evaluator1, evaluator2, args.matches / 2);
}

fn duel<G: State>(
    state: &G,
    evaluator1: impl PartialEvaluator<G>,
    evaluator2: impl PartialEvaluator<G>,
    rounds: usize,
) {
    let duel = Duel::new(evaluator1, evaluator2);
    let mut results = ResultCounter::default();
    let mut unique = std::collections::HashSet::new();
    let mut game_length = std::collections::HashMap::new();
    let mut phases = std::collections::HashMap::new();
    let mut captures = 0;
    let mut possible = 0;
    let start = std::time::Instant::now();
    for round in 0..rounds {
        // if round % 1000 == 0 {
        //     println!("{},{}", round, unique.len());
        // }
        let outcome = duel.single_duel(
            state,
            &mut FastrandDice::new(),
            &mut unique,
            &mut game_length,
            &mut phases,
            &mut captures,
            &mut possible,
        );
        results = results.combine(&outcome);
        let probs = results.probabilities();
        print!(
            "\rAfter {} games is the equity {:.3} ({:.1}%). {:?}",
            results.sum(),
            probs.equity(),
            probs.win_prob() * 100.0,
            probs,
        );
        stdout().flush().unwrap()
    }

    let elapsed = start.elapsed();
    let probs = results.probabilities();
    print!(
        "\rAfter {} games is the equity {:.3} ({:.1}%). {:?}",
        results.sum(),
        probs.equity(),
        probs.win_prob() * 100.0,
        probs,
    );

    // print all game length sorted by depth
    // let mut kv = game_length.iter().collect::<Vec<_>>();
    // kv.sort_by_key(|p| *p.0);
    // for (depth, count) in kv.iter() {
    //     println!("{},{}", depth, count);
    // }

    println!("\nDone");

    println!("Captures: {}", captures);
    println!("Possible: {}", possible);
    println!("Elapsed: {:?}", elapsed);
    println!("Unique: {}", unique.len());
    println!("Phases: {:?}", phases);
}

fn main() {
    let args = Args::parse();
    run(&args);
}
