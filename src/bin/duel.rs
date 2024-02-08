#![allow(unused_imports)]
use bkgm::{Backgammon, Hypergammon, State};
use burn::backend::LibTorch;
use burn::record::{NoStdTrainingRecorder, Recorder};
use clap::Parser;
use rassay::dicegen::FastrandDice;
use rassay::duel::Duel;
use rassay::evaluator::{self, NNEvaluator, PartialEvaluator, PubEval, RandomEvaluator};
use rassay::model::{EquityModel, LargeModel, RassayModel};
use rassay::probabilities::ResultCounter;
use serde::de;
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
}

fn run(args: &Args) {
    let device = burn::backend::libtorch::LibTorchDevice::Cpu;
    let model1 = RassayModel::<LibTorch>::init_with(device, &args.model1);
    let model2 = LargeModel::<LibTorch>::init_with(device, &args.model2);

    // let evaluator1 = PubEval::<Backgammon>::new();
    let evaluator1 = NNEvaluator::new(device, model1);
    let evaluator2 = NNEvaluator::new(device, model2);
    // let evaluator2 = RandomEvaluator::new();
    duel::<Backgammon>(evaluator1, evaluator2, args.matches);
}

fn duel<G: State>(
    evaluator1: impl PartialEvaluator<G>,
    evaluator2: impl PartialEvaluator<G>,
    rounds: usize,
) {
    let duel = Duel::new(evaluator1, evaluator2);
    let mut results = ResultCounter::default();
    for _ in 0..rounds {
        let outcome = duel.duel(&mut FastrandDice::new());
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
    println!("\nDone");
}

fn main() {
    let args = Args::parse();
    run(&args);
}
