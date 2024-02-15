use bkgm::{
    dice::{ALL_21, ALL_SINGLES},
    utils::mcomb,
    GameState::{GameOver, Ongoing},
    Hypergammon, State,
};
use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rassay::probabilities::Probabilities;
use rayon::{
    iter::ParallelBridge,
    prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    iter::zip,
    path::PathBuf,
    sync::Arc,
};

/// Make Hypergammon database

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Output file
    #[arg(short = 'f', long = "file", default_value = "data/hyper.db")]
    file: PathBuf,

    /// Unique file
    #[arg(short = 'u', long = "unqiue", default_value = "data/unique.db")]
    uniquefile: PathBuf,

    /// Number of iterations
    #[arg(short = 'i', long = "iter", default_value = "100")]
    iterations: usize,

    /// Discount Factor
    #[arg(short = 'y', long = "discount", default_value = "1.0")]
    discount: f32,

    /// Delta
    #[arg(short = 'd', long = "delta", default_value = "0.001")]
    delta: f32,

    /// Equities
    #[arg(short = 'e', long = "equities")]
    equities: bool,

    /// Verbose
    #[arg(short = 'v', long = "verbose")]
    verbose: bool,
}

fn read_unique(args: &Args) -> io::Result<Vec<Hypergammon>> {
    let file = File::open(&args.uniquefile)?;
    let mut reader = BufReader::new(file);

    let mut buffer = [0u8; 10];
    let mut unique = Vec::new();

    while reader.read_exact(&mut buffer).is_ok() {
        let hypergammon = Hypergammon::decode(buffer);
        unique.push(hypergammon);
    }

    Ok(unique)
}

fn write_unique(args: &Args, unique: &Vec<Hypergammon>) -> io::Result<()> {
    let file = File::create(&args.uniquefile)?;
    let mut buf_writer = BufWriter::new(file);

    for position in unique.iter() {
        buf_writer.write_all(&position.encode())?;
    }

    buf_writer.flush()
}

fn write_file(args: &Args, probs: &[Probabilities]) -> io::Result<()> {
    let file = File::create(&args.file)?;
    let mut buf_writer = BufWriter::new(file);

    for prob in probs.iter() {
        let wgbgb = [
            prob.win_n + prob.win_g + prob.win_b,
            prob.win_g + prob.win_b,
            prob.win_b,
            prob.lose_g + prob.lose_b,
            prob.lose_b,
        ];
        for wgbgb in wgbgb.iter() {
            buf_writer.write_all(&wgbgb.to_le_bytes())?;
        }
    }

    buf_writer.flush()
}

const POSSIBLE: usize = mcomb(26, Hypergammon::NUM_CHECKERS as usize).pow(2);
const STYLE: &str =
    "{wide_bar} {pos}/{len} ({percent}%) Elapsed: {elapsed_precise} ETA: {eta_precise}";

fn equity_update(positions: &PosMap, probs: &Vec<Probabilities>) -> Vec<Probabilities> {
    let shared_probs = Arc::new(probs);

    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    probs
        .par_iter()
        .progress_with_style(style)
        .enumerate()
        .map(|(hash, equity)| match positions.get(&hash) {
            Some(rolls) => {
                let mut possiblilies = 0.0;
                let mut total = Probabilities::empty();
                for (n, children) in rolls {
                    let best = children
                        .iter()
                        .map(|child| (shared_probs[*child], shared_probs[*child].equity()))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap()
                        .0;
                    possiblilies += n;
                    total = Probabilities {
                        win_n: total.win_n + n * best.win_n,
                        win_g: total.win_g + n * best.win_g,
                        win_b: total.win_b + n * best.win_b,
                        lose_n: total.lose_n + n * best.lose_n,
                        lose_g: total.lose_g + n * best.lose_g,
                        lose_b: total.lose_b + n * best.lose_b,
                    }
                }
                Probabilities {
                    win_n: total.win_n / possiblilies,
                    win_g: total.win_g / possiblilies,
                    win_b: total.win_b / possiblilies,
                    lose_n: total.lose_n / possiblilies,
                    lose_g: total.lose_g / possiblilies,
                    lose_b: total.lose_b / possiblilies,
                }
            }
            None => *equity,
        })
        .collect()
}

type PosMap = HashMap<usize, Vec<(f32, Vec<usize>)>>;

fn unqiue(verbose: bool) -> Vec<Hypergammon> {
    let position = Hypergammon::new();
    let mut found = HashSet::new();
    let mut new_positons = vec![];
    let before = found.len();

    for die in ALL_SINGLES {
        let children = position.possible_positions(&die);
        for child in children {
            if !found.contains(&child) {
                found.insert(child);
                new_positons.push(child);
            }
        }
    }

    let mut depth = 1;
    let discovered = found.len() - before;
    if verbose {
        println!(
            "{}\t{}\tpositions reached after {} roll",
            discovered,
            found.len(),
            depth
        );
    }

    while !new_positons.is_empty() {
        let mut queue = new_positons;
        new_positons = vec![];
        let before = found.len();
        while let Some(position) = queue.pop() {
            match position.game_state() {
                Ongoing => {
                    for (die, _) in ALL_21 {
                        let children = position.possible_positions(&die);
                        for child in children {
                            if !found.contains(&child) {
                                found.insert(child);
                                new_positons.push(child);
                            }
                        }
                    }
                }
                GameOver(_) => {}
            }
        }
        let discovered = found.len() - before;
        depth += 1;
        if verbose {
            println!(
                "{}\t{}\tpositions reached after {} rolls",
                discovered,
                found.len(),
                depth
            );
        }
    }

    found.into_iter().collect()
}

fn split_positions(positions: Vec<Hypergammon>) -> (Vec<Hypergammon>, Vec<Hypergammon>) {
    let mut ongoing = vec![];
    let mut gameover = vec![];
    for position in positions {
        match position.game_state() {
            Ongoing => ongoing.push(position),
            GameOver(_) => gameover.push(position),
        }
    }
    (ongoing, gameover)
}

fn initial_equities(gameover: Vec<Hypergammon>) -> Vec<Probabilities> {
    let mut equities = vec![Probabilities::empty(); POSSIBLE];
    gameover.iter().for_each(|p| {
        equities[p.dbhash()] = Probabilities::from_result(match &p.game_state() {
            Ongoing => panic!("Should not be ongoing"),
            GameOver(result) => result,
        })
    });
    equities
}

fn create_posmap(ongoing: Vec<Hypergammon>) -> PosMap {
    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let posmap = ongoing
        .par_iter()
        .progress_with_style(style)
        .map(|position| {
            let mut c = vec![];
            for (die, n) in ALL_21 {
                let children = position.possible_positions(&die);
                c.push((n, children.iter().map(|pos| pos.dbhash()).collect()));
            }
            (position.dbhash(), c)
        })
        .collect();

    posmap
}

fn check_open(equities: &Vec<Probabilities>) -> Probabilities {
    let starting = Hypergammon::new();
    let mut possibilies = 0.0;
    let mut open = Probabilities::empty();
    for die in ALL_SINGLES {
        let children = starting.possible_positions(&die);
        let best = children
            .iter()
            .map(|child| (equities[child.dbhash()], equities[child.dbhash()].equity()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        possibilies += 1.0;
        open = Probabilities {
            win_n: open.win_n + best.win_n,
            win_g: open.win_g + best.win_g,
            win_b: open.win_b + best.win_b,
            lose_n: open.lose_n + best.lose_n,
            lose_g: open.lose_g + best.lose_g,
            lose_b: open.lose_b + best.lose_b,
        }
    }
    Probabilities {
        win_n: open.win_n / possibilies,
        win_g: open.win_g / possibilies,
        win_b: open.win_b / possibilies,
        lose_n: open.lose_n / possibilies,
        lose_g: open.lose_g / possibilies,
        lose_b: open.lose_b / possibilies,
    }
}

// fn initialise_equities(gameover: Vec<Hypergammon>) -> Vec<Probabilities> {
//     let mut equities = vec![Probabilities::empty(); POSSIBLE];
//     gameover.iter().for_each(|p| {
//         equities[p.dbhash()] = Probabilities::from_result(match &p.game_state() {
//             Ongoing => panic!("Should not be ongoing"),
//             GameOver(result) => result,
//         })
//     });
//     equities
// }

// fn initialise_probabilities(gameover: Vec<Hypergammon>) -> Vec<f32> {
//     let mut probs = vec![0.0; POSSIBLE];
//     gameover.iter().for_each(|p| {
//         probs[p.dbhash()] = match &p.game_state() {
//             Ongoing => panic!("Should not be ongoing"),
//             GameOver(result) => result.value(),
//         }
//     });
//     probs
// }

// fn iterate_probabilities(positions: &PosMap, probs: &Vec<Probabilities>) -> (Vec<Probabilities>, f32, f32) {
//     let mut probs = vec![];
//     for _ in 0..POSSIBLE {
//         probs.push(Probabilities::empty());
//     }
//     probs
// }

fn iterate_probabilities(
    positions: &PosMap,
    probs: &Vec<Probabilities>,
) -> (Vec<Probabilities>, f32, f32) {
    let prev_probs = Arc::new(probs);

    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let new_probs: Vec<Probabilities> = probs
        .par_iter()
        .progress_with_style(style)
        .enumerate()
        .map(|(hash, prob)| match positions.get(&hash) {
            Some(rolls) => {
                let mut possiblilies = 0.0;
                let mut total = Probabilities::empty();
                for (n, children) in rolls {
                    let best = children
                        .iter()
                        .map(|child| (prev_probs[*child], prev_probs[*child].equity()))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .unwrap()
                        .0;
                    possiblilies += n;
                    total = Probabilities {
                        win_n: total.win_n + n * best.win_n,
                        win_g: total.win_g + n * best.win_g,
                        win_b: total.win_b + n * best.win_b,
                        lose_n: total.lose_n + n * best.lose_n,
                        lose_g: total.lose_g + n * best.lose_g,
                        lose_b: total.lose_b + n * best.lose_b,
                    }
                }
                Probabilities {
                    win_n: total.win_n / possiblilies,
                    win_g: total.win_g / possiblilies,
                    win_b: total.win_b / possiblilies,
                    lose_n: total.lose_n / possiblilies,
                    lose_g: total.lose_g / possiblilies,
                    lose_b: total.lose_b / possiblilies,
                }
            }
            None => *prob,
        })
        .collect();

    let (sum_err, min_err) = zip(prev_probs.iter(), new_probs.iter())
        .par_bridge()
        .map(|(a, b)| {
            let delta = (a.equity() - b.equity()).abs();
            (delta, delta)
        })
        .reduce(
            || (0.0, f32::INFINITY),
            |a, b| (a.0 + b.0, f32::min(a.1, b.1)),
        );
    let avg_err = sum_err / new_probs.len() as f32;

    (new_probs, avg_err, min_err)
}

fn iterate_equities(
    positions: &PosMap,
    equities: &Vec<f32>,
    discount: f32,
) -> (Vec<f32>, f32, f32) {
    let prev_equities = Arc::new(equities);

    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let new_equities: Vec<f32> = equities
        .par_iter()
        .progress_with_style(style)
        .enumerate()
        .map(|(hash, equity)| match positions.get(&hash) {
            Some(rolls) => {
                let mut possiblilies = 0.0;
                let mut total = 0.0;
                for (n, children) in rolls {
                    let best = children
                        .iter()
                        .map(|child| prev_equities[*child])
                        .min_by(|a, b| a.partial_cmp(&b).unwrap())
                        .unwrap();
                    possiblilies += n;
                    total += n * best;
                }
                discount * (total / possiblilies)
            }
            None => *equity,
        })
        .collect();

    let (sum_err, min_err) = zip(new_equities.iter(), equities.iter())
        .par_bridge()
        .map(|(a, b)| {
            let delta = (a - b).abs();
            (delta, delta)
        })
        .reduce(
            || (0.0, f32::INFINITY),
            |a, b| (a.0 + b.0, f32::min(a.1, b.1)),
        );
    let avg_err = sum_err / new_equities.len() as f32;

    (new_equities, avg_err, min_err)
}

// fn save_equities() {}

// fn save_probabilities() {}

// fn load_equities() {}

// fn load_probabilities() {}

// fn read_unique() {}

// fn write_unique() {}

fn run(args: &Args) -> io::Result<()> {
    let positions = match read_unique(args) {
        Ok(positions) => positions,
        Err(err) => {
            println!("Error reading unique file: {}", err);
            let positions = unqiue(args.verbose);
            write_unique(args, &positions)?;
            positions
        }
    };
    let reachable = positions.len();
    let (ongoing, gameover) = split_positions(positions);
    println!(
        "Posssible: {} Reachable: {} Ongoing: {} Gameover: {}",
        POSSIBLE,
        reachable,
        ongoing.len(),
        gameover.len()
    );
    let posmap = create_posmap(ongoing);
    println!("Position Map Created");
    let mut avg_err;
    let mut max_err;
    let mut equities = initial_equities(gameover);
    for iteration in 0..args.iterations {
        (equities, avg_err, max_err) = iterate_probabilities(&posmap, &equities);
        let probs = check_open(&equities);
        println!(
            "Iteration: {:03} Equity: {:.5} AvgErr: {:.5} MaxErr: {:.5}",
            iteration,
            probs.equity(),
            avg_err,
            max_err,
        );
        // println!(
        //     "Itr: {:03} Start Equity: {:.5} wn:{:.5} wg:{:.5} wb:{:.5} ln:{:.5} lg:{:.5} lb:{:.5}",
        //     iteration,
        //     probs.equity(),
        //     probs.win_n + probs.win_g + probs.win_b,
        //     probs.win_g + probs.win_b,
        //     probs.win_b,
        //     probs.lose_n + probs.lose_g + probs.lose_b,
        //     probs.lose_g + probs.lose_b,
        //     probs.lose_b,
        // );
    }
    println!("Writing to {}", args.file.display());
    write_file(&args, &equities)
}

fn main() -> io::Result<()> {
    let args = Args::parse();
    run(&args)
}

/*
BwAAAAYCAAAAAA, -1.24241
lAAAMAAAQAAAAA, +0.82902
AwAIQAkAAAAAAA, −1.76175
DABAAAAAcAAAAA, -1.73588
BQAAogIAAAAAAA, +2.06012
 */
