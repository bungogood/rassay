use bkgm::{
    dice::{ALL_21, ALL_SINGLES},
    GameState, Position, State, BACKGAMMON, HYPERGAMMON, HYPERGAMMON2, HYPERGAMMON4, HYPERGAMMON5,
    LONGGAMMON, NACKGAMMON,
};
use clap::{Parser, ValueEnum};
use crossbeam::thread;
use dashmap::DashSet;
use std::sync::Arc;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose the backgammon variant to play
    #[arg(short = 'v', long = "variant")]
    variant: Variant,
}

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

pub fn dbhash<const N: u8>(position: &Position<N>, pre_mcomb: &Vec<Vec<u128>>) -> u128 {
    let points = 26;

    let mut x_remaining = (N - position.x_off) as usize;
    let mut o_remaining = (N - position.o_off) as usize;

    let mut x_index = if x_remaining > 0 {
        pre_mcomb[points][x_remaining - 1]
    } else {
        0
    };
    let mut o_index = if o_remaining > 0 {
        pre_mcomb[points][o_remaining - 1]
    } else {
        0
    };

    for i in 1..=24 {
        let n = position.pips[i];
        match n {
            n if n < 0 => o_remaining -= n.unsigned_abs() as usize,
            n if n > 0 => x_remaining -= n as usize,
            _ => {}
        }
        if o_remaining > 0 {
            o_index += pre_mcomb[points - i][o_remaining - 1];
        }
        if x_remaining > 0 {
            x_index += pre_mcomb[points - i][x_remaining - 1];
        }
    }

    x_index * pre_mcomb[points][N as usize] + o_index
}

pub const fn comb(n: usize, k: usize) -> u128 {
    match (n, k) {
        (0, _) => 0,
        (_, 0) => 1,
        (n, k) if n == k => 1,
        (n, k) if n < k => 0,
        (n, k) => (n as u128 * comb(n - 1, k - 1)) / k as u128,
    }
}

pub const fn mcomb(n: usize, k: usize) -> u128 {
    comb(n + k - 1, k)
}

pub fn precompute_mcomb(max_n: usize, max_k: usize) -> Vec<Vec<u128>> {
    let mut pre_mcomb = vec![vec![0; max_k + 1]; max_n + 1];
    for n in 0..=max_n {
        pre_mcomb[n][0] = 1;
        for k in 1..=max_k.min(n) {
            pre_mcomb[n][k] = mcomb(n, k)
        }
    }
    pre_mcomb
}

/// Fully expands all reachable positions from `position`
fn unique<const N: u8>(position: Position<N>, verbose: bool) {
    let num_threads = std::thread::available_parallelism().unwrap().get();

    let pre_mcomb = Arc::new(precompute_mcomb(26, N as usize));

    let found = Arc::new(DashSet::new());
    let mut new_positions = Vec::new();

    // Initial expansion using ALL_SINGLES
    for die in ALL_SINGLES {
        for child in position.possible_positions(&die) {
            let child_hash = dbhash(&child, &pre_mcomb);
            if found.insert(child_hash) {
                new_positions.push(child);
            }
        }
    }

    let mut depth = 1;
    if verbose {
        println!(
            "{}\t{}\tpositions reached after {} roll",
            new_positions.len(),
            found.len(),
            depth
        );
    }

    while !new_positions.is_empty() {
        // let queue = std::mem::take(&mut new_positions);
        let chunk_size = (new_positions.len() + num_threads - 1) / num_threads;

        new_positions = thread::scope(|s| {
            let mut handles = Vec::new();

            for i in 0..num_threads {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(new_positions.len());
                if start >= end {
                    continue;
                }

                let chunk = &new_positions[start..end];
                let found = Arc::clone(&found);
                let pre_mcomb = Arc::clone(&pre_mcomb);

                let handle = s.spawn(move |_| {
                    let mut local_new = Vec::new();

                    for pos in chunk {
                        // let pos: Position<N> = dbunhash(*pos_hash, &pre_mcomb);
                        if pos.game_state() == GameState::Ongoing {
                            for (die, _) in ALL_21 {
                                for child in pos.possible_positions(&die) {
                                    let child_hash = dbhash(&child, &pre_mcomb);
                                    if found.insert(child_hash) {
                                        local_new.push(child);
                                    }
                                }
                            }
                        }
                    }

                    local_new
                });

                handles.push(handle);
            }

            let mut new_positions = Vec::new();
            for handle in handles {
                new_positions.extend(handle.join().unwrap());
            }
            new_positions
        })
        .unwrap();

        depth += 1;

        if verbose {
            println!(
                "{}\t{}\tpositions reached after {} rolls",
                new_positions.len(),
                found.len(),
                depth
            );
        }
    }
}

fn main() {
    let args = Args::parse();

    match args.variant {
        Variant::BACKGAMMON => unique(BACKGAMMON, true),
        Variant::HYPERGAMMON => unique(HYPERGAMMON, true),
        Variant::HYPERGAMMON2 => unique(HYPERGAMMON2, true),
        Variant::HYPERGAMMON4 => unique(HYPERGAMMON4, true),
        Variant::HYPERGAMMON5 => unique(HYPERGAMMON5, true),
        Variant::LONGGAMMON => unique(LONGGAMMON, true),
        Variant::NACKGAMMON => unique(NACKGAMMON, true),
    };
}
