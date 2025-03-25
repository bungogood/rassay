use std::{collections::HashSet, iter::zip, sync::Arc};

use bkgm::{
    dice::{ALL_21, ALL_SINGLES},
    utils::mcomb,
    GameState, Position, State, HYPERGAMMON, HYPERGAMMON2, HYPERGAMMON4,
};
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::{
    iter::ParallelBridge,
    prelude::{IntoParallelRefIterator, ParallelIterator},
};

const STYLE: &str =
    "{wide_bar} {pos}/{len} ({percent}%) Elapsed: {elapsed_precise} ETA: {eta_precise}";

fn num_possible(checkers: u8) -> usize {
    mcomb(26, checkers as usize).pow(2)
}

fn unqiue<const N: u8>(position: &Position<N>, verbose: bool) -> Vec<Position<N>> {
    let position = *position;
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
                GameState::Ongoing => {
                    for (die, _) in ALL_21 {
                        let children = position.possible_positions(&die);
                        // let hash = position.dbhash();
                        for child in children {
                            if !found.contains(&child) {
                                found.insert(child);
                                new_positons.push(child);
                            }
                        }
                    }
                }
                GameState::GameOver(_) => {}
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

fn split_positions<const N: u8>(
    positions: Vec<Position<N>>,
) -> (Vec<Position<N>>, Vec<Position<N>>) {
    let mut ongoing = vec![];
    let mut gameover = vec![];
    for position in positions {
        match position.game_state() {
            GameState::Ongoing => ongoing.push(position),
            GameState::GameOver(_) => gameover.push(position),
        }
    }
    (ongoing, gameover)
}

fn initial_equities<const N: u8>(gameover: Vec<Position<N>>) -> Vec<f32> {
    let mut equities = vec![0.0; num_possible(N)];
    gameover.iter().for_each(|p| {
        equities[p.dbhash()] = match &p.game_state() {
            GameState::Ongoing => panic!("Should not be ongoing"),
            GameState::GameOver(result) => result.value(),
        }
    });
    equities
}

use std::thread;

struct ThreadMetrics {
    sum_err: f32,
    max_err: f32,
}

fn iterate_values_thread<const N: u8>(
    positions: &Vec<Position<N>>,
    values: &Vec<f32>,
    num_threads: usize,
) -> (Vec<f32>, f32, f32) {
    let positions = Arc::new(positions.clone());
    let prev_values = Arc::new(values.clone());
    let mut new_values = vec![0.0f32; values.len()];
    let new_values_ptr = new_values.as_mut_ptr() as usize;

    let chunk_size = (positions.len() + num_threads - 1) / num_threads;
    let mut handles = vec![];

    for thread_idx in 0..num_threads {
        let positions = Arc::clone(&positions);
        let prev_values = Arc::clone(&prev_values);
        let ptr_usize = new_values_ptr;

        let start = thread_idx * chunk_size;
        let end = ((thread_idx + 1) * chunk_size).min(positions.len());

        let handle = thread::spawn(move || -> ThreadMetrics {
            let ptr = ptr_usize as *mut f32;

            let mut sum_err = 0.0;
            let mut max_err = 0.0;

            for position in &positions[start..end] {
                let mut possibilities = 0.0;
                let mut total = 0.0;

                for (die, n) in ALL_21 {
                    let children = position.possible_positions(&die);
                    let best = children
                        .iter()
                        .map(|child| prev_values[child.dbhash()])
                        .min_by(|a, b| a.partial_cmp(&b).unwrap())
                        .unwrap();
                    possibilities += n as f32;
                    total += best * n as f32;
                }

                let value = total / possibilities;
                let idx = position.dbhash();

                // SAFETY: each thread writes only to unique indices
                unsafe {
                    *ptr.add(idx) = value;
                }

                // Error computation (compare to previous value)
                let prev = prev_values[idx];
                let delta = (prev - value).abs();
                sum_err += delta;
                max_err = f32::max(max_err, delta);
            }

            ThreadMetrics { sum_err, max_err }
        });

        handles.push(handle);
    }

    let mut total_sum_err = 0.0;
    let mut global_max_err = 0.0;

    for handle in handles {
        let ThreadMetrics { sum_err, max_err } = handle.join().unwrap();
        total_sum_err += sum_err;
        global_max_err = f32::max(global_max_err, max_err);
    }

    let avg_err = total_sum_err / values.len() as f32;

    (new_values, avg_err, global_max_err)
}

fn iterate_values<const N: u8>(
    positions: &Vec<Position<N>>,
    values: &Vec<f32>,
) -> (Vec<f32>, f32, f32) {
    let prev_values = Arc::new(values);

    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let new_updates: Vec<(usize, f32)> = positions
        .par_iter()
        .progress_with_style(style)
        .map(|position| {
            let mut possibilities = 0.0;
            let mut total = 0.0;
            for (die, n) in ALL_21 {
                let children = position.possible_positions(&die);
                let best = children
                    .iter()
                    .map(|child| prev_values[child.dbhash()])
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                    .unwrap();
                possibilities += n as f32;
                total += best * n as f32;
            }
            let idx = position.dbhash();
            let val = total / possibilities;
            (idx, val)
        })
        .collect();

    // Apply updates
    let mut new_values = values.clone();
    for (idx, val) in new_updates {
        new_values[idx] = val;
    }

    let (sum_err, max_err) = zip(prev_values.iter(), new_values.iter())
        .par_bridge()
        .map(|(a, b)| {
            let delta = (a - b).abs();
            (delta, delta)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, f32::max(a.1, b.1)));
    let avg_err = sum_err / new_values.len() as f32;

    (new_values, avg_err, max_err)
}

fn iterate_map_values(
    pos_map: &Vec<(usize, Vec<(f32, Vec<usize>)>)>,
    values: &Vec<f32>,
) -> (Vec<f32>, f32, f32) {
    let prev_values = Arc::new(values);

    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let new_updates: Vec<(usize, f32)> = pos_map
        .par_iter()
        .progress_with_style(style)
        .map(|(idx, pos)| {
            let mut possibilities = 0.0;
            let mut total = 0.0;
            for (n, children) in pos {
                let best = children
                    .iter()
                    .map(|child| prev_values[*child])
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                    .unwrap();
                possibilities += n;
                total += best * n;
            }
            let val = total / possibilities;
            (*idx, val)
        })
        .collect();

    // Apply updates
    let mut new_values = values.clone();
    for (idx, val) in new_updates {
        new_values[idx] = val;
    }

    let (sum_err, max_err) = zip(prev_values.iter(), new_values.iter())
        .par_bridge()
        .map(|(a, b)| {
            let delta = (a - b).abs();
            (delta, delta)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, f32::max(a.1, b.1)));
    let avg_err = sum_err / new_values.len() as f32;

    (new_values, avg_err, max_err)
}

fn check_open<const N: u8>(position: &Position<N>, equities: &Vec<f32>) -> f32 {
    let starting = *position;
    let mut possibilies = 0.0;
    let mut total = 0.0;
    for die in ALL_SINGLES {
        let children = starting.possible_positions(&die);
        let best = children
            .iter()
            .map(|child| equities[child.dbhash()])
            .min_by(|a, b| a.partial_cmp(&b).unwrap())
            .unwrap();
        possibilies += 1.0;
        total += best;
    }
    total / possibilies
}

fn create_posmap<const N: u8>(ongoing: Vec<Position<N>>) -> Vec<(usize, Vec<(f32, Vec<usize>)>)> {
    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let posmap = ongoing
        .par_iter()
        .progress_with_style(style)
        .map(|position| {
            let mut c = vec![];
            for (die, n) in ALL_21 {
                let children = position.possible_positions(&die);
                c.push((n as f32, children.iter().map(|pos| pos.dbhash()).collect()));
            }
            (position.dbhash(), c)
        })
        .collect();

    posmap
}

fn run<const N: u8>(position: &Position<N>) {
    let positions = unqiue(position, true);
    let reachable = positions.len();
    let (ongoing, gameover) = split_positions(positions);

    println!(
        "Posssible: {} Reachable: {} Ongoing: {} Gameover: {}",
        num_possible(N),
        reachable,
        ongoing.len(),
        gameover.len()
    );
    let mut avg_err;
    let mut max_err;
    let mut equities = initial_equities(gameover);
    let mut iteration = 1;
    loop {
        let num_threads = std::thread::available_parallelism().unwrap().get();
        (equities, avg_err, max_err) = iterate_values_thread(&ongoing, &equities, num_threads);
        let values = check_open(position, &equities);
        println!(
            "Iteration: {:03} Equity: {:.5} AvgErr: {:.5} MaxErr: {:.5}",
            iteration, values, avg_err, max_err,
        );
        if avg_err < 0.0001 {
            break;
        }
        iteration += 1;
    }
}

fn run_pos_map<const N: u8>(position: &Position<N>) {
    let positions = unqiue(position, true);
    let reachable = positions.len();
    let (ongoing, gameover) = split_positions(positions);

    println!(
        "Posssible: {} Reachable: {} Ongoing: {} Gameover: {}",
        num_possible(N),
        reachable,
        ongoing.len(),
        gameover.len()
    );

    let pos_map = create_posmap(ongoing);
    println!("Position Map Created");
    let mut avg_err;
    let mut max_err;
    let mut equities = initial_equities(gameover);
    let mut iteration = 1;
    loop {
        (equities, avg_err, max_err) = iterate_map_values(&pos_map, &equities);
        let values = check_open(position, &equities);
        println!(
            "Iteration: {:03} Equity: {:.5} AvgErr: {:.5} MaxErr: {:.5}",
            iteration, values, avg_err, max_err,
        );
        if avg_err < 0.0001 {
            break;
        }
        iteration += 1;
    }
}

fn main() {
    run(&HYPERGAMMON);
}
