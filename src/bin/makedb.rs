use std::{iter::zip, sync::Arc};

use bkgm::{
    dice::{ALL_21, ALL_SINGLES},
    utils::mcomb,
    GameState, Position, State, HYPERGAMMON, HYPERGAMMON2, HYPERGAMMON4,
};
use crossbeam::thread;
use dashmap::DashSet;
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

fn unique<const N: u8>(position: &Position<N>, verbose: bool) -> Vec<Position<N>> {
    let num_threads = std::thread::available_parallelism().unwrap().get();

    let found = Arc::new(DashSet::new());
    let mut new_positions = Vec::new();

    // Initial expansion using ALL_SINGLES
    for die in ALL_SINGLES {
        for child in position.possible_positions(&die) {
            if found.insert(child) {
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

                let handle = s.spawn(move |_| {
                    let mut local_new = Vec::new();

                    for pos in chunk {
                        if pos.game_state() == GameState::Ongoing {
                            for (die, _) in ALL_21 {
                                for child in pos.possible_positions(&die) {
                                    if found.insert(child) {
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

    Arc::try_unwrap(found).unwrap().into_iter().collect()
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

struct ThreadMetrics {
    sum_err: f32,
    max_err: f32,
}

fn iterate_values_with_refs<const N: u8>(
    positions: &Vec<Position<N>>,
    prev_values: &Vec<f32>,
    new_values: &mut Vec<f32>,
) -> (f32, f32) {
    let num_threads = std::thread::available_parallelism().unwrap().get();
    assert_eq!(prev_values.len(), new_values.len());

    let new_values_ptr = new_values.as_mut_ptr() as usize;

    let chunk_size = (positions.len() + num_threads - 1) / num_threads;

    let (total_sum_err, global_max_err) = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for thread_idx in 0..num_threads {
            let start = thread_idx * chunk_size;
            let end = ((thread_idx + 1) * chunk_size).min(positions.len());

            // SAFETY: each thread will only write to disjoint indices
            let handle = s.spawn(move |_| {
                let ptr = new_values_ptr as *mut f32;
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

                    let val = -total / possibilities;
                    let idx = position.dbhash();

                    unsafe {
                        *ptr.add(idx) = val;
                    }

                    let delta = (prev_values[idx] - val).abs();
                    sum_err += delta;
                    max_err = f32::max(max_err, delta);
                }

                ThreadMetrics { sum_err, max_err }
            });

            handles.push(handle);
        }

        handles.into_iter().map(|h| h.join().unwrap()).fold(
            (0.0f32, 0.0f32),
            |(sum_a, max_a), ThreadMetrics { sum_err, max_err }| {
                (sum_a + sum_err, max_a.max(max_err))
            },
        )
    })
    .unwrap();

    let avg_err = total_sum_err / prev_values.len() as f32;
    (avg_err, global_max_err)
}

fn iterate_values_map_refs(
    pos_map: &Vec<(u32, Vec<(f32, Vec<u32>)>)>,
    prev_values: &Vec<f32>,
    new_values: &mut Vec<f32>,
) -> (f32, f32) {
    let num_threads = std::thread::available_parallelism().unwrap().get();
    assert_eq!(prev_values.len(), new_values.len());

    let new_values_ptr = new_values.as_mut_ptr() as usize;

    let chunk_size = (pos_map.len() + num_threads - 1) / num_threads;

    let (total_sum_err, global_max_err) = thread::scope(|s| {
        let mut handles = Vec::with_capacity(num_threads);
        for thread_idx in 0..num_threads {
            let start = thread_idx * chunk_size;
            let end = ((thread_idx + 1) * chunk_size).min(pos_map.len());

            // SAFETY: each thread will only write to disjoint indices
            let handle = s.spawn(move |_| {
                let ptr = new_values_ptr as *mut f32;
                let mut sum_err = 0.0;
                let mut max_err = 0.0;

                for (idx, possible) in &pos_map[start..end] {
                    let mut possibilities = 0.0;
                    let mut total = 0.0;

                    for (n, children) in possible {
                        let best = children
                            .iter()
                            .map(|child| prev_values[*child as usize])
                            .min_by(|a, b| a.partial_cmp(&b).unwrap())
                            .unwrap();
                        possibilities += n;
                        total += best * n;
                    }

                    let val = -total / possibilities;

                    unsafe {
                        *ptr.add(*idx as usize) = val;
                    }

                    let delta = (prev_values[*idx as usize] - val).abs();
                    sum_err += delta;
                    max_err = f32::max(max_err, delta);
                }

                ThreadMetrics { sum_err, max_err }
            });

            handles.push(handle);
        }

        handles.into_iter().map(|h| h.join().unwrap()).fold(
            (0.0f32, 0.0f32),
            |(sum_a, max_a), ThreadMetrics { sum_err, max_err }| {
                (sum_a + sum_err, max_a.max(max_err))
            },
        )
    })
    .unwrap();

    let avg_err = total_sum_err / prev_values.len() as f32;
    (avg_err, global_max_err)
}

fn iterate_map_values(
    pos_map: &Vec<(u32, Vec<(f32, Vec<u32>)>)>,
    values: &Vec<f32>,
) -> (Vec<f32>, f32, f32) {
    let prev_values = Arc::new(values);

    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let new_updates: Vec<(u32, f32)> = pos_map
        .par_iter()
        .progress_with_style(style)
        .map(|(idx, pos)| {
            let mut possibilities = 0.0;
            let mut total = 0.0;
            for (n, children) in pos {
                let best = children
                    .iter()
                    .map(|child| prev_values[*child as usize])
                    .min_by(|a, b| a.partial_cmp(&b).unwrap())
                    .unwrap();
                possibilities += n;
                total += best * n;
            }
            let val = -total / possibilities;
            (*idx, val)
        })
        .collect();

    // Apply updates
    let mut new_values = values.clone();
    for (idx, val) in new_updates {
        new_values[idx as usize] = val;
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
    -total / possibilies
}

fn create_posmap<const N: u8>(ongoing: &Vec<Position<N>>) -> Vec<(u32, Vec<(f32, Vec<u32>)>)> {
    let style = ProgressStyle::default_bar().template(STYLE).unwrap();

    let posmap = ongoing
        .par_iter()
        .progress_with_style(style)
        .map(|position| {
            let mut c = vec![];
            for (die, n) in ALL_21 {
                let children = position.possible_positions(&die);
                c.push((
                    n as f32,
                    children.iter().map(|pos| pos.dbhash() as u32).collect(),
                ));
            }
            (position.dbhash() as u32, c)
        })
        .collect();

    posmap
}

fn run<const N: u8>(position: &Position<N>) {
    let start = std::time::Instant::now();
    let positions = unique(position, true);
    let elapsed = start.elapsed();
    println!("Unique Positions: {} Time: {:?}", positions.len(), elapsed);
    let reachable = positions.len();
    let (ongoing, gameover) = split_positions(positions);

    println!(
        "Posssible: {} Reachable: {} Ongoing: {} Gameover: {}",
        num_possible(N),
        reachable,
        ongoing.len(),
        gameover.len()
    );
    // let pos_map_start = std::time::Instant::now();
    // let pos_map = create_posmap(&ongoing);
    // let pos_map_elapsed = pos_map_start.elapsed();
    // let pos_map_size = pos_map.iter().fold(0, |acc, (idx, children)| {
    //     acc + std::mem::size_of_val(idx)
    //         + children.iter().fold(0, |acc, (n, children)| {
    //             acc + std::mem::size_of_val(n) + children.len() * std::mem::size_of::<usize>()
    //         })
    // });
    // println!("Pos Map Size: {} Time: {:?}", pos_map_size, pos_map_elapsed);
    let mut avg_err;
    let mut max_err;
    let mut equities = initial_equities(gameover);
    let mut iteration = 1;
    let iteration_start = std::time::Instant::now();
    loop {
        let mut new_values = equities.clone();
        (avg_err, max_err) = iterate_values_with_refs(&ongoing, &equities, &mut new_values);
        // (avg_err, max_err) = iterate_values_map_refs(&pos_map, &equities, &mut new_values);
        equities = new_values;
        let values = check_open(position, &equities);
        // println!(
        //     "Iteration: {:03} Equity: {:.5} AvgErr: {:.5} MaxErr: {:.5}",
        //     iteration, values, avg_err, max_err,
        // );
        println!("{},{},{},{}", iteration, values, avg_err, max_err,);
        if max_err < 0.0001 {
            break;
        }
        iteration += 1;
    }
    let iteration_elapsed = iteration_start.elapsed();
    let total_elapsed = start.elapsed();
    println!("Total Time: {:?}", total_elapsed);
    println!("Iteration Time: {:?}", iteration_elapsed);
}

fn main() {
    run(&HYPERGAMMON4);
}
