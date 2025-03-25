use std::collections::HashSet;

use bkgm::{
    dice::{ALL_21, ALL_SINGLES},
    GameState, State, BACKGAMMON, HYPERGAMMON, HYPERGAMMON2, HYPERGAMMON4, HYPERGAMMON5,
    LONGGAMMON, NACKGAMMON,
};

fn unqiue(verbose: bool) {
    let position = HYPERGAMMON;
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
}

fn main() {
    unqiue(true);
}
