use std::{marker::PhantomData, path::PathBuf};

use bkgm::position::GameState::Ongoing;
use bkgm::State;
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::AutodiffModule,
    optim::{decay::WeightDecayConfig, AdamConfig, GradientsParams, Optimizer},
    record::NoStdTrainingRecorder,
    tensor::backend::AutodiffBackend,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    data::{OutcomeProb, PositionBatch, PositionBatcher, PositionItem},
    dicegen::{DiceGen, FastrandDice},
    evaluator::{Evaluator, GreedyEvaluator, NNEvaluator, RolloutEvaluator},
    model::EquityModel,
    probabilities::Probabilities,
};

#[derive(Config)]
pub struct SelfPlayConfig {
    #[config(default = 100000)]
    pub num_games: usize,

    #[config(default = 1000)]
    pub batch_size: usize,

    #[config(default = 100)]
    pub num_rollouts: usize,

    #[config(default = 0.1)]
    pub epsilon: f32,

    #[config(default = 1e-5)]
    pub learning_rate: f64,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

pub struct SelfPlay<B: AutodiffBackend, G: State, M: EquityModel<B> + AutodiffModule<B>> {
    model: M,
    phantom: PhantomData<G>,
    device: B::Device,
}

impl<B: AutodiffBackend, G: State, M: EquityModel<B> + AutodiffModule<B>> SelfPlay<B, G, M> {
    pub fn new(device: B::Device, model: M) -> Self {
        Self {
            model,
            phantom: PhantomData,
            device,
        }
    }

    fn find_positions<E: Evaluator<G>>(&self, evaluator: E, batch_size: usize) -> Vec<G> {
        let mut dice_gen = FastrandDice::new();
        let mut positions = vec![];
        loop {
            let mut pos = G::new();
            while pos.game_state() == Ongoing {
                let dice = dice_gen.roll();
                let possible = pos.possible_positions(&dice);
                let mut ongoing_games = possible
                    .into_iter()
                    .filter(|p| p.game_state() == Ongoing)
                    .collect();
                positions.append(&mut ongoing_games);
                if positions.len() >= batch_size {
                    return positions;
                }
                pos = evaluator.best_position(&pos, &dice);
            }
        }
    }

    fn rollout(&self, pos: &G, model: &M, num_rollouts: usize) -> Probabilities {
        let evaluator = NNEvaluator::new(self.device.clone(), model.clone());
        let rollout = RolloutEvaluator::new(evaluator, num_rollouts);
        rollout.eval(pos)
    }

    fn train<O: Optimizer<M, B>>(
        &self,
        model: M,
        optim: &mut O,
        batch: PositionBatch<B>,
        lr: f64,
    ) -> M {
        let reg = model.forward_step(batch);

        // Gradients for the current backward pass
        let grads = reg.loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);
        // Update the model using the optimizer.
        optim.step(lr, model, grads)
    }

    fn self_play(&self, model: &M, batch_size: usize, epsilon: f32) -> PositionBatch<B> {
        let batcher = PositionBatcher::<B>::new(self.device.clone());
        let evaluator = NNEvaluator::new(self.device.clone(), model.clone());
        let greedy = GreedyEvaluator::new(evaluator, epsilon);
        let positions = self.find_positions(greedy, batch_size);
        let items = positions
            .par_iter()
            .map(|&pos| {
                let probs = self.rollout(&pos, model, 100);
                PositionItem {
                    position: pos.position(),
                    probs: OutcomeProb::from(probs),
                }
            })
            .collect();
        batcher.batch(items)
    }

    pub fn learn(&self, model: M) {
        let config_optimizer =
            AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
        let config = SelfPlayConfig::new(config_optimizer);
        let rounds = config.num_games / config.batch_size;
        let mut model = model;
        let mut optim = config.optimizer.init();
        let model_path = PathBuf::from("model/self");
        for round in 0..rounds {
            let data = self.self_play(&model, config.batch_size, config.epsilon);
            model = self.train(model, &mut optim, data, config.learning_rate);
            println!("Round: {}", round);
            model
                .clone()
                .save_file(
                    format!("{}/round-{}", model_path.display(), round),
                    &NoStdTrainingRecorder::new(),
                )
                .expect("Failed to save trained model");
            // save model;
            // save rollout data;
            // save stats (num positions seen);
        }
        // save model;
    }
}
