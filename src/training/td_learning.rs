use std::io::{stdout, Write};

use bkgm::{
    GameState::{GameOver, Ongoing},
    Hypergammon, State,
};
use burn::{
    module::Module,
    optim::{momentum::MomentumConfig, GradientsParams, Optimizer, SgdConfig},
    record::NoStdTrainingRecorder,
    tensor::backend::AutodiffBackend,
};

use crate::{
    dicegen::{DiceGen, FastrandDice},
    duel,
    evaluator::{HyperEvaluator, PartialEvaluator, RandomEvaluator},
    model::{FState, TDModel},
};

pub struct TDConfig {
    learning_rate: f64,
    td_decay: f64,
}

impl TDConfig {
    pub fn new(learning_rate: f64, td_decay: f64) -> Self {
        Self {
            learning_rate,
            td_decay,
        }
    }
}

pub struct TDTrainer<B: AutodiffBackend> {
    device: B::Device,
    optim: SgdConfig,
    config: TDConfig,
}

impl<B: AutodiffBackend> TDTrainer<B> {
    pub fn new(device: B::Device, config: TDConfig) -> Self {
        let optim = SgdConfig::new().with_momentum(Some(
            MomentumConfig::new()
                .with_dampening(0.0)
                .with_momentum(config.td_decay),
        ));
        Self {
            device,
            optim,
            config,
        }
    }

    fn get_grads_value<G: State>(
        &self,
        state: &FState<G>,
        model: &TDModel<B>,
    ) -> (f32, GradientsParams) {
        let state = if state.turn { *state } else { state.flip() };

        model.forward_grads_pos(&state.state, &self.device)
    }

    fn get_value<G: State>(&self, state: &FState<G>, model: &TDModel<B>) -> f32 {
        let state = if state.turn { *state } else { state.flip() };

        match state.game_state() {
            GameOver(result) => model.from_result(result),
            Ongoing => model.forward_pos(state.state, &self.device),
        }
    }

    fn train_game<G: State>(&mut self, model: TDModel<B>) -> TDModel<B> {
        let mut optim = self.optim.init();
        let mut model = model;

        // let hyper = HyperEvaluator::new().unwrap();

        let mut dicegen = FastrandDice::new();
        let mut dice = dicegen.first_roll();
        let mut state = FState::<Hypergammon>::new();

        while state.game_state() == Ongoing {
            let (cur_value, grads) = self.get_grads_value(&state, &model);
            // let cur_value = self.get_value(&state, &model);
            // let grads = GradientsParams::from_grads(cur_value.backward(), &model);
            state = model.best_position(&state, &dice);

            dice = dicegen.roll();
            let next_value = self.get_value(&state, &model);
            let td_error = next_value - cur_value;
            model = optim.step(-self.config.learning_rate * td_error as f64, model, grads);
        }

        model
    }

    pub fn train<G: State>(&mut self, model: TDModel<B>, num_episodes: usize) -> TDModel<B> {
        // self.train_one(model.clone());
        let mut model = model;
        let mut prev_model = model.clone();
        for ep in 1..=num_episodes {
            model = self.train_game::<G>(model.clone());
            if ep % 100 == 0 {
                print!("\rEpisode: {}", ep);
                stdout().flush().unwrap();
            }

            if ep % 2_000 == 0 {
                println!("Saving model");
                model
                    .clone()
                    .save_file(
                        format!("model/td/games-{}", ep),
                        &NoStdTrainingRecorder::new(),
                    )
                    .expect("Failed to save model");
            }
            if ep % 10_000 == 0 {
                // let probs = duel::duel::<FState<Hypergammon>>(model.clone(), prev_model, 1000);
                let probs =
                    duel::duel::<FState<Hypergammon>>(model.clone(), RandomEvaluator::new(), 1000);
                println!(
                    "Equity: {:.3} ({:.1}%). {:?}",
                    probs.equity(),
                    probs.win_prob() * 100.0,
                    probs,
                );
                prev_model = model.clone();
            }
        }
        model
    }
}
