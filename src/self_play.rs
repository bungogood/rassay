use std::path::PathBuf;

use crate::data::PositionBatcher;
use crate::dataset::PositionDataset;
use crate::model::EquityModel;
#[allow(unused_imports)]
use crate::model::{LargeModel, RassayModel};
use indicatif::{ProgressBar, ProgressStyle};

use burn::module::{AutodiffModule, Module};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::NoStdTrainingRecorder;
use burn::tensor::Data;
use burn::{config::Config, data::dataloader::DataLoaderBuilder, tensor::backend::AutodiffBackend};

static ARTIFACT_DIR: &str = "logs";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 128)]
    pub batch_size: usize,

    #[config(default = 8)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1e-5)]
    pub lr: f64,

    pub optimizer: AdamConfig,
}

const STYLE: &str =
    "{wide_bar} {pos}/{len} ({percent}%) Elapsed: {elapsed_precise} ETA: {eta_precise}";

pub fn run<B: AutodiffBackend>(
    device: B::Device,
    load_model: bool,
    model_path: &PathBuf,
    test: &PathBuf,
    train: &PathBuf,
) {
    // Config
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config = MnistTrainingConfig::new(config_optimizer);
    B::seed(config.seed);

    // Data
    let batcher_train = PositionBatcher::<B>::new(device.clone());
    let batcher_valid = PositionBatcher::<B::InnerBackend>::new(device.clone());

    let train_data = PositionDataset::new(train).unwrap();
    let test_data = PositionDataset::new(test).unwrap();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_data);
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_data);

    // Model
    let mut model = if load_model {
        // train_model(config, model, dataloader_train, dataloader_test, model_path);
        LargeModel::<B>::init_with(device, model_path)
    } else {
        LargeModel::<B>::default()
    };

    let mut optim = config.optimizer.init();

    let train_iters = dataloader_train.num_items() / config.batch_size;
    let test_iters = dataloader_test.num_items() / config.batch_size;

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop.
        let mut train_loss = 0.0;
        let mut test_loss = 0.0;
        let style = ProgressStyle::default_bar().template(STYLE).unwrap();
        let pb = ProgressBar::new(train_iters as u64).with_style(style);
        for batch in dataloader_train.iter() {
            let reg = model.forward_step(batch);

            let loss: Data<f64, 1> = reg.loss.to_data().convert();
            train_loss += loss.value[0];

            // Gradients for the current backward pass
            let grads = reg.loss.backward();
            // Gradients linked to each parameter of the model.
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer.
            model = optim.step(config.lr, model, grads);
            pb.inc(1);
        }
        pb.finish_and_clear();

        // Get the model without autodiff.
        let model_valid = model.valid();

        // Implement our validation loop.
        for batch in dataloader_test.iter() {
            let reg = model_valid.forward_step(batch);
            let loss: Data<f64, 1> = reg.loss.into_data().convert();
            test_loss += loss.value[0];
        }

        println!(
            "[Epoch {}] Train Loss {:.5}, Test Loss {:.5}",
            epoch,
            train_loss / train_iters as f64,
            test_loss / test_iters as f64
        );
    }

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model
        .save_file(
            format!("{}", model_path.display()),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
