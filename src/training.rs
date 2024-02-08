use std::path::Path;

use crate::data::PositionBatcher;
use crate::dataset::PositionDataset;
#[allow(unused_imports)]
use crate::model::{LargeModel, RassayModel};

use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, NoStdTrainingRecorder};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{CpuMemory, CpuUse};
use burn::train::{MetricEarlyStoppingStrategy, StoppingCondition};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};

static ARTIFACT_DIR: &str = "logs";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device, test: impl AsRef<Path>, train: impl AsRef<Path>) {
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
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        // .metric_train_numeric(AccuracyMetric::new())
        // .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(LargeModel::new(&device), config.optimizer.init(), 1e-5);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(format!("model/model"), &NoStdTrainingRecorder::new())
        .expect("Failed to save trained model");
}
