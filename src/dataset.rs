use crate::data::PositionItem;
use burn::data::dataset::{Dataset, InMemDataset};
use std::path::Path;

pub struct PositionDataset {
    pub dataset: InMemDataset<PositionItem>,
}

impl PositionDataset {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let rdr = csv::ReaderBuilder::new();
        let dataset = InMemDataset::from_csv(path, &rdr).unwrap();
        Ok(Self { dataset })
    }
}

impl Dataset<PositionItem> for PositionDataset {
    fn get(&self, index: usize) -> Option<PositionItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
