mod model;

pub use model::Model;

pub mod mnist {
    include!(concat!(env!("OUT_DIR"), "/tmp/mnist.rs"));
}
