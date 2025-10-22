use ndarray::prelude::*;
use ndarray_rand::rand_distr::Bernoulli;
use ndarray_rand::{rand::rngs::StdRng, RandomExt};

pub fn generate_dropout_mask<S: ShapeBuilder>(
    shape: S,
    dropout_prob: f64,
    rng: &mut StdRng,
) -> Array<f32, S::Dim> {
    let dist = Bernoulli::new(1.0 - dropout_prob).unwrap();
    let scaled = (1.0 / (1.0 - dropout_prob)) as f32;
    Array::random_using(shape, dist, rng).map(|x| if *x { scaled } else { 0.0 })
}
