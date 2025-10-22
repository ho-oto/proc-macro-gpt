mod attn;
mod dropout;
mod ff;
mod ln;
mod op;
mod softmax;
mod transformer;

use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Mul;
use std::ops::{Add, Div};
use std::path::Path;

use ndarray::prelude::*;
use ndarray_rand::{rand::rngs::StdRng, rand::SeedableRng};

use transformer::*;

#[derive(Debug, Clone, Copy)]
struct AdamConfig {
    beta1: f32,
    beta2: f32,
    eps: f32,
}

fn adam_step(
    grad: TransformerParams,
    p: TransformerParams,
    m: TransformerParams,
    v: TransformerParams,
    lr: f32,
    t: i32,
    config: AdamConfig,
) -> (TransformerParams, TransformerParams, TransformerParams) {
    let AdamConfig { beta1, beta2, eps } = config;
    let m = m.mul(beta1).add(&grad.mul(1.0 - beta1));
    let v = v.mul(beta2).add(&grad.pow2().mul(1.0 - beta2));
    let m_hat = m.mul(1.0 / (1.0 - beta1.powi(t)));
    let v_hat = v.mul(1.0 / (1.0 - beta2.powi(t)));
    (p.add(&m_hat.div(&v_hat.sqrt().add(eps)).mul(-lr)), m, v)
}

fn load_tokens(file_name: &str) -> Vec<String> {
    let json = std::fs::read_to_string(file_name).unwrap();
    serde_json::from_str::<serde_json::Value>(&json)
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap().to_owned())
        .collect()
}

fn main() -> () {
    let ref mut rng: StdRng = SeedableRng::seed_from_u64(12345);

    let model_conf = TransformerConfig {
        num_vocab: 8000,
        max_len: 128,
        num_layer: 8,
        num_head: 16,
        dim_emb: 64,
        dim_hidden: 64 * 4,
        layer_norm_eps: 1e-5,
    };
    let adam_conf = AdamConfig {
        beta1: 0.9,
        beta2: 0.95,
        eps: 1e-8,
    };
    let batch_size = 64;
    let dropout = 0.2;
    let lr = 2e-3;

    let mut p = TransformerParams::new(model_conf, rng);
    let mut m = p.mul(0.0);
    let mut v = p.mul(0.0);

    let data_seq_len = 1024;
    let vocab_size = 8000;
    let tokens = load_tokens(&format!("tokens.{vocab_size}.json"));
    let train_data = File::open(&format!(
        "tiny_stories.train.{data_seq_len}.{vocab_size}.bin"
    ))
    .unwrap();
    let mut reader = BufReader::new(train_data);
    let mut bytes_buff = vec![0u8; batch_size * data_seq_len * std::mem::size_of::<u16>()];

    let mut t = 1;
    loop {
        let Ok(()) = reader.read_exact(&mut bytes_buff) else {
            break;
        };
        let data = Array::from_iter(
            bytes_buff
                .chunks_exact(std::mem::size_of::<u16>())
                .map(|x| u16::from_ne_bytes(x.try_into().unwrap()) as usize),
        )
        .to_shape([batch_size, data_seq_len])
        .unwrap()
        .slice(s![.., ..model_conf.max_len])
        .to_owned();
        let start = std::time::Instant::now();
        let (loss, grad) = transformer_cross_entropy(p.clone(), data, &mut Some((dropout, rng)));
        (p, m, v) = adam_step(grad, p, m, v, lr, t, adam_conf);
        println!("step {t}: {}s, loss={loss}", start.elapsed().as_secs());
        if t % 100 == 0 {
            p.save(Path::new(&format!("p_{t}.bson")));
            m.save(Path::new(&format!("m_{t}.bson")));
            v.save(Path::new(&format!("v_{t}.bson")));
        }
        if t % 50 == 0 {
            let init = [104, 20];
            let decoded = transformer_greedy_decode(p.clone(), &init, 50);
            let init: Vec<_> = init.iter().map(|&i| tokens[i].clone()).collect();
            let decoded: Vec<_> = decoded.iter().map(|&i| tokens[i].clone()).collect();
            println!("{:?} -> {:?}", init, decoded);
        }
        t += 1;
    }
}
