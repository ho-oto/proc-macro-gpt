use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use ndarray::prelude::*;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};

use ad_derive::*;

use super::attn::*;
use super::dropout::*;
use super::ff::*;
use super::ln::*;
use super::op::*;
use super::softmax::*;

#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    pub num_vocab: usize,
    pub max_len: usize,
    pub num_layer: usize,
    pub num_head: usize,
    pub dim_emb: usize,
    pub dim_hidden: usize,
    pub layer_norm_eps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerParams {
    pub embedding: Array2<f32>,
    pub pos_emb: Array2<f32>,
    pub attn: AttentionList,
    pub ln: LayerNormList,
    pub ffn: FeedForwardList,
    pub linear: Array2<f32>,
}

impl std::ops::Add<&TransformerParams> for &TransformerParams {
    type Output = TransformerParams;
    fn add(self, rhs: &TransformerParams) -> Self::Output {
        Self::Output {
            embedding: &self.embedding + &rhs.embedding,
            pos_emb: &self.pos_emb + &rhs.pos_emb,
            attn: &self.attn + &rhs.attn,
            ln: &self.ln + &rhs.ln,
            ffn: &self.ffn + &rhs.ffn,
            linear: &self.linear + &rhs.linear,
        }
    }
}

impl std::ops::Div<&TransformerParams> for &TransformerParams {
    type Output = TransformerParams;
    fn div(self, rhs: &TransformerParams) -> Self::Output {
        Self::Output {
            embedding: &self.embedding / &rhs.embedding,
            pos_emb: &self.pos_emb / &rhs.pos_emb,
            attn: &self.attn / &rhs.attn,
            ln: &self.ln / &rhs.ln,
            ffn: &self.ffn / &rhs.ffn,
            linear: &self.linear / &rhs.linear,
        }
    }
}

impl std::ops::Add<f32> for &TransformerParams {
    type Output = TransformerParams;
    fn add(self, rhs: f32) -> Self::Output {
        Self::Output {
            embedding: &self.embedding + rhs,
            pos_emb: &self.pos_emb + rhs,
            attn: &self.attn + rhs,
            ln: &self.ln + rhs,
            ffn: &self.ffn + rhs,
            linear: &self.linear + rhs,
        }
    }
}

impl std::ops::Mul<f32> for &TransformerParams {
    type Output = TransformerParams;
    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            embedding: &self.embedding * rhs,
            pos_emb: &self.pos_emb * rhs,
            attn: &self.attn * rhs,
            ln: &self.ln * rhs,
            ffn: &self.ffn * rhs,
            linear: &self.linear * rhs,
        }
    }
}

impl TransformerParams {
    pub fn new(config: TransformerConfig, rng: &mut StdRng) -> Self {
        let a = (config.dim_emb as f32).sqrt().powi(-1);
        Self {
            embedding: Array::random_using(
                [config.num_vocab, config.dim_emb],
                Normal::new(0.0, 1.0).unwrap(),
                rng,
            ),
            pos_emb: Array::random_using(
                [config.max_len, config.dim_emb],
                Normal::new(0.0, 1.0).unwrap(),
                rng,
            ),
            attn: AttentionList::new(config.num_layer, config.dim_emb, config.num_head, rng),
            ln: LayerNormList::new(
                2 * config.num_layer + 1,
                config.dim_emb,
                config.layer_norm_eps,
            ),
            ffn: FeedForwardList::new(config.num_layer, config.dim_emb, config.dim_hidden, rng),
            linear: Array::random_using(
                [config.dim_emb, config.num_vocab],
                Uniform::new(-a, a),
                rng,
            ),
        }
    }

    pub fn save(&self, path: &Path) -> () {
        let mut file = File::create(path).unwrap();
        file.write_all(&bson::to_vec(self).unwrap()).unwrap();
    }

    pub fn load(path: &Path) -> Self {
        let mut file = File::open(path).unwrap();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).unwrap();
        bson::from_slice::<Self>(&bytes).unwrap()
    }

    pub fn to_tuple(
        self,
    ) -> (
        Array2<f32>,
        Array2<f32>,
        AttentionList,
        LayerNormList,
        FeedForwardList,
        Array2<f32>,
    ) {
        (
            self.embedding,
            self.pos_emb,
            self.attn,
            self.ln,
            self.ffn,
            self.linear,
        )
    }

    pub fn from_tuple(
        (embedding, pos_emb, attn, ln, ffn, linear): (
            Array2<f32>,
            Array2<f32>,
            AttentionList,
            LayerNormList,
            FeedForwardList,
            Array2<f32>,
        ),
    ) -> Self {
        Self {
            embedding,
            pos_emb,
            attn,
            ln,
            ffn,
            linear,
        }
    }

    pub fn pow2(&self) -> Self {
        Self {
            embedding: self.embedding.pow2(),
            pos_emb: self.pos_emb.pow2(),
            attn: self.attn.pow2(),
            ln: self.ln.pow2(),
            ffn: self.ffn.pow2(),
            linear: self.linear.pow2(),
            ..*self
        }
    }

    pub fn sqrt(&self) -> Self {
        Self {
            embedding: self.embedding.sqrt(),
            pos_emb: self.pos_emb.sqrt(),
            attn: self.attn.sqrt(),
            ln: self.ln.sqrt(),
            ffn: self.ffn.sqrt(),
            linear: self.linear.sqrt(),
        }
    }
}

fn transformer(
    x: TransformerParams,
    data: Array2<usize>,
    dropout: &mut Option<(f64, &mut StdRng)>,
) -> (Array3<f32>, impl FnOnce(Array3<f32>) -> TransformerParams) {
    let (embedding, pos_emb, attn, ln, ffn, linear) = x.to_tuple();

    let (batch_size, seq) = data.dim();
    let (vocab, emb) = embedding.dim();
    let num_layers = attn.len();
    let data_in: Vec<_> = data.iter().cloned().collect();
    let emb_dropout = if let Some((prob, rng)) = dropout {
        generate_dropout_mask([batch_size, seq, emb], *prob, *rng)
    } else {
        Array::ones([batch_size, seq, emb])
    };
    let f = ad!(|(embedding, pos_emb, mut attn, mut ln, mut ffn, linear)| {
        let (e, __) = advanced_indexing_along(embedding, Axis(0), &data_in);
        let (e, __) = reshape(e, [batch_size, seq, emb]);

        let ((p, rest), __) = split_at(pos_emb, Axis(0), seq);
        let ((), __) = detach(rest);

        let (p, __) = reshape(p, [1, seq, emb]);

        let (x, __) = add((e, p));

        let (emb_dropout, __) = attach(emb_dropout)(());
        let (mut x, __) = mul((x, emb_dropout));

        let mut __;
        for l in 0..num_layers {
            let mut f_inner = ad!(|(x, attn, ln, ffn)| {
                let ((x, res), __) = fork(x);
                let ((x, ln), __) = LayerNormList::call((x, ln), 2 * l);
                let ((x, attn), __) = AttentionList::call((x, attn), l, dropout);
                let (x, __) = add((x, res));
                let ((x, res), __) = fork(x);
                let ((x, ln), __) = LayerNormList::call((x, ln), 2 * l + 1);
                let ((x, ffn), __) = FeedForwardList::call((x, ffn), l, dropout);
                let (x, __) = add((x, res));
                (x, attn, ln, ffn)
            });

            ((x, attn, ln, ffn), __) = f_inner((x, attn, ln, ffn));
        }

        let ((x, ln), __) = LayerNormList::call((x, ln), 2 * num_layers);

        let (x, __) = reshape(x, [batch_size * seq, emb]);

        let (logit, __) = mm((x, linear)); // (bat * seq, vocab)
        let (prob, __) = softmax(logit, Axis(1));
        let (prob, __) = reshape(prob, [batch_size, seq, vocab]);

        let ((), __) = LayerNormList::detach(ln);
        let ((), __) = AttentionList::detach(attn);
        let ((), __) = FeedForwardList::detach(ffn);

        prob
    });

    let (prob, pb) = f((embedding, pos_emb, attn, ln, ffn, linear));
    (prob, |x| TransformerParams::from_tuple(pb(x)))
}

pub fn transformer_cross_entropy(
    x: TransformerParams,
    data: Array2<usize>,
    dropout: &mut Option<(f64, &mut StdRng)>,
) -> (f32, TransformerParams) {
    let (batch_size, seq) = data.dim();
    let (vocab, _) = x.embedding.dim();
    let data_in = data.slice(s![.., ..-1]).to_owned();
    let data_out: Vec<_> = data.slice(s![.., 1..]).iter().cloned().collect();
    let seq_range: Vec<_> = (0..batch_size * (seq - 1)).collect();
    let f = ad!(|x| {
        let (prob, __) = transformer(x, data_in, dropout);
        let (prob, __) = reshape(prob, [batch_size * (seq - 1), vocab]);
        let (x, __) = advanced_indexing_zipped(prob, (Axis(0), &seq_range), (Axis(1), &data_out));
        let (x, __) = log(x);
        let (x, __) = div_num(x, -1.0);
        let (x, __) = sum(x);
        x
    });

    let (loss, pb) = f(x);
    let grad = pb(1.0);
    (loss, grad)
}

pub fn transformer_greedy_decode(x: TransformerParams, data: &[usize], n: usize) -> Vec<usize> {
    let mut data = data.to_vec();
    for _ in 0..n {
        let inp = Array::from_iter(data.iter().cloned())
            .to_shape([1, data.len()])
            .unwrap()
            .to_owned();
        let (prob, _) = transformer(x.clone(), inp, &mut None);
        let next = prob.slice(s![0, -1, ..]).argmax().unwrap();
        data.push(next);
    }
    data
}
