use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use ad_derive::*;

use super::dropout::*;
use super::op::*;
use super::softmax::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionList {
    pub w_qkv: Array3<f32>,
    pub w_out: Array3<f32>,
    pub b_out: Array2<f32>,
    pub num_head: usize,
    size: usize,
}

impl std::ops::Add<&AttentionList> for &AttentionList {
    type Output = AttentionList;
    fn add(self, rhs: &AttentionList) -> Self::Output {
        Self::Output {
            w_qkv: &self.w_qkv + &rhs.w_qkv,
            w_out: &self.w_out + &rhs.w_out,
            b_out: &self.b_out + &rhs.b_out,
            ..*self
        }
    }
}

impl std::ops::Div<&AttentionList> for &AttentionList {
    type Output = AttentionList;
    fn div(self, rhs: &AttentionList) -> Self::Output {
        Self::Output {
            w_qkv: &self.w_qkv / &rhs.w_qkv,
            w_out: &self.w_out / &rhs.w_out,
            b_out: &self.b_out / &rhs.b_out,
            ..*self
        }
    }
}

impl std::ops::Add<f32> for &AttentionList {
    type Output = AttentionList;
    fn add(self, rhs: f32) -> Self::Output {
        Self::Output {
            w_qkv: &self.w_qkv + rhs,
            w_out: &self.w_out + rhs,
            b_out: &self.b_out + rhs,
            ..*self
        }
    }
}

impl std::ops::Mul<f32> for &AttentionList {
    type Output = AttentionList;
    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            w_qkv: &self.w_qkv * rhs,
            w_out: &self.w_out * rhs,
            b_out: &self.b_out * rhs,
            ..*self
        }
    }
}

impl AttentionList {
    pub fn new(size: usize, emb: usize, num_head: usize, rng: &mut StdRng) -> Self {
        let a = (1.5 / (emb as f32)).sqrt();
        let w_qkv = Array::random_using([size, emb, 3 * emb], Uniform::new(-a, a), rng);
        let w_out = Array::random_using([size, emb, emb], Uniform::new(-a, a), rng);
        let b_out = Array::zeros([size, emb]);
        Self {
            w_qkv,
            w_out,
            b_out,
            num_head,
            size,
        }
    }

    pub fn zeros(size: usize, emb: usize, num_head: usize) -> Self {
        let w_qkv = Array::zeros([size, emb, 3 * emb]);
        let w_out = Array::zeros([size, emb, emb]);
        let b_out = Array::zeros([size, emb]);
        Self {
            w_qkv,
            w_out,
            b_out,
            num_head,
            size,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn pow2(&self) -> Self {
        Self {
            w_qkv: self.w_qkv.pow2(),
            w_out: self.w_out.pow2(),
            b_out: self.b_out.pow2(),
            ..*self
        }
    }

    pub fn sqrt(&self) -> Self {
        Self {
            w_qkv: self.w_qkv.sqrt(),
            w_out: self.w_out.sqrt(),
            b_out: self.b_out.sqrt(),
            ..*self
        }
    }

    pub fn get(
        self,
        idx: usize,
    ) -> (
        ((Array2<f32>, Array2<f32>, Array1<f32>), Self),
        impl FnOnce(((Array2<f32>, Array2<f32>, Array1<f32>), Self)) -> Self,
    ) {
        let w_qkv = self.w_qkv.index_axis(Axis(0), idx).to_owned();
        let w_out = self.w_out.index_axis(Axis(0), idx).to_owned();
        let b_out = self.b_out.index_axis(Axis(0), idx).to_owned();
        (
            ((w_qkv, w_out, b_out), self),
            move |((w_qkv, w_out, b_out), mut params)| {
                Zip::from(params.w_qkv.index_axis_mut(Axis(0), idx))
                    .and(&w_qkv)
                    .par_for_each(|w_qkv_mut, &w_qkv| *w_qkv_mut += w_qkv);
                Zip::from(params.w_out.index_axis_mut(Axis(0), idx))
                    .and(&w_out)
                    .par_for_each(|w_out_mut, &w_out| *w_out_mut += w_out);
                Zip::from(params.b_out.index_axis_mut(Axis(0), idx))
                    .and(&b_out)
                    .par_for_each(|b_out_mut, &b_out| *b_out_mut += b_out);
                params
            },
        )
    }

    pub fn detach(mut self) -> ((), impl FnOnce(()) -> Self) {
        self.w_qkv.par_mapv_inplace(|_| 0.0);
        self.w_out.par_mapv_inplace(|_| 0.0);
        self.b_out.par_mapv_inplace(|_| 0.0);
        ((), move |()| self)
    }

    pub fn call(
        (x, params): (Array3<f32>, AttentionList),
        idx: usize,
        dropout: &mut Option<(f64, &mut StdRng)>,
    ) -> (
        (Array3<f32>, AttentionList),
        impl FnOnce((Array3<f32>, AttentionList)) -> (Array3<f32>, AttentionList),
    ) {
        let (bat, seq, emb) = x.dim();
        let num_heads = params.num_head;
        let scaling = ((emb / num_heads) as f32).sqrt();
        let mask_causal =
            Array::from_shape_fn(
                [1, seq, seq],
                |(_, t, s)| {
                    if t < s {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                },
            );
        let mask_dropout_attn = if let Some((prob, rng)) = dropout {
            generate_dropout_mask([bat * num_heads, seq, seq], *prob, *rng)
        } else {
            Array::ones([bat * num_heads, seq, seq])
        };
        let mask_dropout_resid = if let Some((prob, rng)) = dropout {
            generate_dropout_mask([bat, seq, emb], *prob, *rng)
        } else {
            Array::ones([bat, seq, emb])
        };
        let f = ad!(|(x, p)| {
            let (((w_qkv, w_out, b_out), p), __) = AttentionList::get(p, idx);
            let (b_out, __) = reshape(b_out, [1, emb]);

            let (x, __) = reshape(x, [bat * seq, emb]); // x = Q = K = V
            let (qkv, __) = mm((x, w_qkv)); // qkv = cat(QW^Q,KW^K,VW^V)
            let ((q, kv), __) = split_at(qkv, Axis(1), emb);
            let ((k, v), __) = split_at(kv, Axis(1), emb);

            let (q, __) = reshape(q, [bat, seq, num_heads, emb / num_heads]);
            let (q, __) = swap_axes(q, Axis(1), Axis(2));
            let (q, __) = reshape(q, [bat * num_heads, seq, emb / num_heads]); // q[(b,h),..] = QWₕ^Q

            let (k, __) = reshape(k, [bat, seq, num_heads, emb / num_heads]);
            let (k, __) = swap_axes(k, Axis(1), Axis(2));
            let (k, __) = reshape(k, [bat * num_heads, seq, emb / num_heads]); // k[(b,h),..] = KWₕ^K

            let (v, __) = reshape(v, [bat, seq, num_heads, emb / num_heads]);
            let (v, __) = swap_axes(v, Axis(1), Axis(2));
            let (v, __) = reshape(v, [bat * num_heads, seq, emb / num_heads]); // v[(b,h),..] = VWₕ^V

            let (kt, __) = swap_axes(k, Axis(1), Axis(2));
            let (a, __) = bmm((q, kt)); // a[(b,h),..] = (QWₕ^Q)(KWₕ^K)ᵀ
            let (a, __) = div_num(a, scaling); // a[(b,h),..] = (QWₕ^Q)(KWₕ^K)ᵀ/√dₖ
            let (a, __) = reshape(a, [bat * num_heads, seq, seq]); // (b,h),t,s

            let (m, __) = attach(mask_causal)(());
            let (a, __) = add((m, a));

            let (a, __) = softmax(a, Axis(2)); // a[(b,h),..] = softmax((QWₕ^Q)(KWₕ^K)ᵀ/√dₖ)

            let (m, __) = attach(mask_dropout_attn)(());
            let (a, __) = mul((a, m));

            let (a, __) = bmm((a, v)); // a[(b,h),..] = headₕ = softmax((QWₕ^Q)(KWₕ^K)ᵀ/√dₖ)(VWₕ^V)
            let (a, __) = reshape(a, [bat, num_heads, seq, emb / num_heads]);
            let (a, __) = swap_axes(a, Axis(1), Axis(2));
            let (a, __) = reshape(a, [bat * seq, emb]); // a[(b,t),..] = cat(head₁,…,headₕ)

            let (a, __) = mm((a, w_out));
            let (a, __) = add((a, b_out));
            let (a, __) = reshape(a, [bat, seq, emb]);

            let (m, __) = attach(mask_dropout_resid)(());
            let (a, __) = mul((a, m));

            (a, p)
        });

        f((x, params))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test() {
        /*
        x = torch.tensor(
            [
                [[1.93, 1.49, 0.90, -2.11, 1.31, 1.29], [0.68, -1.23, -0.04, -1.604, 0.58, -1.11]],
                [[-0.75, 1.65, -0.39, -1.40, 0.29, 1.41], [-0.72, -0.56, -0.77, 0.76, -0.55, -0.27]]
            ],
            dtype=torch.float32
        )
        w_qkv = torch.tensor(
            [
                [ 0.52,  0.39, -0.58, -0.17,  1.93,  1.01, -1.44, -1.13,  0.27, -0.56, 0.68,  0.44,  1.14,  0.02, -1.81,  0.93,  1.48,  0.34],
                [-1.42, -0.12, -0.97,  0.96,  1.62,  1.45, -0.43,  0.26, -1.44,  0.52, 0.35,  0.97, -0.47,  1.60, -0.43, -1.34, -0.19,  0.65],
                [-1.90,  0.23,  0.02, -0.35, -1.65,  0.68,  1.46, -0.31, -1.60,  1.35, 1.29,  0.05, -1.30,  0.05, -0.59, -0.39,  0.04,  0.12],
                [-0.81, -0.21, -1.16, -0.96, -0.37,  0.80, -0.52, -1.50, -1.93,  0.13, -0.96, -1.25, -0.75, -0.59,  1.77, -0.92,  0.96, -0.34],
                [-0.04,  0.24, -0.71, -0.72,  0.53,  2.11, -0.52, -0.93,  1.06,  0.21, -0.58,  0.33, -0.81, -1.02, -0.49, -0.59,  0.40,  0.63],
                [ 0.31, -0.03, -0.46, -0.06, -1.37,  0.33, -0.98,  0.30,  0.18, -0.13, -1.58,  2.25,  1.00,  1.36,  0.63,  0.41,  0.34, -0.22]
            ],
            dtype=torch.float32
        )
        w_out = torch.tensor(
            [
                [ 0.17,  1.05,  0.01, -0.08,  0.64,  0.57],
                [ 0.59, -0.02, -0.91,  1.48, -0.91, -0.53],
                [-0.81,  0.52, -0.71,  0.22,  0.56,  1.86],
                [ 1.04, -0.86,  0.80,  0.91, -0.09,  0.34],
                [ 0.97, -1.02, -0.54, -0.44,  0.25,  0.08],
                [-0.21,  2.17,  2.02,  0.25,  0.94,  0.71]
            ],
            dtype=torch.float32
        )
        b_out = torch.tensor([ 0.97,  1.01, -0.03, -1.01, -1.23, -1.05], dtype=torch.float32)
        attn = lambda x, w_qkv, w_out, b_out: nn.functional.multi_head_attention_forward(
            query=x.transpose(0, 1),
            key=x.transpose(0, 1),
            value=x.transpose(0, 1),
            embed_dim_to_check=6,
            num_heads=2,
            in_proj_weight=w_qkv.T,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=w_out.T,
            out_proj_bias=b_out,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=torch.nn.Transformer.generate_square_subsequent_mask(2),
            use_separate_proj_weight=False,
            average_attn_weights=True,
            is_causal=True,
        )[0].transpose(0, 1)
        y = attn(x, w_qkv, w_out, b_out)
        jx, jw_qkv, jw_out, jb_out = torch.autograd.functional.jacobian(attn, (x, w_qkv, w_out, b_out))
        */
        let x = array![
            [
                [1.93, 1.49, 0.90, -2.11, 1.31, 1.29],
                [0.68, -1.23, -0.04, -1.604, 0.58, -1.11]
            ],
            [
                [-0.75, 1.65, -0.39, -1.40, 0.29, 1.41],
                [-0.72, -0.56, -0.77, 0.76, -0.55, -0.27]
            ]
        ];
        #[rustfmt::skip]
        let w_qkv = array![[
            [0.52, 0.39, -0.58, -0.17, 1.93, 1.01, -1.44, -1.13, 0.27, -0.56, 0.68, 0.44, 1.14, 0.02, -1.81, 0.93, 1.48, 0.34],
            [-1.42, -0.12, -0.97, 0.96, 1.62, 1.45, -0.43, 0.26, -1.44, 0.52, 0.35, 0.97, -0.47, 1.60, -0.43, -1.34, -0.19, 0.65],
            [-1.90, 0.23, 0.02, -0.35, -1.65, 0.68, 1.46, -0.31, -1.60, 1.35, 1.29, 0.05, -1.30, 0.05, -0.59, -0.39, 0.04, 0.12],
            [-0.81, -0.21, -1.16, -0.96, -0.37, 0.80, -0.52, -1.50, -1.93, 0.13, -0.96, -1.25, -0.75, -0.59, 1.77, -0.92, 0.96, -0.34],
            [-0.04, 0.24, -0.71, -0.72, 0.53, 2.11, -0.52, -0.93, 1.06, 0.21, -0.58, 0.33, -0.81, -1.02, -0.49, -0.59, 0.40, 0.63],
            [0.31, -0.03, -0.46, -0.06, -1.37, 0.33, -0.98, 0.30, 0.18, -0.13, -1.58, 2.25, 1.00, 1.36, 0.63, 0.41, 0.34, -0.22]
        ]];
        let w_out = array![[
            [0.17, 1.05, 0.01, -0.08, 0.64, 0.57],
            [0.59, -0.02, -0.91, 1.48, -0.91, -0.53],
            [-0.81, 0.52, -0.71, 0.22, 0.56, 1.86],
            [1.04, -0.86, 0.80, 0.91, -0.09, 0.34],
            [0.97, -1.02, -0.54, -0.44, 0.25, 0.08],
            [-0.21, 2.17, 2.02, 0.25, 0.94, 0.71]
        ]];
        let b_out = array![[0.97, 1.01, -0.03, -1.01, -1.23, -1.05]];

        let mut p = AttentionList::zeros(1, 6, 2);
        p.w_qkv.assign(&w_qkv);
        p.w_out.assign(&w_out);
        p.b_out.assign(&b_out);

        let ((y_actual, _), _) = AttentionList::call((x.clone(), p.clone()), 0, &mut None);

        assert!((&y_actual - y_expected()).abs().iter().all(|x| *x < 5e-4));

        let mut jx_actual = Array::zeros((2, 2, 6, 2, 2, 6));
        let mut jw_qkv_actual = Array::zeros((2, 2, 6, 6, 18));
        let mut jw_out_actual = Array::zeros((2, 2, 6, 6, 6));
        let mut jb_out_actual = Array::zeros((2, 2, 6, 6));
        for b in 0..2 {
            for s in 0..2 {
                for e in 0..6 {
                    let mut y_adj = Array::zeros((2, 2, 6));
                    y_adj[(b, s, e)] = 1.0;
                    let p_adj = AttentionList::zeros(1, 6, 2);
                    let (_, pb) = AttentionList::call((x.clone(), p.clone()), 0, &mut None);
                    let (jx_elem, jp_elem) = pb((y_adj, p_adj));
                    jx_actual
                        .slice_mut(s![b, s, e, .., .., ..])
                        .assign(&jx_elem);
                    jw_qkv_actual
                        .slice_mut(s![b, s, e, .., ..])
                        .assign(&jp_elem.w_qkv.slice(s![0, .., ..]));
                    jw_out_actual
                        .slice_mut(s![b, s, e, .., ..])
                        .assign(&jp_elem.w_out.slice(s![0, .., ..]));
                    jb_out_actual
                        .slice_mut(s![b, s, e, ..])
                        .assign(&jp_elem.b_out.slice(s![0, ..]));
                }
            }
        }
        assert!(
            (&jx_actual - jx_expected())
                .iter()
                .map(|x| { *x * *x })
                .sum::<f32>()
                < 1e-8
        );
        assert!(
            (&jw_qkv_actual - jw_qkv_expected())
                .iter()
                .map(|x| { *x * *x })
                .sum::<f32>()
                < 1e-8
        );
        assert!(
            (&jw_out_actual - jw_out_expected())
                .iter()
                .map(|x| { *x * *x })
                .sum::<f32>()
                < 1e-8
        );
        assert!(
            (&jb_out_actual - jb_out_expected())
                .iter()
                .map(|x| { *x * *x })
                .sum::<f32>()
                < 1e-8
        );
    }

    fn y_expected() -> Array3<f32> {
        array![
            [
                [12.49850, 2.82700, 8.19862, 4.23080, -5.13101, -14.68762],
                [5.4929609, -1.0281358, 9.8033991, -3.6382098, -0.0934732, -5.7976961]
            ],
            [
                [1.3692590, 7.1946115, -1.4183381, 6.4694557, -5.0009022, -4.4294786],
                [-1.8211997, 0.6558988, -4.0658793, -0.9596822, -0.5933415, 3.4493053]
            ]
        ]
    }

    fn jx_expected() -> Array6<f32> {
        let mut jx_expected = Array::zeros((2, 2, 6, 2, 2, 6));
        jx_expected
            .slice_mut(s![0, 0, .., 0, 0, ..])
            .assign(&array![
                [4.0031004, -0.5020001, -0.1056000, -1.8635001, -0.7005000, 1.2645000],
                [-1.3162000, 2.0076001, -1.1177999, -0.7811000, 0.3816000, 0.1736000],
                [1.9099000, -0.8118001, 0.2692000, -2.6684999, 1.8526000, -1.9749001],
                [-0.1797000, 1.3377001, -0.2943000, -1.7684000, -2.1080000, 2.2399001],
                [0.3036999, -1.3135002, -1.0500000, 1.0513000, 0.8807000, -0.4035000],
                [-2.0513999, -1.9250002, -1.9090998, 2.7000000, -0.5538001, 1.0314000]
            ]);
        jx_expected
            .slice_mut(s![1, 0, .., 1, 0, ..])
            .assign(&array![
                [4.0031004, -0.5020001, -0.1056000, -1.8635001, -0.7005000, 1.2645000],
                [-1.3162000, 2.0076001, -1.1177999, -0.7811000, 0.3816000, 0.1736000],
                [1.9099000, -0.8118001, 0.2692000, -2.6684999, 1.8526000, -1.9749001],
                [-0.1797000, 1.3377001, -0.2943000, -1.7684000, -2.1080000, 2.2399001],
                [0.3036999, -1.3135002, -1.0500000, 1.0513000, 0.8807000, -0.4035000],
                [-2.0513999, -1.9250002, -1.9090998, 2.7000000, -0.5538001, 1.0314000]
            ]);
        jx_expected
            .slice_mut(s![0, 1, .., 0, 0, ..])
            .assign(&array![
                [
                    7.4609314e-05,
                    -2.5221513e-04,
                    -8.7444801e-05,
                    -2.2871733e-04,
                    1.6496080e-05,
                    1.6487768e-04
                ],
                [
                    2.3659479e-05,
                    3.1286152e-05,
                    5.4911920e-04,
                    -7.7587683e-06,
                    -2.5909182e-04,
                    -1.6054808e-03
                ],
                [
                    1.9818376e-04,
                    -1.5136224e-05,
                    2.4539762e-04,
                    -3.7412727e-05,
                    -1.3252403e-04,
                    -7.4588391e-04
                ],
                [
                    -1.9970766e-04,
                    -1.5449520e-04,
                    -2.4246181e-04,
                    -4.2221500e-04,
                    1.1194777e-04,
                    5.6686217e-04
                ],
                [
                    2.7147852e-04,
                    5.2946776e-05,
                    3.2411003e-04,
                    2.6519923e-04,
                    -1.7950579e-04,
                    -8.8417419e-04
                ],
                [
                    2.4835477e-04,
                    1.0908268e-04,
                    1.2397763e-04,
                    2.8306496e-04,
                    -9.6038610e-05,
                    -3.0980719e-04
                ]
            ]);
        jx_expected
            .slice_mut(s![1, 1, .., 1, 0, ..])
            .assign(&array![
                [-0.1924103, -0.6437430, 0.6076143, -0.6819100, 0.2798941, -0.6228314],
                [0.1137912, 0.0285952, -0.3469005, 0.0892292, -0.2190022, 0.3203181],
                [0.3500482, 0.1225236, -0.1482686, 0.0328669, 0.0713041, -0.0030260],
                [-0.5646762, -0.5779894, 0.6395299, -0.5067978, 0.1383037, -0.4646311],
                [0.4020768, 0.5345942, -0.7514250, 0.5569745, -0.3143232, 0.6624738],
                [0.2365140, 0.9153301, -1.1019449, 1.0526628, -0.6071634, 1.1610301]
            ]);
        jx_expected
            .slice_mut(s![0, 1, .., 0, 1, ..])
            .assign(&array![
                [4.0028224, -0.5018668, -0.1054827, -1.8632457, -0.7009012, 1.2642393],
                [-1.3143210, 2.0109761, -1.1166867, -0.7796299, 0.3862704, 0.1761679],
                [1.9106364, -0.8102277, 0.2696722, -2.6678321, 1.8548578, -1.9736879],
                [-0.1802908, 1.3367515, -0.2944829, -1.7683527, -2.1098347, 2.2389441],
                [0.3046093, -1.3116843, -1.0495007, 1.0517585, 0.8835309, -0.4020275],
                [-2.0511322, -1.9245135, -1.9090517, 2.6998696, -0.5526246, 1.0319592]
            ]);
        jx_expected
            .slice_mut(s![1, 1, .., 1, 1, ..])
            .assign(&array![
                [4.0646420, 0.1054314, 1.0805100, -1.9790165, -1.1038883, 1.2228720],
                [-1.4076439, 1.9852329, -1.0768967, -0.7342751, 0.6217117, -0.0333554],
                [1.6067322, -0.9213008, -0.2249267, -2.4157708, 1.8255463, -1.7339021],
                [0.2384545, 1.8750494, 1.0746095, -2.1544738, -2.3845582, 1.9605396],
                [0.0338298, -1.8114152, -2.1106853, 1.2999226, 1.3197770, -0.3947053],
                [-2.0972886, -2.7874503, -3.4200673, 2.8089371, 0.2332381, 0.8382807]
            ]);
        jx_expected
    }

    fn jw_qkv_expected() -> Array5<f32> {
        let mut jw_qkb_expected = Array::zeros((2, 2, 6, 6, 18));
        jw_qkb_expected
            .slice_mut(s![.., 0, .., .., 12..18])
            .assign(&array![
                [
                    [
                        [
                            3.2810000e-01,
                            1.1386999e+00,
                            -1.5633000e+00,
                            2.0071998e+00,
                            1.8721000e+00,
                            -4.0529996e-01
                        ],
                        [
                            2.5330001e-01,
                            8.7909997e-01,
                            -1.2069000e+00,
                            1.5496000e+00,
                            1.4453001e+00,
                            -3.1290001e-01
                        ],
                        [
                            1.5300000e-01,
                            5.3099996e-01,
                            -7.2899997e-01,
                            9.3599993e-01,
                            8.7300003e-01,
                            -1.8900000e-01
                        ],
                        [
                            -3.5869998e-01,
                            -1.2448999e+00,
                            1.7090999e+00,
                            -2.1943998e+00,
                            -2.0467000e+00,
                            4.4309998e-01
                        ],
                        [
                            2.2270000e-01,
                            7.7289993e-01,
                            -1.0611000e+00,
                            1.3623999e+00,
                            1.2707000e+00,
                            -2.7509999e-01
                        ],
                        [
                            2.1930000e-01,
                            7.6109993e-01,
                            -1.0448999e+00,
                            1.3415999e+00,
                            1.2513000e+00,
                            -2.7089998e-01
                        ]
                    ],
                    [
                        [
                            2.0264997e+00,
                            -3.8599998e-02,
                            1.0035999e+00,
                            -1.6597999e+00,
                            -1.9685999e+00,
                            4.1880999e+00
                        ],
                        [
                            1.5645000e+00,
                            -2.9800000e-02,
                            7.7480000e-01,
                            -1.2814001e+00,
                            -1.5197999e+00,
                            3.2333002e+00
                        ],
                        [
                            9.4499993e-01,
                            -1.7999999e-02,
                            4.6799996e-01,
                            -7.7399999e-01,
                            -9.1799998e-01,
                            1.9530001e+00
                        ],
                        [
                            -2.2154999e+00,
                            4.2199995e-02,
                            -1.0971999e+00,
                            1.8146000e+00,
                            2.1521997e+00,
                            -4.5787001e+00
                        ],
                        [
                            1.3754998e+00,
                            -2.6199998e-02,
                            6.8119997e-01,
                            -1.1266000e+00,
                            -1.3361999e+00,
                            2.8427000e+00
                        ],
                        [
                            1.3544999e+00,
                            -2.5799999e-02,
                            6.7079997e-01,
                            -1.1094000e+00,
                            -1.3158000e+00,
                            2.7993000e+00
                        ]
                    ],
                    [
                        [
                            1.9299999e-02,
                            -1.7563000e+00,
                            -1.3702999e+00,
                            1.5440000e+00,
                            -1.0422000e+00,
                            3.8985999e+00
                        ],
                        [
                            1.4900000e-02,
                            -1.3559000e+00,
                            -1.0579000e+00,
                            1.1920000e+00,
                            -8.0460006e-01,
                            3.0098000e+00
                        ],
                        [
                            8.9999996e-03,
                            -8.1900001e-01,
                            -6.3899994e-01,
                            7.1999997e-01,
                            -4.8600000e-01,
                            1.8180000e+00
                        ],
                        [
                            -2.1099998e-02,
                            1.9201000e+00,
                            1.4980999e+00,
                            -1.6880000e+00,
                            1.1394000e+00,
                            -4.2621999e+00
                        ],
                        [
                            1.3099999e-02,
                            -1.1920999e+00,
                            -9.3009990e-01,
                            1.0480000e+00,
                            -7.0740002e-01,
                            2.6461999e+00
                        ],
                        [
                            1.2900000e-02,
                            -1.1739000e+00,
                            -9.1589993e-01,
                            1.0319999e+00,
                            -6.9660002e-01,
                            2.6057999e+00
                        ]
                    ],
                    [
                        [
                            -1.5439999e-01,
                            2.8564000e+00,
                            4.2459998e-01,
                            1.7563000e+00,
                            -8.4919995e-01,
                            4.8249999e-01
                        ],
                        [
                            -1.1920000e-01,
                            2.2052000e+00,
                            3.2780001e-01,
                            1.3559000e+00,
                            -6.5560001e-01,
                            3.7250000e-01
                        ],
                        [
                            -7.1999997e-02,
                            1.3320000e+00,
                            1.9800000e-01,
                            8.1900001e-01,
                            -3.9600000e-01,
                            2.2499999e-01
                        ],
                        [
                            1.6879998e-01,
                            -3.1227999e+00,
                            -4.6419996e-01,
                            -1.9201000e+00,
                            9.2839992e-01,
                            -5.2749997e-01
                        ],
                        [
                            -1.0479999e-01,
                            1.9388000e+00,
                            2.8819999e-01,
                            1.1920999e+00,
                            -5.7639998e-01,
                            3.2749999e-01
                        ],
                        [
                            -1.0320000e-01,
                            1.9092000e+00,
                            2.8379998e-01,
                            1.1739000e+00,
                            -5.6759995e-01,
                            3.2249999e-01
                        ]
                    ],
                    [
                        [
                            1.2351999e+00,
                            -1.7563000e+00,
                            1.0807999e+00,
                            -1.7370000e-01,
                            4.8249999e-01,
                            1.8141999e+00
                        ],
                        [
                            9.5359999e-01,
                            -1.3559000e+00,
                            8.3440000e-01,
                            -1.3410001e-01,
                            3.7250000e-01,
                            1.4006000e+00
                        ],
                        [
                            5.7599998e-01,
                            -8.1900001e-01,
                            5.0400001e-01,
                            -8.1000000e-02,
                            2.2499999e-01,
                            8.4599996e-01
                        ],
                        [
                            -1.3503999e+00,
                            1.9201000e+00,
                            -1.1816000e+00,
                            1.8990000e-01,
                            -5.2749997e-01,
                            -1.9833999e+00
                        ],
                        [
                            8.3839995e-01,
                            -1.1920999e+00,
                            7.3359996e-01,
                            -1.1790000e-01,
                            3.2749999e-01,
                            1.2313999e+00
                        ],
                        [
                            8.2559997e-01,
                            -1.1739000e+00,
                            7.2240001e-01,
                            -1.1610000e-01,
                            3.2249999e-01,
                            1.2126000e+00
                        ]
                    ],
                    [
                        [
                            1.1000999e+00,
                            -1.0228999e+00,
                            3.5897999e+00,
                            6.5619999e-01,
                            1.5439999e-01,
                            1.3702999e+00
                        ],
                        [
                            8.4929997e-01,
                            -7.8969997e-01,
                            2.7714000e+00,
                            5.0660002e-01,
                            1.1920000e-01,
                            1.0579000e+00
                        ],
                        [
                            5.1299995e-01,
                            -4.7699997e-01,
                            1.6740000e+00,
                            3.0599999e-01,
                            7.1999997e-02,
                            6.3899994e-01
                        ],
                        [
                            -1.2026999e+00,
                            1.1182998e+00,
                            -3.9245999e+00,
                            -7.1739995e-01,
                            -1.6879998e-01,
                            -1.4980999e+00
                        ],
                        [
                            7.4669993e-01,
                            -6.9429994e-01,
                            2.4366000e+00,
                            4.4540000e-01,
                            1.0479999e-01,
                            9.3009990e-01
                        ],
                        [
                            7.3529994e-01,
                            -6.8369997e-01,
                            2.3994000e+00,
                            4.3860000e-01,
                            1.0320000e-01,
                            9.1589993e-01
                        ]
                    ]
                ],
                [
                    [
                        [
                            -1.2750000e-01,
                            -4.4250000e-01,
                            6.0750002e-01,
                            -7.7999997e-01,
                            -7.2750002e-01,
                            1.5750000e-01
                        ],
                        [
                            2.8049999e-01,
                            9.7349995e-01,
                            -1.3364999e+00,
                            1.7160000e+00,
                            1.6005000e+00,
                            -3.4649998e-01
                        ],
                        [
                            -6.6299997e-02,
                            -2.3009998e-01,
                            3.1590000e-01,
                            -4.0559998e-01,
                            -3.7830001e-01,
                            8.1899993e-02
                        ],
                        [
                            -2.3800001e-01,
                            -8.2599998e-01,
                            1.1339999e+00,
                            -1.4560000e+00,
                            -1.3580000e+00,
                            2.9400000e-01
                        ],
                        [
                            4.9300000e-02,
                            1.7109999e-01,
                            -2.3490000e-01,
                            3.0159998e-01,
                            2.8130001e-01,
                            -6.0899995e-02
                        ],
                        [
                            2.3969999e-01,
                            8.3189994e-01,
                            -1.1421000e+00,
                            1.4663999e+00,
                            1.3677000e+00,
                            -2.9609999e-01
                        ]
                    ],
                    [
                        [
                            -7.8749996e-01,
                            1.5000000e-02,
                            -3.8999999e-01,
                            6.4499998e-01,
                            7.6499999e-01,
                            -1.6275001e+00
                        ],
                        [
                            1.7324998e+00,
                            -3.3000000e-02,
                            8.5799998e-01,
                            -1.4190000e+00,
                            -1.6830000e+00,
                            3.5805001e+00
                        ],
                        [
                            -4.0949997e-01,
                            7.7999993e-03,
                            -2.0279999e-01,
                            3.3539999e-01,
                            3.9779997e-01,
                            -8.4630001e-01
                        ],
                        [
                            -1.4699999e+00,
                            2.7999999e-02,
                            -7.2799999e-01,
                            1.2040000e+00,
                            1.4280000e+00,
                            -3.0380001e+00
                        ],
                        [
                            3.0449998e-01,
                            -5.7999999e-03,
                            1.5079999e-01,
                            -2.4939999e-01,
                            -2.9580000e-01,
                            6.2930000e-01
                        ],
                        [
                            1.4804999e+00,
                            -2.8199999e-02,
                            7.3319995e-01,
                            -1.2126000e+00,
                            -1.4382000e+00,
                            3.0597000e+00
                        ]
                    ],
                    [
                        [
                            -7.4999998e-03,
                            6.8250000e-01,
                            5.3249997e-01,
                            -6.0000002e-01,
                            4.0500003e-01,
                            -1.5150000e+00
                        ],
                        [
                            1.6500000e-02,
                            -1.5015000e+00,
                            -1.1715000e+00,
                            1.3200001e+00,
                            -8.9100003e-01,
                            3.3329999e+00
                        ],
                        [
                            -3.8999997e-03,
                            3.5490000e-01,
                            2.7689999e-01,
                            -3.1200001e-01,
                            2.1060000e-01,
                            -7.8779995e-01
                        ],
                        [
                            -1.4000000e-02,
                            1.2740000e+00,
                            9.9399996e-01,
                            -1.1200000e+00,
                            7.5600004e-01,
                            -2.8279998e+00
                        ],
                        [
                            2.9000000e-03,
                            -2.6390001e-01,
                            -2.0589998e-01,
                            2.3199999e-01,
                            -1.5660000e-01,
                            5.8579999e-01
                        ],
                        [
                            1.4099999e-02,
                            -1.2831000e+00,
                            -1.0010999e+00,
                            1.1280000e+00,
                            -7.6139998e-01,
                            2.8481998e+00
                        ]
                    ],
                    [
                        [
                            5.9999999e-02,
                            -1.1100000e+00,
                            -1.6499999e-01,
                            -6.8250000e-01,
                            3.2999998e-01,
                            -1.8750000e-01
                        ],
                        [
                            -1.3200000e-01,
                            2.4419999e+00,
                            3.6300001e-01,
                            1.5015000e+00,
                            -7.2600001e-01,
                            4.1249999e-01
                        ],
                        [
                            3.1199997e-02,
                            -5.7720000e-01,
                            -8.5800000e-02,
                            -3.5490000e-01,
                            1.7160000e-01,
                            -9.7499996e-02
                        ],
                        [
                            1.1200000e-01,
                            -2.0720000e+00,
                            -3.0800000e-01,
                            -1.2740000e+00,
                            6.1600000e-01,
                            -3.4999999e-01
                        ],
                        [
                            -2.3200000e-02,
                            4.2919999e-01,
                            6.3800000e-02,
                            2.6390001e-01,
                            -1.2760000e-01,
                            7.2499998e-02
                        ],
                        [
                            -1.1279999e-01,
                            2.0867999e+00,
                            3.1020001e-01,
                            1.2831000e+00,
                            -6.2040001e-01,
                            3.5249999e-01
                        ]
                    ],
                    [
                        [
                            -4.7999999e-01,
                            6.8250000e-01,
                            -4.2000002e-01,
                            6.7500003e-02,
                            -1.8750000e-01,
                            -7.0499998e-01
                        ],
                        [
                            1.0560000e+00,
                            -1.5015000e+00,
                            9.2399997e-01,
                            -1.4850001e-01,
                            4.1249999e-01,
                            1.5510000e+00
                        ],
                        [
                            -2.4959998e-01,
                            3.5490000e-01,
                            -2.1839999e-01,
                            3.5100002e-02,
                            -9.7499996e-02,
                            -3.6659998e-01
                        ],
                        [
                            -8.9599997e-01,
                            1.2740000e+00,
                            -7.8399998e-01,
                            1.2600000e-01,
                            -3.4999999e-01,
                            -1.3160000e+00
                        ],
                        [
                            1.8560000e-01,
                            -2.6390001e-01,
                            1.6239999e-01,
                            -2.6100000e-02,
                            7.2499998e-02,
                            2.7260000e-01
                        ],
                        [
                            9.0239996e-01,
                            -1.2831000e+00,
                            7.8959996e-01,
                            -1.2690000e-01,
                            3.5249999e-01,
                            1.3254000e+00
                        ]
                    ],
                    [
                        [
                            -4.2750001e-01,
                            3.9749998e-01,
                            -1.3950000e+00,
                            -2.5500000e-01,
                            -5.9999999e-02,
                            -5.3249997e-01
                        ],
                        [
                            9.4049996e-01,
                            -8.7449992e-01,
                            3.0690000e+00,
                            5.6099999e-01,
                            1.3200000e-01,
                            1.1715000e+00
                        ],
                        [
                            -2.2229999e-01,
                            2.0669998e-01,
                            -7.2539997e-01,
                            -1.3259999e-01,
                            -3.1199997e-02,
                            -2.7689999e-01
                        ],
                        [
                            -7.9799998e-01,
                            7.4199992e-01,
                            -2.6040001e+00,
                            -4.7600001e-01,
                            -1.1200000e-01,
                            -9.9399996e-01
                        ],
                        [
                            1.6530000e-01,
                            -1.5369999e-01,
                            5.3939998e-01,
                            9.8600000e-02,
                            2.3200000e-02,
                            2.0589998e-01
                        ],
                        [
                            8.0369997e-01,
                            -7.4729997e-01,
                            2.6225998e+00,
                            4.7939998e-01,
                            1.1279999e-01,
                            1.0010999e+00
                        ]
                    ]
                ]
            ]);
        jw_qkb_expected
            .slice_mut(s![0, 1, .., .., ..])
            .assign(&array![
                [
                    [
                        -7.2049705e-05,
                        -3.4669958e-06,
                        -5.1208783e-05,
                        -2.8735318e-05,
                        1.1933641e-05,
                        -1.5491496e-04,
                        1.0144823e-04,
                        2.9797928e-05,
                        8.9998292e-05,
                        2.7904325e-06,
                        -5.3905249e-05,
                        4.6216945e-05,
                        1.1560129e-01,
                        4.0120444e-01,
                        -5.5080611e-01,
                        7.0731324e-01,
                        6.5970564e-01,
                        -1.4282286e-01
                    ],
                    [
                        1.3032521e-04,
                        6.2711838e-06,
                        9.2627655e-05,
                        5.1977120e-05,
                        -2.1585851e-05,
                        2.8021383e-04,
                        2.2022912e-04,
                        6.4686908e-05,
                        1.9537301e-04,
                        6.1092160e-06,
                        -1.1801711e-04,
                        1.0118477e-04,
                        -2.0909721e-01,
                        -7.2569025e-01,
                        9.9628663e-01,
                        -1.2789536e+00,
                        -1.1928703e+00,
                        2.5825024e-01
                    ],
                    [
                        4.2382180e-06,
                        2.0394093e-07,
                        3.0122812e-06,
                        1.6903127e-06,
                        -7.0197882e-07,
                        9.1126440e-06,
                        7.6182805e-05,
                        2.2376831e-05,
                        6.7584442e-05,
                        2.1059818e-06,
                        -4.0683106e-05,
                        3.4880624e-05,
                        -6.7990352e-03,
                        -2.3596650e-02,
                        3.2395400e-02,
                        -4.1514844e-02,
                        -3.8720578e-02,
                        8.3828047e-03
                    ],
                    [
                        1.6995253e-04,
                        8.1780308e-06,
                        1.2079248e-04,
                        6.7781541e-05,
                        -2.8149352e-05,
                        3.6541704e-04,
                        -4.1322313e-05,
                        -1.2137415e-05,
                        -3.6658472e-05,
                        -1.1113090e-06,
                        2.1468128e-05,
                        -1.8406212e-05,
                        -2.7268052e-01,
                        -9.4636172e-01,
                        1.2992424e+00,
                        -1.6682057e+00,
                        -1.5559227e+00,
                        3.3684924e-01
                    ],
                    [
                        -6.1454157e-05,
                        -2.9571434e-06,
                        -4.3678079e-05,
                        -2.4509534e-05,
                        1.0178694e-05,
                        -1.3213334e-04,
                        5.9281007e-05,
                        1.7412342e-05,
                        5.2590269e-05,
                        1.6270999e-06,
                        -3.1432119e-05,
                        2.6949076e-05,
                        9.8600745e-02,
                        3.4220257e-01,
                        -4.6980351e-01,
                        6.0326606e-01,
                        5.6266165e-01,
                        -1.2181334e-01
                    ],
                    [
                        1.1761055e-04,
                        5.6593608e-06,
                        8.3590807e-05,
                        4.6906182e-05,
                        -1.9479914e-05,
                        2.5287588e-04,
                        1.9431504e-04,
                        5.7075282e-05,
                        1.7238372e-04,
                        5.3908243e-06,
                        -1.0413930e-04,
                        8.9286295e-05,
                        -1.8869753e-01,
                        -6.5489143e-01,
                        8.9908820e-01,
                        -1.1541826e+00,
                        -1.0764972e+00,
                        2.3305608e-01
                    ]
                ],
                [
                    [
                        8.7995168e-06,
                        4.1911869e-07,
                        6.2036711e-06,
                        2.8692323e-04,
                        -1.1670431e-04,
                        1.5510113e-03,
                        -1.2439806e-05,
                        -3.6538884e-06,
                        -1.1035791e-05,
                        -2.8050159e-05,
                        5.4186967e-04,
                        -4.6458482e-04,
                        7.1400785e-01,
                        -1.3600150e-02,
                        3.5360390e-01,
                        -5.8489364e-01,
                        -6.9371098e-01,
                        1.4758363e+00
                    ],
                    [
                        -1.5916774e-05,
                        -7.5811175e-07,
                        -1.1221347e-05,
                        -5.1899353e-04,
                        2.1109750e-04,
                        -2.8055059e-03,
                        -2.6873135e-05,
                        -7.8933253e-06,
                        -2.3840104e-05,
                        -6.1046550e-05,
                        1.1792899e-03,
                        -1.0110922e-03,
                        -1.2914826e+00,
                        2.4599671e-02,
                        -6.3959140e-01,
                        1.0575962e+00,
                        1.2543582e+00,
                        -2.6685860e+00
                    ],
                    [
                        -5.1761862e-07,
                        -2.4654041e-08,
                        -3.6492182e-07,
                        -1.6877837e-05,
                        6.8649588e-06,
                        -9.1235961e-05,
                        -9.3148765e-06,
                        -2.7360168e-06,
                        -8.2635543e-06,
                        -2.1095631e-05,
                        4.0752289e-04,
                        -3.4939943e-04,
                        -4.1994035e-02,
                        7.9988647e-04,
                        -2.0797048e-02,
                        3.4329582e-02,
                        4.0716480e-02,
                        -8.6622320e-02
                    ],
                    [
                        -2.0756508e-05,
                        -9.8862699e-07,
                        -1.4633365e-05,
                        -6.7680125e-04,
                        2.7528487e-04,
                        -3.6585620e-03,
                        5.1316883e-06,
                        1.5073076e-06,
                        4.5525012e-06,
                        1.1350094e-05,
                        -2.1925977e-04,
                        1.8798762e-04,
                        -1.6842029e+00,
                        3.2080058e-02,
                        -8.3408153e-01,
                        1.3794779e+00,
                        1.6361247e+00,
                        -3.4807756e+00
                    ],
                    [
                        7.5054700e-06,
                        3.5748360e-07,
                        5.2913665e-06,
                        2.4472864e-04,
                        -9.9541903e-05,
                        1.3229215e-03,
                        -7.2780672e-06,
                        -2.1377541e-06,
                        -6.4566293e-06,
                        -1.6380658e-05,
                        3.1643963e-04,
                        -2.7130701e-04,
                        6.0900450e-01,
                        -1.1600087e-02,
                        3.0160227e-01,
                        -4.9885467e-01,
                        -5.9166479e-01,
                        1.2587379e+00
                    ],
                    [
                        -1.4363917e-05,
                        -6.8414965e-07,
                        -1.0126581e-05,
                        -4.6836000e-04,
                        1.9050261e-04,
                        -2.5317981e-03,
                        -2.3709803e-05,
                        -6.9641737e-06,
                        -2.1033802e-05,
                        -5.3864685e-05,
                        1.0405516e-03,
                        -8.9214160e-04,
                        -1.1654847e+00,
                        2.2199709e-02,
                        -5.7719243e-01,
                        9.5442021e-01,
                        1.1319866e+00,
                        -2.4082465e+00
                    ]
                ],
                [
                    [
                        3.7936799e-05,
                        1.8507237e-06,
                        2.7258975e-05,
                        1.3985661e-04,
                        -5.6549739e-05,
                        7.5659069e-04,
                        -5.3124641e-05,
                        -1.5604061e-05,
                        -4.7128742e-05,
                        -1.3698342e-05,
                        2.6462294e-04,
                        -2.2688073e-04,
                        6.8000751e-03,
                        -6.1880684e-01,
                        -4.8280534e-01,
                        5.4408711e-01,
                        -3.6725882e-01,
                        1.3738199e+00
                    ],
                    [
                        -6.8620982e-05,
                        -3.3476326e-06,
                        -4.9306676e-05,
                        -2.5297591e-04,
                        1.0228850e-04,
                        -1.3685391e-03,
                        -1.1609744e-04,
                        -3.4100780e-05,
                        -1.0299413e-04,
                        -2.9762532e-05,
                        5.7494902e-04,
                        -4.9294624e-04,
                        -1.2299836e-02,
                        1.1192850e+00,
                        8.7328833e-01,
                        -9.8381042e-01,
                        6.6407210e-01,
                        -2.4841213e+00
                    ],
                    [
                        -2.2315764e-06,
                        -1.0886610e-07,
                        -1.6034691e-06,
                        -8.2268589e-06,
                        3.3264550e-06,
                        -4.4505334e-05,
                        -4.0051109e-05,
                        -1.1764031e-05,
                        -3.5530749e-05,
                        -1.0291984e-05,
                        1.9881934e-04,
                        -1.7046246e-04,
                        -3.9994324e-04,
                        3.6394835e-02,
                        2.8395969e-02,
                        -3.1934496e-02,
                        2.1555787e-02,
                        -8.0634601e-02
                    ],
                    [
                        -8.9486217e-05,
                        -4.3655305e-06,
                        -6.4299114e-05,
                        -3.2989704e-04,
                        1.3339084e-04,
                        -1.7846639e-03,
                        2.1260545e-05,
                        6.2447634e-06,
                        1.8860983e-05,
                        5.5671858e-06,
                        -1.0754621e-04,
                        9.2207294e-05,
                        -1.6040029e-02,
                        1.4596428e+00,
                        1.1388421e+00,
                        -1.2832352e+00,
                        8.6618382e-01,
                        -3.2401690e+00
                    ],
                    [
                        3.2357857e-05,
                        1.5785583e-06,
                        2.3250303e-05,
                        1.1928945e-04,
                        -4.8233600e-05,
                        6.4532732e-04,
                        -3.0991167e-05,
                        -9.1028951e-06,
                        -2.7493355e-05,
                        -8.0028740e-06,
                        1.5459856e-04,
                        -1.3254872e-04,
                        5.8000437e-03,
                        -5.2780396e-01,
                        -4.1180310e-01,
                        4.6405083e-01,
                        -3.1323436e-01,
                        1.1717284e+00
                    ],
                    [
                        -6.1926250e-05,
                        -3.0210342e-06,
                        -4.4496268e-05,
                        -2.2829534e-04,
                        9.2309136e-05,
                        -1.2350230e-03,
                        -1.0244346e-04,
                        -3.0090254e-05,
                        -9.0881200e-05,
                        -2.6260646e-05,
                        5.0730002e-04,
                        -4.3494572e-04,
                        -1.1099854e-02,
                        1.0100868e+00,
                        7.8808969e-01,
                        -8.8783270e-01,
                        5.9928715e-01,
                        -2.2417777e+00
                    ]
                ],
                [
                    [
                        -9.4483541e-05,
                        -4.5544025e-06,
                        -6.7246088e-05,
                        -1.1670009e-04,
                        4.7506732e-05,
                        -6.3077419e-04,
                        1.3294441e-04,
                        3.9049166e-05,
                        1.1793967e-04,
                        1.1405790e-05,
                        -2.2033569e-04,
                        1.8891002e-04,
                        -5.4400600e-02,
                        1.0064112e+00,
                        1.4960165e-01,
                        6.1889911e-01,
                        -2.9924789e-01,
                        1.7002721e-01
                    ],
                    [
                        1.7090405e-04,
                        8.2381102e-06,
                        1.2163631e-04,
                        2.1108988e-04,
                        -8.5931300e-05,
                        1.1409592e-03,
                        2.8884446e-04,
                        8.4840984e-05,
                        2.5624409e-04,
                        2.4828680e-05,
                        -4.7963747e-04,
                        4.1122857e-04,
                        9.8398685e-02,
                        -1.8203758e+00,
                        -2.7059639e-01,
                        -1.1190845e+00,
                        5.4109573e-01,
                        -3.0744076e-01
                    ],
                    [
                        5.5578553e-06,
                        2.6790602e-07,
                        3.9556521e-06,
                        6.8647109e-06,
                        -2.7945136e-06,
                        3.7104364e-05,
                        9.9884142e-05,
                        2.9338518e-05,
                        8.8610737e-05,
                        8.5791235e-06,
                        -1.6573048e-04,
                        1.4209298e-04,
                        3.1995459e-03,
                        -5.9191599e-02,
                        -8.7987510e-03,
                        -3.6325492e-02,
                        1.7563973e-02,
                        -9.9795293e-03
                    ],
                    [
                        2.2287000e-04,
                        1.0743031e-05,
                        1.5862165e-04,
                        2.7527491e-04,
                        -1.1206000e-04,
                        1.4878850e-03,
                        -5.4032924e-05,
                        -1.5870848e-05,
                        -4.7934511e-05,
                        -4.6123159e-06,
                        8.9100191e-05,
                        -7.6392142e-05,
                        1.2832023e-01,
                        -2.3739245e+00,
                        -3.5288066e-01,
                        -1.4596801e+00,
                        7.0577931e-01,
                        -4.0101099e-01
                    ],
                    [
                        -8.0588899e-05,
                        -3.8846370e-06,
                        -5.7356956e-05,
                        -9.9538309e-05,
                        4.0520448e-05,
                        -5.3801324e-04,
                        7.7669407e-05,
                        2.2813485e-05,
                        6.8903260e-05,
                        6.6603279e-06,
                        -1.2866342e-04,
                        1.1031263e-04,
                        -4.6400350e-02,
                        8.5840648e-01,
                        1.2760095e-01,
                        5.2785784e-01,
                        -2.5522795e-01,
                        1.4501588e-01
                    ],
                    [
                        1.5423050e-04,
                        7.4343920e-06,
                        1.0976935e-04,
                        1.9049573e-04,
                        -7.7547753e-05,
                        1.0296461e-03,
                        2.5485872e-04,
                        7.4858501e-05,
                        2.2609414e-04,
                        2.1907746e-05,
                        -4.2321123e-04,
                        3.6285020e-04,
                        8.8798836e-02,
                        -1.6427786e+00,
                        -2.4419680e-01,
                        -1.0099097e+00,
                        4.8830798e-01,
                        -2.7744773e-01
                    ]
                ],
                [
                    [
                        7.6791526e-05,
                        3.7243867e-06,
                        5.4921486e-05,
                        1.7610956e-04,
                        -7.1586939e-05,
                        9.5206546e-04,
                        -1.0778714e-04,
                        -3.1659831e-05,
                        -9.5621770e-05,
                        -1.7220211e-05,
                        3.3265795e-04,
                        -2.8521221e-04,
                        4.3520480e-01,
                        -6.1880684e-01,
                        3.8080421e-01,
                        -6.1209798e-02,
                        1.7002721e-01,
                        6.3930231e-01
                    ],
                    [
                        -1.3890232e-04,
                        -6.7367587e-06,
                        -9.9343277e-05,
                        -3.1855115e-04,
                        1.2948815e-04,
                        -1.7221184e-03,
                        -2.3488385e-04,
                        -6.8991365e-05,
                        -2.0837371e-04,
                        -3.7470367e-05,
                        7.2384812e-04,
                        -6.2060833e-04,
                        -7.8718948e-01,
                        1.1192850e+00,
                        -6.8879080e-01,
                        1.1067867e-01,
                        -3.0744076e-01,
                        -1.1559772e+00
                    ],
                    [
                        -4.5171482e-06,
                        -2.1908157e-07,
                        -3.2306755e-06,
                        -1.0359386e-05,
                        4.2109964e-06,
                        -5.6003846e-05,
                        -8.1124912e-05,
                        -2.3828454e-05,
                        -7.1968760e-05,
                        -1.2949433e-05,
                        2.5015563e-04,
                        -2.1447684e-04,
                        -2.5596367e-02,
                        3.6394835e-02,
                        -2.2396822e-02,
                        3.5926308e-03,
                        -9.9795293e-03,
                        -3.7523031e-02
                    ],
                    [
                        -1.8113766e-04,
                        -8.7851713e-06,
                        -1.2955008e-04,
                        -4.1541137e-04,
                        1.6886096e-04,
                        -2.2457542e-03,
                        4.3465996e-05,
                        1.2767069e-05,
                        3.8560218e-05,
                        6.9711382e-06,
                        -1.3466760e-04,
                        1.1546051e-04,
                        -1.0265619e+00,
                        1.4596428e+00,
                        -8.9824170e-01,
                        1.4436395e-01,
                        -4.0101099e-01,
                        -1.5078014e+00
                    ],
                    [
                        6.5498651e-05,
                        3.1766829e-06,
                        4.6844794e-05,
                        1.5021110e-04,
                        -6.1059451e-05,
                        8.1205578e-04,
                        -6.2924810e-05,
                        -1.8482619e-05,
                        -5.5822817e-05,
                        -1.0056658e-05,
                        1.9427331e-04,
                        -1.6656483e-04,
                        3.7120280e-01,
                        -5.2780396e-01,
                        3.2480246e-01,
                        -5.2205719e-02,
                        1.4501588e-01,
                        5.4525971e-01
                    ],
                    [
                        -1.2535087e-04,
                        -6.0795137e-06,
                        -8.9651250e-05,
                        -2.8747297e-04,
                        1.1685515e-04,
                        -1.5541068e-03,
                        -2.0725353e-04,
                        -6.0875642e-05,
                        -1.8386189e-04,
                        -3.3062079e-05,
                        6.3868932e-04,
                        -5.4759544e-04,
                        -7.1039069e-01,
                        1.0100868e+00,
                        -6.2159187e-01,
                        9.9881187e-02,
                        -2.7744773e-01,
                        -1.0432035e+00
                    ]
                ],
                [
                    [
                        9.7620694e-05,
                        4.6934238e-06,
                        6.9335867e-05,
                        7.5068994e-05,
                        -3.0517753e-05,
                        4.0582538e-04,
                        -1.3749959e-04,
                        -4.0387127e-05,
                        -1.2198072e-04,
                        -7.3401188e-06,
                        1.4179554e-04,
                        -1.2157177e-04,
                        3.8760430e-01,
                        -3.6040398e-01,
                        1.2648140e+00,
                        2.3123702e-01,
                        5.4408707e-02,
                        4.8287728e-01
                    ],
                    [
                        -1.7657860e-04,
                        -8.4895755e-06,
                        -1.2541635e-04,
                        -1.3578657e-04,
                        5.5201232e-05,
                        -7.3406653e-04,
                        -2.9836787e-04,
                        -8.7638247e-05,
                        -2.6469264e-04,
                        -1.5972179e-05,
                        3.0854862e-04,
                        -2.6454151e-04,
                        -7.0109063e-01,
                        6.5189123e-01,
                        -2.2877693e+00,
                        -4.1811946e-01,
                        -9.8381035e-02,
                        -8.7313175e-01
                    ],
                    [
                        -5.7423936e-06,
                        -2.7608374e-07,
                        -4.0785803e-06,
                        -4.4158232e-06,
                        1.7951619e-06,
                        -2.3872080e-05,
                        -1.0323055e-04,
                        -3.0321444e-05,
                        -9.1579459e-05,
                        -5.5197856e-06,
                        1.0663056e-04,
                        -9.1422233e-05,
                        -2.2796763e-02,
                        2.1196989e-02,
                        -7.4389443e-02,
                        -1.3572161e-02,
                        -3.1934495e-03,
                        -2.8341863e-02
                    ],
                    [
                        -2.3026997e-04,
                        -1.1070959e-05,
                        -1.6355107e-04,
                        -1.7707450e-04,
                        7.1985996e-05,
                        -9.5727044e-04,
                        5.6067442e-05,
                        1.6468437e-05,
                        4.9739399e-05,
                        2.9712407e-06,
                        -5.7398076e-05,
                        4.9211609e-05,
                        -9.1428167e-01,
                        8.5012156e-01,
                        -2.9834454e+00,
                        -5.4537499e-01,
                        -1.2832351e-01,
                        -1.1388712e+00
                    ],
                    [
                        8.3264706e-05,
                        4.0032141e-06,
                        5.9139416e-05,
                        6.4029431e-05,
                        -2.6029848e-05,
                        3.4614519e-04,
                        -8.0355865e-05,
                        -2.3602563e-05,
                        -7.1286515e-05,
                        -4.2866241e-06,
                        8.2808481e-05,
                        -7.0997827e-05,
                        3.3060250e-01,
                        -3.0740231e-01,
                        1.0788081e+00,
                        1.9722161e-01,
                        4.6405081e-02,
                        4.1184512e-01
                    ],
                    [
                        -1.5935142e-04,
                        -7.6613242e-06,
                        -1.1318061e-04,
                        -1.2253910e-04,
                        4.9815746e-05,
                        -6.6245027e-04,
                        -2.6325817e-04,
                        -7.7325640e-05,
                        -2.3354561e-04,
                        -1.4093099e-05,
                        2.7224881e-04,
                        -2.3341896e-04,
                        -6.3269174e-01,
                        5.8829230e-01,
                        -2.0645730e+00,
                        -3.7732893e-01,
                        -8.8783272e-02,
                        -7.8795153e-01
                    ]
                ]
            ]);
        jw_qkb_expected
            .slice_mut(s![1, 1, .., .., ..])
            .assign(&array![
                [
                    [
                        5.1065028e-01,
                        -1.3438474e+00,
                        -6.0824305e-01,
                        9.6085159e-07,
                        1.1323421e-07,
                        6.3339371e-06,
                        -1.9561674e-02,
                        1.0931193e-02,
                        -9.3795331e-03,
                        -1.7048041e-08,
                        -4.5246683e-08,
                        -9.9556885e-08,
                        -1.2316389e-01,
                        -4.2745110e-01,
                        5.8683968e-01,
                        -7.4880004e-01,
                        -6.9840002e-01,
                        1.5120000e-01
                    ],
                    [
                        3.9717245e-01,
                        -1.0452145e+00,
                        -4.7307792e-01,
                        7.4732901e-07,
                        8.8071047e-08,
                        4.9263954e-06,
                        1.4410448e+00,
                        -8.0526531e-01,
                        6.9096118e-01,
                        1.0112918e-06,
                        2.6840405e-06,
                        5.9057188e-06,
                        -3.8927067e-02,
                        -1.3509978e-01,
                        1.8547601e-01,
                        -5.8239871e-01,
                        -5.4319876e-01,
                        1.1759973e-01
                    ],
                    [
                        5.4611212e-01,
                        -1.4371700e+00,
                        -6.5048212e-01,
                        1.0275774e-06,
                        1.2109768e-07,
                        6.7737938e-06,
                        2.4778147e-01,
                        -1.3846190e-01,
                        1.1880782e-01,
                        1.7081310e-07,
                        4.5335008e-07,
                        9.9751026e-07,
                        -1.2122411e-01,
                        -4.2071891e-01,
                        5.7759720e-01,
                        -8.0079973e-01,
                        -7.4689978e-01,
                        1.6169995e-01
                    ],
                    [
                        -5.3901976e-01,
                        1.4185054e+00,
                        6.4203429e-01,
                        -1.0142322e-06,
                        -1.1952498e-07,
                        -6.6858224e-06,
                        -1.4084419e+00,
                        7.8704667e-01,
                        -6.7532855e-01,
                        -9.8744147e-07,
                        -2.6207399e-06,
                        -5.7664379e-06,
                        7.4200206e-02,
                        2.5751832e-01,
                        -3.5354212e-01,
                        7.9039872e-01,
                        7.3719877e-01,
                        -1.5959974e-01
                    ],
                    [
                        3.9008009e-01,
                        -1.0265501e+00,
                        -4.6463010e-01,
                        7.3398388e-07,
                        8.6498353e-08,
                        4.8384240e-06,
                        5.4772741e-01,
                        -3.0607370e-01,
                        2.6262778e-01,
                        3.8284398e-07,
                        1.0160951e-06,
                        2.2357235e-06,
                        -7.2111197e-02,
                        -2.5026822e-01,
                        3.4358862e-01,
                        -5.7199949e-01,
                        -5.3349954e-01,
                        1.1549990e-01
                    ],
                    [
                        1.9149387e-01,
                        -5.0394279e-01,
                        -2.2809115e-01,
                        3.6031938e-07,
                        4.2462826e-08,
                        2.3752266e-06,
                        1.0954548e+00,
                        -6.1214739e-01,
                        5.2525556e-01,
                        7.6947532e-07,
                        2.0422422e-06,
                        4.4935641e-06,
                        -3.1223865e-03,
                        -1.0836507e-02,
                        1.4877245e-02,
                        -2.8079903e-01,
                        -2.6189908e-01,
                        5.6699801e-02
                    ]
                ],
                [
                    [
                        -8.7116987e-02,
                        2.2926044e-01,
                        1.0376631e-01,
                        -2.4145170e-06,
                        -3.2897862e-07,
                        -1.5820675e-05,
                        3.3372107e-03,
                        -1.8648562e-03,
                        1.6001465e-03,
                        2.9152988e-08,
                        7.7374175e-08,
                        1.7024648e-07,
                        -7.6071811e-01,
                        1.4489870e-02,
                        -3.7673658e-01,
                        6.1919999e-01,
                        7.3440003e-01,
                        -1.5624002e+00
                    ],
                    [
                        -6.7757659e-02,
                        1.7831367e-01,
                        8.0707133e-02,
                        -1.8779576e-06,
                        -2.5587227e-07,
                        -1.2304969e-05,
                        -2.4584235e-01,
                        1.3737833e-01,
                        -1.1787803e-01,
                        -2.5242277e-06,
                        -6.6994799e-06,
                        -1.4740926e-05,
                        -2.4043186e-01,
                        4.5796544e-03,
                        -1.1907101e-01,
                        4.8159891e-01,
                        5.7119870e-01,
                        -1.2151973e+00
                    ],
                    [
                        -9.3166776e-02,
                        2.4518131e-01,
                        1.1097230e-01,
                        -2.5821917e-06,
                        -3.5182433e-07,
                        -1.6919332e-05,
                        -4.2271551e-02,
                        2.3621621e-02,
                        -2.0268630e-02,
                        -4.3876398e-07,
                        -1.1645108e-06,
                        -2.5622840e-06,
                        -7.4873710e-01,
                        1.4261660e-02,
                        -3.7080312e-01,
                        6.6219980e-01,
                        7.8539973e-01,
                        -1.6708996e+00
                    ],
                    [
                        9.1956817e-02,
                        -2.4199714e-01,
                        -1.0953110e-01,
                        2.5486568e-06,
                        3.4725520e-07,
                        1.6699601e-05,
                        2.4028030e-01,
                        -1.3427022e-01,
                        1.1521111e-01,
                        2.4686128e-06,
                        6.5518739e-06,
                        1.4416148e-05,
                        4.5829535e-01,
                        -8.7294364e-03,
                        2.2696532e-01,
                        -6.5359890e-01,
                        -7.7519870e-01,
                        1.6491975e+00
                    ],
                    [
                        -6.6547699e-02,
                        1.7512950e-01,
                        7.9265930e-02,
                        -1.8444226e-06,
                        -2.5130311e-07,
                        -1.2085237e-05,
                        -9.3442351e-02,
                        5.2216202e-02,
                        -4.4804327e-02,
                        -9.6180395e-07,
                        -2.5526961e-06,
                        -5.6167205e-06,
                        -4.4539264e-01,
                        8.4836697e-03,
                        -2.2057539e-01,
                        4.7299957e-01,
                        5.6099951e-01,
                        -1.1934991e+00
                    ],
                    [
                        -3.2668870e-02,
                        8.5972667e-02,
                        3.8912367e-02,
                        -9.0544387e-07,
                        -1.2336699e-07,
                        -5.9327531e-06,
                        -1.8688469e-01,
                        1.0443240e-01,
                        -8.9608639e-02,
                        -1.9177760e-06,
                        -5.0899134e-06,
                        -1.1199383e-05,
                        -1.9285316e-02,
                        3.6733947e-04,
                        -9.5508257e-03,
                        2.3219919e-01,
                        2.7539903e-01,
                        -5.8589798e-01
                    ]
                ],
                [
                    [
                        -1.8288359e-01,
                        4.8128355e-01,
                        2.1783531e-01,
                        -1.5904837e-06,
                        -2.8318252e-07,
                        -1.0277977e-05,
                        7.0057651e-03,
                        -3.9148675e-03,
                        3.3591716e-03,
                        -1.2742103e-09,
                        -3.3818641e-09,
                        -7.4412223e-09,
                        -7.2449348e-03,
                        6.5928912e-01,
                        5.1439035e-01,
                        -5.7600003e-01,
                        3.8880005e-01,
                        -1.4544001e+00
                    ],
                    [
                        -1.4224279e-01,
                        3.7433162e-01,
                        1.6942745e-01,
                        -1.2370429e-06,
                        -2.2025307e-07,
                        -7.9939819e-06,
                        -5.1609373e-01,
                        2.8839660e-01,
                        -2.4745987e-01,
                        -1.6372575e-06,
                        -4.3453979e-06,
                        -9.5612186e-06,
                        -2.2898272e-03,
                        2.0837431e-01,
                        1.6257772e-01,
                        -4.4799900e-01,
                        3.0239934e-01,
                        -1.1311975e+00
                    ],
                    [
                        -1.9558384e-01,
                        5.1470596e-01,
                        2.3296276e-01,
                        -1.7009338e-06,
                        -3.0284795e-07,
                        -1.0991725e-05,
                        -8.8740118e-02,
                        4.9588561e-02,
                        -4.2549666e-02,
                        -3.0327826e-07,
                        -8.0492208e-07,
                        -1.7710775e-06,
                        -7.1308301e-03,
                        6.4890552e-01,
                        5.0628889e-01,
                        -6.1599982e-01,
                        4.1579989e-01,
                        -1.5553994e+00
                    ],
                    [
                        1.9304378e-01,
                        -5.0802147e-01,
                        -2.2993727e-01,
                        1.6788438e-06,
                        2.9891487e-07,
                        1.0848975e-05,
                        5.0441742e-01,
                        -2.8187177e-01,
                        2.4186124e-01,
                        1.6070841e-06,
                        4.2653155e-06,
                        9.3850131e-06,
                        4.3647182e-03,
                        -3.9718935e-01,
                        -3.0989495e-01,
                        6.0799903e-01,
                        -4.1039935e-01,
                        1.5351975e+00
                    ],
                    [
                        -1.3970275e-01,
                        3.6764714e-01,
                        1.6640197e-01,
                        -1.2149528e-06,
                        -2.1631999e-07,
                        -7.8512321e-06,
                        -1.9616233e-01,
                        1.0961681e-01,
                        -9.4057150e-02,
                        -6.3319499e-07,
                        -1.6805445e-06,
                        -3.6977174e-06,
                        -4.2418349e-03,
                        3.8600701e-01,
                        3.0117026e-01,
                        -4.3999961e-01,
                        2.9699975e-01,
                        -1.1109990e+00
                    ],
                    [
                        -6.8581350e-02,
                        1.8048133e-01,
                        8.1688240e-02,
                        -5.9643139e-07,
                        -1.0619345e-07,
                        -3.8542412e-06,
                        -3.9232463e-01,
                        2.1923360e-01,
                        -1.8811429e-01,
                        -1.2395833e-06,
                        -3.2899425e-06,
                        -7.2388907e-06,
                        -1.8366973e-04,
                        1.6713955e-02,
                        1.3040550e-02,
                        -2.1599925e-01,
                        1.4579950e-01,
                        -5.4539812e-01
                    ]
                ],
                [
                    [
                        5.7177508e-01,
                        -1.5047055e+00,
                        -6.8104964e-01,
                        -1.3506114e-07,
                        -1.5211199e-08,
                        -8.9184493e-07,
                        -2.1903189e-02,
                        1.2239655e-02,
                        -1.0502286e-02,
                        2.6136475e-09,
                        6.9368058e-09,
                        1.5263122e-08,
                        5.7959478e-02,
                        -1.0722504e+00,
                        -1.5938856e-01,
                        -6.5520006e-01,
                        3.1680000e-01,
                        -1.8000001e-01
                    ],
                    [
                        4.4471392e-01,
                        -1.1703265e+00,
                        -5.2970529e-01,
                        -1.0504755e-07,
                        -1.1830933e-08,
                        -6.9365711e-07,
                        1.6135375e+00,
                        -9.0165550e-01,
                        7.7366918e-01,
                        -1.4242177e-07,
                        -3.7799751e-07,
                        -8.3171136e-07,
                        1.8318618e-02,
                        -3.3889449e-01,
                        -5.0376203e-02,
                        -5.0959885e-01,
                        2.4639945e-01,
                        -1.3999969e-01
                    ],
                    [
                        6.1148161e-01,
                        -1.6091989e+00,
                        -7.2834474e-01,
                        -1.4444038e-07,
                        -1.6267531e-08,
                        -9.5377857e-07,
                        2.7744085e-01,
                        -1.5503578e-01,
                        1.3302909e-01,
                        -2.3858879e-08,
                        -6.3323156e-08,
                        -1.3933051e-07,
                        5.7046641e-02,
                        -1.0553628e+00,
                        -1.5687825e-01,
                        -7.0069981e-01,
                        3.3879989e-01,
                        -1.9249994e-01
                    ],
                    [
                        -6.0354030e-01,
                        1.5883002e+00,
                        7.1888572e-01,
                        1.4256453e-07,
                        1.6056266e-08,
                        9.4139182e-07,
                        -1.5770322e+00,
                        8.8125604e-01,
                        -7.5616539e-01,
                        1.3900070e-07,
                        3.6891777e-07,
                        8.1173312e-07,
                        -3.4917746e-02,
                        6.4597827e-01,
                        9.6023791e-02,
                        6.9159889e-01,
                        -3.3439943e-01,
                        1.8999968e-01
                    ],
                    [
                        4.3677261e-01,
                        -1.1494279e+00,
                        -5.2024627e-01,
                        -1.0317171e-07,
                        -1.1619666e-08,
                        -6.8127042e-07,
                        6.1329031e-01,
                        -3.4271067e-01,
                        2.9406431e-01,
                        -5.3817921e-08,
                        -1.4283658e-07,
                        -3.1428462e-07,
                        3.3934679e-02,
                        -6.2779158e-01,
                        -9.3320362e-02,
                        -5.0049961e-01,
                        2.4199979e-01,
                        -1.3749988e-01
                    ],
                    [
                        2.1441564e-01,
                        -5.6426460e-01,
                        -2.5539362e-01,
                        -5.0647930e-08,
                        -5.7041998e-09,
                        -3.3444184e-07,
                        1.2265805e+00,
                        -6.8542135e-01,
                        5.8812863e-01,
                        -1.0841190e-07,
                        -2.8773292e-07,
                        -6.3310137e-07,
                        1.4693579e-03,
                        -2.7183143e-02,
                        -4.0407372e-03,
                        -2.4569915e-01,
                        1.1879958e-01,
                        -6.7499764e-02
                    ]
                ],
                [
                    [
                        -5.1588410e-01,
                        1.3576207e+00,
                        6.1447710e-01,
                        -6.4919828e-07,
                        -1.6292145e-07,
                        -4.0931473e-06,
                        1.9762151e-02,
                        -1.1043222e-02,
                        9.4756885e-03,
                        -1.5100307e-08,
                        -4.0077300e-08,
                        -8.8182460e-08,
                        -4.6367583e-01,
                        6.5928912e-01,
                        -4.0571636e-01,
                        6.4800009e-02,
                        -1.8000001e-01,
                        -6.7680001e-01
                    ],
                    [
                        -4.0124315e-01,
                        1.0559272e+00,
                        4.7792658e-01,
                        -5.0493196e-07,
                        -1.2671669e-07,
                        -3.1835586e-06,
                        -1.4558142e+00,
                        8.1351864e-01,
                        -6.9804305e-01,
                        -6.5013808e-07,
                        -1.7255126e-06,
                        -3.7966611e-06,
                        -1.4654894e-01,
                        2.0837431e-01,
                        -1.2823033e-01,
                        5.0399888e-02,
                        -1.3999969e-01,
                        -5.2639884e-01
                    ],
                    [
                        -5.5170935e-01,
                        1.4518998e+00,
                        6.5714908e-01,
                        -6.9428143e-07,
                        -1.7423544e-07,
                        -4.3773930e-06,
                        -2.5032103e-01,
                        1.3988103e-01,
                        -1.2002549e-01,
                        -1.3394204e-07,
                        -3.5549166e-07,
                        -7.8219159e-07,
                        -4.5637313e-01,
                        6.4890552e-01,
                        -3.9932647e-01,
                        6.9299981e-02,
                        -1.9249994e-01,
                        -7.2379977e-01
                    ],
                    [
                        5.4454428e-01,
                        -1.4330440e+00,
                        -6.4861465e-01,
                        6.8526481e-07,
                        1.7197264e-07,
                        4.3205437e-06,
                        1.4228773e+00,
                        -7.9511327e-01,
                        6.8225020e-01,
                        6.4242226e-07,
                        1.7050344e-06,
                        3.7516024e-06,
                        2.7934197e-01,
                        -3.9718935e-01,
                        2.4442419e-01,
                        -6.8399891e-02,
                        1.8999968e-01,
                        7.1439886e-01
                    ],
                    [
                        -3.9407811e-01,
                        1.0370713e+00,
                        4.6939221e-01,
                        -4.9591534e-07,
                        -1.2445389e-07,
                        -3.1267095e-06,
                        -5.5334121e-01,
                        3.0921072e-01,
                        -2.6531953e-01,
                        -2.5819776e-07,
                        -6.8527521e-07,
                        -1.5078173e-06,
                        -2.7147743e-01,
                        3.8600701e-01,
                        -2.3754275e-01,
                        4.9499959e-02,
                        -1.3749988e-01,
                        -5.1699954e-01
                    ],
                    [
                        -1.9345653e-01,
                        5.0910777e-01,
                        2.3042890e-01,
                        -2.4344936e-07,
                        -6.1095548e-08,
                        -1.5349301e-06,
                        -1.1066824e+00,
                        6.1842144e-01,
                        -5.3063905e-01,
                        -4.8910266e-07,
                        -1.2981133e-06,
                        -2.8562504e-06,
                        -1.1754863e-02,
                        1.6713955e-02,
                        -1.0285502e-02,
                        2.4299916e-02,
                        -6.7499764e-02,
                        -2.5379911e-01
                    ]
                ],
                [
                    [
                        -7.4386096e-01,
                        1.9575735e+00,
                        8.8602364e-01,
                        -3.9800486e-07,
                        -6.9518272e-08,
                        -2.5748773e-06,
                        2.8495377e-02,
                        -1.5923411e-02,
                        1.3663143e-02,
                        9.5665413e-11,
                        2.5391411e-10,
                        5.5866611e-10,
                        -4.1296127e-01,
                        3.8398153e-01,
                        -1.3475578e+00,
                        -2.4480000e-01,
                        -5.7600003e-02,
                        -5.1120001e-01
                    ],
                    [
                        -5.7855850e-01,
                        1.5225571e+00,
                        6.8912947e-01,
                        -3.0955934e-07,
                        -5.4069766e-08,
                        -2.0026821e-06,
                        -2.0991604e+00,
                        1.1730247e+00,
                        -1.0065186e+00,
                        -4.1022565e-07,
                        -1.0887682e-06,
                        -2.3956263e-06,
                        -1.3052016e-01,
                        1.2136083e-01,
                        -4.2590785e-01,
                        -1.9039957e-01,
                        -4.4799902e-02,
                        -3.9759910e-01
                    ],
                    [
                        -7.9551792e-01,
                        2.0935161e+00,
                        9.4755298e-01,
                        -4.2564406e-07,
                        -7.4345927e-08,
                        -2.7536880e-06,
                        -3.6094159e-01,
                        2.0169654e-01,
                        -1.7306654e-01,
                        -7.5604170e-08,
                        -2.0065886e-07,
                        -4.4151150e-07,
                        -4.0645730e-01,
                        3.7793395e-01,
                        -1.3263344e+00,
                        -2.6179990e-01,
                        -6.1599981e-02,
                        -5.4669982e-01
                    ],
                    [
                        7.8518653e-01,
                        -2.0663276e+00,
                        -9.3524712e-01,
                        4.2011624e-07,
                        7.3380392e-08,
                        2.7179258e-06,
                        2.0516682e+00,
                        -1.1464856e+00,
                        9.8374671e-01,
                        4.0254423e-07,
                        1.0683810e-06,
                        2.3507687e-06,
                        2.4878892e-01,
                        -2.3133004e-01,
                        8.1183755e-01,
                        2.5839958e-01,
                        6.0799900e-02,
                        5.3959912e-01
                    ],
                    [
                        -5.6822711e-01,
                        1.4953686e+00,
                        6.7682362e-01,
                        -3.0403149e-07,
                        -5.3104234e-08,
                        -1.9669201e-06,
                        -7.9787093e-01,
                        4.4585550e-01,
                        -3.8256815e-01,
                        -1.5845890e-07,
                        -4.2056121e-07,
                        -9.2536470e-07,
                        -2.4178459e-01,
                        2.2481723e-01,
                        -7.8898126e-01,
                        -1.8699984e-01,
                        -4.3999963e-02,
                        -3.9049965e-01
                    ],
                    [
                        -2.7894786e-01,
                        7.3409009e-01,
                        3.3225885e-01,
                        -1.4925182e-07,
                        -2.6069351e-08,
                        -9.6557892e-07,
                        -1.5957419e+00,
                        8.9171106e-01,
                        -7.6513630e-01,
                        -3.1067455e-07,
                        -8.2455239e-07,
                        -1.8142703e-06,
                        -1.0469174e-02,
                        9.7344890e-03,
                        -3.4162562e-02,
                        -9.1799676e-02,
                        -2.1599924e-02,
                        -1.9169933e-01
                    ]
                ]
            ]);
        jw_qkb_expected
    }

    fn jw_out_expected() -> Array5<f32> {
        let mut jw_out_expected = Array::zeros((2, 2, 6, 6, 6));
        jw_out_expected
            .slice_mut(s![.., .., 0, .., 0])
            .assign(&array![
                [
                    [2.1413000, 4.1307001, -8.2289000, 1.1444999, 1.5463002, 2.9916000],
                    [1.0285066, -3.1111960, -4.5009022, 2.9744205, -0.4465663, 0.5820699]
                ],
                [
                    [1.1015999, 5.0532999, -0.8537000, -1.0613999, -2.1877000, 1.1192000],
                    [0.2065750, -0.2663467, 2.7993484, -0.1043006, -0.5722010, -1.2466987]
                ]
            ]);
        jw_out_expected
            .slice_mut(s![.., .., 1, .., 1])
            .assign(&array![
                [
                    [2.1413000, 4.1307001, -8.2289000, 1.1444999, 1.5463002, 2.9916000],
                    [1.0285066, -3.1111960, -4.5009022, 2.9744205, -0.4465663, 0.5820699]
                ],
                [
                    [1.1015999, 5.0532999, -0.8537000, -1.0613999, -2.1877000, 1.1192000],
                    [0.2065750, -0.2663467, 2.7993484, -0.1043006, -0.5722010, -1.2466987]
                ]
            ]);
        jw_out_expected
            .slice_mut(s![.., .., 2, .., 2])
            .assign(&array![
                [
                    [2.1413000, 4.1307001, -8.2289000, 1.1444999, 1.5463002, 2.9916000],
                    [1.0285066, -3.1111960, -4.5009022, 2.9744205, -0.4465663, 0.5820699]
                ],
                [
                    [1.1015999, 5.0532999, -0.8537000, -1.0613999, -2.1877000, 1.1192000],
                    [0.2065750, -0.2663467, 2.7993484, -0.1043006, -0.5722010, -1.2466987]
                ]
            ]);
        jw_out_expected
            .slice_mut(s![.., .., 3, .., 3])
            .assign(&array![
                [
                    [2.1413000, 4.1307001, -8.2289000, 1.1444999, 1.5463002, 2.9916000],
                    [1.0285066, -3.1111960, -4.5009022, 2.9744205, -0.4465663, 0.5820699]
                ],
                [
                    [1.1015999, 5.0532999, -0.8537000, -1.0613999, -2.1877000, 1.1192000],
                    [0.2065750, -0.2663467, 2.7993484, -0.1043006, -0.5722010, -1.2466987]
                ]
            ]);
        jw_out_expected
            .slice_mut(s![.., .., 4, .., 4])
            .assign(&array![
                [
                    [2.1413000, 4.1307001, -8.2289000, 1.1444999, 1.5463002, 2.9916000],
                    [1.0285066, -3.1111960, -4.5009022, 2.9744205, -0.4465663, 0.5820699]
                ],
                [
                    [1.1015999, 5.0532999, -0.8537000, -1.0613999, -2.1877000, 1.1192000],
                    [0.2065750, -0.2663467, 2.7993484, -0.1043006, -0.5722010, -1.2466987]
                ]
            ]);
        jw_out_expected
            .slice_mut(s![.., .., 5, .., 5])
            .assign(&array![
                [
                    [2.1413000, 4.1307001, -8.2289000, 1.1444999, 1.5463002, 2.9916000],
                    [1.0285066, -3.1111960, -4.5009022, 2.9744205, -0.4465663, 0.5820699]
                ],
                [
                    [1.1015999, 5.0532999, -0.8537000, -1.0613999, -2.1877000, 1.1192000],
                    [0.2065750, -0.2663467, 2.7993484, -0.1043006, -0.5722010, -1.2466987]
                ]
            ]);
        jw_out_expected
    }

    fn jb_out_expected() -> Array4<f32> {
        array![
            [
                [
                    [1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1.]
                ],
                [
                    [1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1.]
                ]
            ],
            [
                [
                    [1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1.]
                ],
                [
                    [1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1.]
                ]
            ]
        ]
    }
}
