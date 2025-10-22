use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

use ad_derive::*;

use super::dropout::*;
use super::op::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardList {
    pub w_fc: Array3<f32>,
    pub b_fc: Array2<f32>,
    pub w_proj: Array3<f32>,
    pub b_proj: Array2<f32>,
    size: usize,
}

impl std::ops::Add<&FeedForwardList> for &FeedForwardList {
    type Output = FeedForwardList;
    fn add(self, rhs: &FeedForwardList) -> Self::Output {
        Self::Output {
            w_fc: &self.w_fc + &rhs.w_fc,
            b_fc: &self.b_fc + &rhs.b_fc,
            w_proj: &self.w_proj + &rhs.w_proj,
            b_proj: &self.b_proj + &rhs.b_proj,
            ..*self
        }
    }
}

impl std::ops::Div<&FeedForwardList> for &FeedForwardList {
    type Output = FeedForwardList;
    fn div(self, rhs: &FeedForwardList) -> Self::Output {
        Self::Output {
            w_fc: &self.w_fc / &rhs.w_fc,
            b_fc: &self.b_fc / &rhs.b_fc,
            w_proj: &self.w_proj / &rhs.w_proj,
            b_proj: &self.b_proj / &rhs.b_proj,
            ..*self
        }
    }
}

impl std::ops::Add<f32> for &FeedForwardList {
    type Output = FeedForwardList;
    fn add(self, rhs: f32) -> Self::Output {
        Self::Output {
            w_fc: &self.w_fc + rhs,
            b_fc: &self.b_fc + rhs,
            w_proj: &self.w_proj + rhs,
            b_proj: &self.b_proj + rhs,
            ..*self
        }
    }
}

impl std::ops::Mul<f32> for &FeedForwardList {
    type Output = FeedForwardList;
    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            w_fc: &self.w_fc * rhs,
            b_fc: &self.b_fc * rhs,
            w_proj: &self.w_proj * rhs,
            b_proj: &self.b_proj * rhs,
            ..*self
        }
    }
}

impl FeedForwardList {
    pub fn new(size: usize, emb: usize, hidden: usize, rng: &mut StdRng) -> Self {
        let a = (emb as f32).sqrt().powi(-1);
        let w_fc = Array::random_using([size, emb, hidden], Uniform::new(-a, a), rng);
        let b_fc = Array::random_using([size, hidden], Uniform::new(-a, a), rng);
        let w_proj = Array::random_using([size, hidden, emb], Uniform::new(-a, a), rng);
        let b_proj = Array::random_using([size, emb], Uniform::new(-a, a), rng);
        Self {
            w_fc,
            b_fc,
            w_proj,
            b_proj,
            size,
        }
    }

    pub fn zeros(size: usize, emb: usize, hidden: usize) -> Self {
        let w_fc = Array::zeros([size, emb, hidden]);
        let b_fc = Array::zeros([size, hidden]);
        let w_proj = Array::zeros([size, hidden, emb]);
        let b_proj = Array::zeros([size, emb]);
        Self {
            w_fc,
            b_fc,
            w_proj,
            b_proj,
            size,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn pow2(&self) -> Self {
        Self {
            w_fc: self.w_fc.pow2(),
            b_fc: self.b_fc.pow2(),
            w_proj: self.w_proj.pow2(),
            b_proj: self.b_proj.pow2(),
            ..*self
        }
    }

    pub fn sqrt(&self) -> Self {
        Self {
            w_fc: self.w_fc.sqrt(),
            b_fc: self.b_fc.sqrt(),
            w_proj: self.w_proj.sqrt(),
            b_proj: self.b_proj.sqrt(),
            ..*self
        }
    }

    pub fn get(
        self,
        idx: usize,
    ) -> (
        ((Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>), Self),
        impl FnOnce(((Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>), Self)) -> Self,
    ) {
        let w_fc = self.w_fc.index_axis(Axis(0), idx).to_owned();
        let b_fc = self.b_fc.index_axis(Axis(0), idx).to_owned();
        let w_proj = self.w_proj.index_axis(Axis(0), idx).to_owned();
        let b_proj = self.b_proj.index_axis(Axis(0), idx).to_owned();
        (
            ((w_fc, b_fc, w_proj, b_proj), self),
            move |((w_fc, b_fc, w_proj, b_proj), mut params)| {
                Zip::from(params.w_fc.index_axis_mut(Axis(0), idx))
                    .and(&w_fc)
                    .par_for_each(|w_fc_mut, &w_fc| *w_fc_mut += w_fc);
                Zip::from(params.b_fc.index_axis_mut(Axis(0), idx))
                    .and(&b_fc)
                    .par_for_each(|b_fc_mut, &b_fc| *b_fc_mut += b_fc);
                Zip::from(params.w_proj.index_axis_mut(Axis(0), idx))
                    .and(&w_proj)
                    .par_for_each(|w_proj_mut, &w_proj| *w_proj_mut += w_proj);
                Zip::from(params.b_proj.index_axis_mut(Axis(0), idx))
                    .and(&b_proj)
                    .par_for_each(|b_proj_mut, &b_proj| *b_proj_mut += b_proj);
                params
            },
        )
    }

    pub fn detach(mut self) -> ((), impl FnOnce(()) -> Self) {
        self.w_fc.par_mapv_inplace(|_| 0.0);
        self.b_fc.par_mapv_inplace(|_| 0.0);
        self.w_proj.par_mapv_inplace(|_| 0.0);
        self.b_proj.par_mapv_inplace(|_| 0.0);
        ((), move |()| self)
    }

    pub fn call(
        (x, params): (Array3<f32>, FeedForwardList),
        idx: usize,
        dropout: &mut Option<(f64, &mut StdRng)>,
    ) -> (
        (Array3<f32>, FeedForwardList),
        impl FnOnce((Array3<f32>, FeedForwardList)) -> (Array3<f32>, FeedForwardList),
    ) {
        let (bat, seq, emb) = x.dim();
        let (_, hidden) = params.b_fc.dim();
        let mask_dropout = if let Some((prob, rng)) = dropout {
            generate_dropout_mask([bat, seq, emb], *prob, *rng)
        } else {
            Array::ones([bat, seq, emb])
        };
        let f = ad!(|(x, p)| {
            let (((w_fc, b_fc, w_proj, b_proj), p), __) = FeedForwardList::get(p, idx);
            let (b_fc, __) = reshape(b_fc, [1, hidden]);
            let (b_proj, __) = reshape(b_proj, [1, emb]);

            let (x, __) = reshape(x, [bat * seq, emb]);
            let (x, __) = mm((x, w_fc));
            let (x, __) = add((x, b_fc));

            let (x, __) = gelu(x);

            let (x, __) = mm((x, w_proj));
            let (x, __) = add((x, b_proj));

            let (x, __) = reshape(x, [bat, seq, emb]);

            let (m, __) = attach(mask_dropout)(());
            let (x, __) = mul((x, m));

            (x, p)
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
                [[1.93, 1.49, 0.90], [0.68, -1.23, -0.04]],
                [[-0.75, 1.65, -0.39], [-0.72, -0.56, -0.77]]
            ],
            dtype=torch.float32
        )
        w_fc = torch.tensor(
            [
                [0.46, 0.27, 0.53, 0.81],
                [0.76, 2.55, 0.57,  1.35],
                [0.30, 0.93, -1.97, -1.41],
            ],
            dtype=torch.float32
        )
        b_fc = torch.tensor(
            [1.11, -1.69, -0.99, 0.96],
            dtype=torch.float32
        )
        w_proj = torch.tensor(
            [
                [0.32, 0.12, 2.85],
                [0.19, -1.33, 0.39],
                [-0.30, -0.71, 0.07],
                [-0.56, 0.39, 1.36]
            ],
            dtype=torch.float32
        )
        b_proj = torch.tensor(
            [-1.00, -0.02, 0.24],
            dtype=torch.float32
        )

        ff = lambda x, w_fc, b_fc, w_proj, b_proj: nn.functional.linear(
            nn.functional.gelu(
                nn.functional.linear(x, w_fc.T, b_fc),
                approximate='tanh'
            ),
            w_proj.T,
            b_proj
        )
        y = ff(x, w_fc, b_fc, w_proj, b_proj)
        jx, jw_fc, jb_fc, jw_proj, jb_proj = torch.autograd.functional.jacobian(
            ff,
            (x, w_fc, b_fc, w_proj, b_proj)
        )
        */
        let x = array![
            [[1.93, 1.49, 0.90], [0.68, -1.23, -0.04]],
            [[-0.75, 1.65, -0.39], [-0.72, -0.56, -0.77]]
        ];
        let w_fc = array![[
            [0.46, 0.27, 0.53, 0.81],
            [0.76, 2.55, 0.57, 1.35],
            [0.30, 0.93, -1.97, -1.41],
        ]];
        let b_fc = array![[1.11, -1.69, -0.99, 0.96]];
        let w_proj = array![[
            [0.32, 0.12, 2.85],
            [0.19, -1.33, 0.39],
            [-0.30, -0.71, 0.07],
            [-0.56, 0.39, 1.36]
        ]];
        let b_proj = array![[-1.00, -0.02, 0.24]];

        let mut p = FeedForwardList::zeros(1, 3, 4);
        p.w_fc.assign(&w_fc);
        p.b_fc.assign(&b_fc);
        p.w_proj.assign(&w_proj);
        p.b_proj.assign(&b_proj);

        let ((y_actual, _), _) = FeedForwardList::call((x.clone(), p.clone()), 0, &mut None);
        assert!((y_actual - y_expected()).abs().iter().all(|x| *x < 1e-5));

        let mut jx_actual = Array::zeros((2, 2, 3, 2, 2, 3));
        let mut jw_fc_actual = Array::zeros((2, 2, 3, 3, 4));
        let mut jb_fc_actual = Array::zeros((2, 2, 3, 4));
        let mut jw_proj_actual = Array::zeros((2, 2, 3, 4, 3));
        let mut jb_proj_actual = Array::zeros((2, 2, 3, 3));
        for b in 0..2 {
            for s in 0..2 {
                for e in 0..3 {
                    let mut y_adj = Array::zeros((2, 2, 3));
                    y_adj[(b, s, e)] = 1.0;
                    let p_adj = FeedForwardList::zeros(1, 3, 4);
                    let (_, pb) = FeedForwardList::call((x.clone(), p.clone()), 0, &mut None);
                    let (jx_elem, jp_elem) = pb((y_adj, p_adj));
                    jx_actual
                        .slice_mut(s![b, s, e, .., .., ..])
                        .assign(&jx_elem);
                    jw_fc_actual
                        .slice_mut(s![b, s, e, .., ..])
                        .assign(&jp_elem.w_fc.slice(s![0, .., ..]));
                    jb_fc_actual
                        .slice_mut(s![b, s, e, ..])
                        .assign(&jp_elem.b_fc.slice(s![0, ..]));
                    jw_proj_actual
                        .slice_mut(s![b, s, e, .., ..])
                        .assign(&jp_elem.w_proj.slice(s![0, .., ..]));
                    jb_proj_actual
                        .slice_mut(s![b, s, e, ..])
                        .assign(&jp_elem.b_proj.slice(s![0, ..]));
                }
            }
        }
        assert!((jx_actual - jx_expected()).abs().iter().all(|x| *x < 1e-5));
        assert!((jw_fc_actual - jw_fc_expected())
            .abs()
            .iter()
            .all(|x| *x < 1e-5));
        assert!((jb_fc_actual - jb_fc_expected())
            .abs()
            .iter()
            .all(|x| *x < 1e-5));
        assert!((jw_proj_actual - jw_proj_expected())
            .abs()
            .iter()
            .all(|x| *x < 1e-5));
        assert!((jb_proj_actual - jb_proj_expected())
            .abs()
            .iter()
            .all(|x| *x < 1e-5));
    }

    fn y_expected() -> Array3<f32> {
        array![
            [
                [-1.0316528, -2.8319306, 15.7079134],
                [-0.8321781, 0.0959280, 1.0984938]
            ],
            [
                [-1.8590002, -1.2519014, 10.5153379],
                [-1.2567530, 0.2507397, 1.1560206]
            ]
        ]
    }

    fn jx_expected() -> Array6<f32> {
        let mut jx_expected = Array::zeros((2, 2, 3, 2, 2, 3));
        jx_expected
            .slice_mut(s![0, 0, .., 0, 0, ..])
            .assign(&array![
                [-0.2485720, -0.0212442, 1.0364835],
                [0.0325011, -2.7588091, -1.8298509],
                [2.5265656, 5.0142436, -0.6989601]
            ]);
        jx_expected
            .slice_mut(s![0, 1, .., 0, 1, ..])
            .assign(&array![
                [-0.0481563, -0.0936120, 0.3455366],
                [0.2277105, 0.3515818, -0.3749775],
                [1.5819896, 2.6231346, -0.0707451]
            ]);
        jx_expected
            .slice_mut(s![1, 0, .., 1, 0, ..])
            .assign(&array![
                [-0.3587221, -0.0945248, 1.5359069],
                [-0.2942598, -3.3727779, -0.8187611],
                [2.6906753, 5.3409419, -0.7027088]
            ]);
        jx_expected
            .slice_mut(s![1, 1, .., 1, 1, ..])
            .assign(&array![
                [-0.4140514, -0.6573886, 1.0449193],
                [0.2059914, 0.4243420, -0.0094853],
                [1.8750191, 3.1054926, -1.4177679]
            ]);
        jx_expected
    }

    fn jw_fc_expected() -> Array5<f32> {
        array![
            [
                [
                    [
                        [6.1971068e-01, 3.6769509e-01, 3.0179674e-02, -1.0865231e+00],
                        [4.7842950e-01, 2.8386822e-01, 2.3299335e-02, -8.3881837e-01],
                        [2.8898424e-01, 1.7146403e-01, 1.4073423e-02, -5.0666881e-01]
                    ],
                    [
                        [2.3239151e-01, -2.5738657e+00, 7.1425222e-02, 7.5668567e-01],
                        [1.7941105e-01, -1.9870777e+00, 5.5141751e-02, 5.8417708e-01],
                        [1.0836910e-01, -1.2002482e+00, 3.3307098e-02, 3.5285860e-01]
                    ],
                    [
                        [5.5192981e+00, 7.5474256e-01, -7.0419232e-03, 2.6386991e+00],
                        [4.2610126e+00, 5.8267689e-01, -5.4365108e-03, 2.0371304e+00],
                        [2.5737660e+00, 3.5195249e-01, -3.2837985e-03, 1.2304814e+00]
                    ]
                ],
                [
                    [
                        [1.8548425e-01, -1.4221937e-06, 2.5019757e-02, -1.6213469e-01],
                        [-3.3550826e-01, 2.5724974e-06, -4.5256328e-02, 2.9327303e-01],
                        [-1.0910837e-02, 8.3658449e-08, -1.4717504e-03, 9.5373346e-03]
                    ],
                    [
                        [6.9556594e-02, 9.9553572e-06, 5.9213422e-02, 1.1291523e-01],
                        [
                            -1.2581562e-01,
                            -1.8007484e-05,
                            -1.0710663e-01,
                            -2.0424372e-01
                        ],
                        [
                            -4.0915641e-03,
                            -5.8560920e-07,
                            -3.4831425e-03,
                            -6.6420720e-03
                        ]
                    ],
                    [
                        [1.6519692e+00, -2.9192397e-06, -5.8379434e-03, 3.9375567e-01],
                        [-2.9881206e+00, 5.2803898e-06, 1.0559809e-02, -7.1223450e-01],
                        [-9.7174652e-02, 1.7171998e-07, 3.4340844e-04, -2.3162097e-02]
                    ]
                ]
            ],
            [
                [
                    [
                        [-2.6317722e-01, -1.5550031e-01, 1.6823274e-01, 4.2336398e-01],
                        [5.7898992e-01, 3.4210068e-01, -3.7011200e-01, -9.3140078e-01],
                        [-1.3685216e-01, -8.0860160e-02, 8.7481014e-02, 2.2014926e-01]
                    ],
                    [
                        [-9.8691456e-02, 1.0885022e+00, 3.9815077e-01, -2.9484275e-01],
                        [2.1712120e-01, -2.3947048e+00, -8.7593168e-01, 6.4865404e-01],
                        [-5.1319554e-02, 5.6602114e-01, 2.0703839e-01, -1.5331823e-01]
                    ],
                    [
                        [
                            -2.3439221e+00,
                            -3.1918484e-01,
                            -3.9254300e-02,
                            -1.0281696e+00
                        ],
                        [5.1566286e+00, 7.0220661e-01, 8.6359464e-02, 2.2619731e+00],
                        [
                            -1.2188395e+00,
                            -1.6597611e-01,
                            -2.0412236e-02,
                            -5.3464818e-01
                        ]
                    ]
                ],
                [
                    [
                        [-1.3755237e-01, 4.0386523e-05, 7.8331061e-02, 3.9489475e-01],
                        [-1.0698517e-01, 3.1411739e-05, 6.0924158e-02, 3.0714035e-01],
                        [-1.4710461e-01, 4.3191139e-05, 8.3770715e-02, 4.2231795e-01]
                    ],
                    [
                        [
                            -5.1582139e-02,
                            -2.8270567e-04,
                            1.8538350e-01,
                            -2.7501595e-01
                        ],
                        [
                            -4.0119439e-02,
                            -2.1988219e-04,
                            1.4418717e-01,
                            -2.1390130e-01
                        ],
                        [
                            -5.5164225e-02,
                            -3.0233801e-04,
                            1.9825734e-01,
                            -2.9411426e-01
                        ]
                    ],
                    [
                        [
                            -1.2250757e+00,
                            8.2898652e-05,
                            -1.8277248e-02,
                            -9.5903009e-01
                        ],
                        [
                            -9.5283663e-01,
                            6.4476728e-05,
                            -1.4215636e-02,
                            -7.4591225e-01
                        ],
                        [
                            -1.3101504e+00,
                            8.8655492e-05,
                            -1.9546499e-02,
                            -1.0256293e+00
                        ]
                    ]
                ]
            ]
        ]
    }

    fn jb_fc_expected() -> Array4<f32> {
        array![
            [
                [
                    [3.2109362e-01, 1.9051559e-01, 1.5637137e-02, -5.6296533e-01],
                    [1.2041011e-01, -1.3336092e+00, 3.7007887e-02, 3.9206514e-01],
                    [2.8597400e+00, 3.9105833e-01, -3.6486650e-03, 1.3672016e+00]
                ],
                [
                    [2.7277094e-01, -2.0914613e-06, 3.6793761e-02, -2.3843336e-01],
                    [1.0228911e-01, 1.4640231e-05, 8.7078564e-02, 1.6605181e-01],
                    [2.4293664e+00, -4.2929996e-06, -8.5852109e-03, 5.7905245e-01]
                ]
            ],
            [
                [
                    [3.5090297e-01, 2.0733374e-01, -2.2431031e-01, -5.6448531e-01],
                    [1.3158861e-01, -1.4513363e+00, -5.3086770e-01, 3.9312366e-01],
                    [3.1252296e+00, 4.2557979e-01, 5.2339070e-02, 1.3708929e+00]
                ],
                [
                    [
                        1.9104494e-01,
                        -5.6092391e-05,
                        -1.0879314e-01,
                        -5.4846489e-01
                    ],
                    [7.1641855e-02, 3.9264676e-04, -2.5747707e-01, 3.8196659e-01],
                    [1.7014940e+00, -1.1513701e-04, 2.5385065e-02, 1.3319862e+00]
                ]
            ]
        ]
    }

    fn jw_proj_expected() -> Array5<f32> {
        array![
            [
                [
                    [
                        [3.3992949e+00, 0., 0.],
                        [3.4669003e+00, 0., 0.],
                        [-1.6626461e-01, 0., 0.],
                        [3.2643170e+00, 0., 0.]
                    ],
                    [
                        [0., 3.3992949e+00, 0.],
                        [0., 3.4669003e+00, 0.],
                        [0., -1.6626461e-01, 0.],
                        [0., 3.2643170e+00, 0.]
                    ],
                    [
                        [0., 0., 3.3992949e+00],
                        [0., 0., 3.4669003e+00],
                        [0., 0., -1.6626461e-01],
                        [0., 0., 3.2643170e+00]
                    ]
                ],
                [
                    [
                        [3.2507592e-01, 0., 0.],
                        [-1.8132121e-06, 0., 0.],
                        [-1.3205293e-01, 0., 0.],
                        [-4.3182377e-02, 0., 0.]
                    ],
                    [
                        [0., 3.2507592e-01, 0.],
                        [0., -1.8132121e-06, 0.],
                        [0., -1.3205293e-01, 0.],
                        [0., -4.3182377e-02, 0.]
                    ],
                    [
                        [0., 0., 3.2507592e-01],
                        [0., 0., -1.8132121e-06],
                        [0., 0., -1.3205293e-01],
                        [0., 0., -4.3182377e-02]
                    ]
                ]
            ],
            [
                [
                    [
                        [1.8476446e+00, 0., 0.],
                        [1.9026678e+00, 0., 0.],
                        [2.0113319e-01, 0., 0.],
                        [3.1275237e+00, 0., 0.]
                    ],
                    [
                        [0., 1.8476446e+00, 0.],
                        [0., 1.9026678e+00, 0.],
                        [0., 2.0113319e-01, 0.],
                        [0., 3.1275237e+00, 0.]
                    ],
                    [
                        [0., 0., 1.8476446e+00],
                        [0., 0., 1.9026678e+00],
                        [0., 0., 2.0113319e-01],
                        [0., 0., 3.1275237e+00]
                    ]
                ],
                [
                    [
                        [6.7042448e-02, 0., 0.],
                        [-6.1229919e-05, 0., 0.],
                        [-7.4946329e-02, 0., 0.],
                        [5.3692633e-01, 0., 0.]
                    ],
                    [
                        [0., 6.7042448e-02, 0.],
                        [0., -6.1229919e-05, 0.],
                        [0., -7.4946329e-02, 0.],
                        [0., 5.3692633e-01, 0.]
                    ],
                    [
                        [0., 0., 6.7042448e-02],
                        [0., 0., -6.1229919e-05],
                        [0., 0., -7.4946329e-02],
                        [0., 0., 5.3692633e-01]
                    ]
                ]
            ]
        ]
    }

    fn jb_proj_expected() -> Array4<f32> {
        array![
            [
                [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],],
                [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],]
            ],
            [
                [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],],
                [[1., 0., 0.,], [0., 1., 0.,], [0., 0., 1.,],]
            ]
        ]
    }
}
