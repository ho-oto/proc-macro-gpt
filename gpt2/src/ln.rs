use ndarray::prelude::*;
use ndarray::Zip;
use serde::{Deserialize, Serialize};

use ad_derive::*;

use super::op::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormList {
    pub w: Array2<f32>,
    pub b: Array2<f32>,
    pub eps: f32,
    size: usize,
}

impl std::ops::Add<&LayerNormList> for &LayerNormList {
    type Output = LayerNormList;
    fn add(self, rhs: &LayerNormList) -> Self::Output {
        Self::Output {
            w: &self.w + &rhs.w,
            b: &self.b + &rhs.b,
            ..*self
        }
    }
}

impl std::ops::Div<&LayerNormList> for &LayerNormList {
    type Output = LayerNormList;
    fn div(self, rhs: &LayerNormList) -> Self::Output {
        Self::Output {
            w: &self.w / &rhs.w,
            b: &self.b / &rhs.b,
            ..*self
        }
    }
}

impl std::ops::Add<f32> for &LayerNormList {
    type Output = LayerNormList;
    fn add(self, rhs: f32) -> Self::Output {
        Self::Output {
            w: &self.w + rhs,
            b: &self.b + rhs,
            ..*self
        }
    }
}

impl std::ops::Mul<f32> for &LayerNormList {
    type Output = LayerNormList;
    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            w: &self.w * rhs,
            b: &self.b * rhs,
            ..*self
        }
    }
}

impl LayerNormList {
    pub fn new(size: usize, emb: usize, eps: f32) -> Self {
        let w = Array::ones([size, emb]);
        let b = Array::zeros([size, emb]);
        Self { w, b, eps, size }
    }

    pub fn zeros(size: usize, emb: usize, eps: f32) -> Self {
        let w = Array::zeros([size, emb]);
        let b = Array::zeros([size, emb]);
        Self { w, b, eps, size }
    }

    pub fn len(&self) -> usize {
        self.w.dim().0
    }

    pub fn pow2(&self) -> Self {
        Self {
            w: self.w.pow2(),
            b: self.b.pow2(),
            ..*self
        }
    }

    pub fn sqrt(&self) -> Self {
        Self {
            w: self.w.sqrt(),
            b: self.b.sqrt(),
            ..*self
        }
    }

    pub fn get(
        self,
        idx: usize,
    ) -> (
        ((Array1<f32>, Array1<f32>), Self),
        impl FnOnce(((Array1<f32>, Array1<f32>), Self)) -> Self,
    ) {
        let w = self.w.index_axis(Axis(0), idx).to_owned();
        let b = self.b.index_axis(Axis(0), idx).to_owned();
        (((w, b), self), move |((w, b), mut params)| {
            Zip::from(params.w.index_axis_mut(Axis(0), idx))
                .and(&w)
                .par_for_each(|w_mut, &w| *w_mut += w);
            Zip::from(params.b.index_axis_mut(Axis(0), idx))
                .and(&b)
                .par_for_each(|b_mut, &b| *b_mut += b);
            params
        })
    }

    pub fn detach(mut self) -> ((), impl FnOnce(()) -> Self) {
        self.w.par_mapv_inplace(|_| 0.0);
        self.b.par_mapv_inplace(|_| 0.0);
        ((), move |()| self)
    }

    pub fn call(
        (x, params): (Array3<f32>, LayerNormList),
        idx: usize,
    ) -> (
        (Array3<f32>, LayerNormList),
        impl FnOnce((Array3<f32>, LayerNormList)) -> (Array3<f32>, LayerNormList),
    ) {
        let (_, _, emb) = x.dim();
        let emb_axis = Axis(2);
        let f = ad!(|(x, p)| {
            let (((w, b), p), __) = LayerNormList::get(p, idx);

            let ((x, y), __) = fork(x);
            let ((x, z), __) = fork(x);

            // mean
            let (e, __) = sum_along(y, emb_axis);
            let (e, __) = div_num(e, emb as f32); // = E[x]

            // standard-deviation
            let ((e, s), __) = fork(e);
            let (s, __) = sub((z, s)); // = x - E[x]
            let (s, __) = square(s); // = (x - E[x])²
            let (s, __) = sum_along(s, emb_axis); // = ∑ (x - E[x])²
            let (s, __) = div_num(s, emb as f32); // = V[x] = ∑ (x - E[x])² / |x|
            let (s, __) = add_num(s, p.eps); // = V[x] + ε
            let (s, __) = sqrt(s); // = √(V[x] + ε)

            // normalize
            let (x, __) = sub((x, e));
            let (x, __) = div((x, s)); // (x - E[x]) / √(V[x] + ε)

            // element-wise affine transformation
            let (w, __) = reshape(w, [1, 1, emb]);
            let (b, __) = reshape(b, [1, 1, emb]);
            let (x, __) = mul((x, w));
            let (x, __) = add((x, b));

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
                [[1.93, 1.49, 0.90, -2.11], [0.68, -1.23, -0.04, -1.604]],
                [[-0.75, 1.65, -0.39, -1.40], [-0.72, -0.56, -0.77, 0.76]]
            ],
            dtype=torch.float32
        )
        w = torch.tensor(
            [0.46, 0.27, 0.53, 0.81],
            dtype=torch.float32
        )
        b = torch.tensor(
            [1.11, -1.69, -0.99, 0.96],
            dtype=torch.float32
        )
        y_expected = nn.functional.layer_norm(i, (4,), weight=w, bias=b, eps=1e-5)
        jx_expected, jw_expected, jb_expected = torch.autograd.functional.jacobian(
            lambda i, w, b: nn.functional.layer_norm(i, (4,), weight=w, bias=b, eps=1e-5),
            (i, w, b)
        )
        */
        let x = array![
            [[1.93, 1.49, 0.90, -2.11], [0.68, -1.23, -0.04, -1.604]],
            [[-0.75, 1.65, -0.39, -1.40], [-0.72, -0.56, -0.77, 0.76]]
        ];
        let w = array![[0.46, 0.27, 0.53, 0.81]];
        let b = array![[1.11, -1.69, -0.99, 0.96]];

        let mut p = LayerNormList::new(1, 4, 1e-5);
        p.w.assign(&w);
        p.b.assign(&b);

        let ((y_actual, _), _) = LayerNormList::call((x.clone(), p.clone()), 0);
        assert!((y_actual - y_expected()).abs().iter().all(|x| *x < 1e-5));

        let mut jx_actual = Array::zeros((2, 2, 4, 2, 2, 4));
        let mut jw_actual = Array::zeros((2, 2, 4, 4));
        let mut jb_actual = Array::zeros((2, 2, 4, 4));
        for b in 0..2 {
            for s in 0..2 {
                for e in 0..4 {
                    let mut y_adj = Array::zeros((2, 2, 4));
                    y_adj[(b, s, e)] = 1.0;
                    let p_adj = LayerNormList::zeros(1, 4, 1e-5);
                    let (_, pb) = LayerNormList::call((x.clone(), p.clone()), 0);
                    let (jx_elem, jp_elem) = pb((y_adj, p_adj));
                    jx_actual
                        .slice_mut(s![b, s, e, .., .., ..])
                        .assign(&jx_elem);
                    jw_actual
                        .slice_mut(s![b, s, e, ..])
                        .assign(&jp_elem.w.slice(s![0, ..]));
                    jb_actual
                        .slice_mut(s![b, s, e, ..])
                        .assign(&jp_elem.b.slice(s![0, ..]));
                }
            }
        }
        assert!((jx_actual - jx_expected()).abs().iter().all(|x| *x < 1e-5));
        assert!((jw_actual - jw_expected()).abs().iter().all(|x| *x < 1e-5));
        assert!((jb_actual - jb_expected()).abs().iter().all(|x| *x < 1e-5));
    }

    fn y_expected() -> Array3<f32> {
        array![
            [
                [1.5110340, -1.5297985, -0.8734366, -0.4049174],
                [1.7278421, -1.8911752, -0.6953467, 0.0252665]
            ],
            [
                [0.8971637, -1.2465436, -1.0678675, 0.1234128],
                [0.8196627, -1.7918205, -1.3665969, 2.3522615]
            ]
        ]
    }

    fn jx_expected() -> Array6<f32> {
        array![
            [
                [
                    [
                        [
                            [1.63029e-01, -1.10432e-01, -8.67382e-02, 3.41408e-02],
                            [0., 0., 0., 0.]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ],
                    [
                        [
                            [-6.48188e-02, 1.13121e-01, -4.82951e-02, -7.44313e-06],
                            [0., 0., 0., 0.]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ],
                    [
                        [
                            [-9.99375e-02, -9.48016e-02, 2.47519e-01, -5.27804e-02],
                            [0., 0., 0., 0.]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ],
                    [
                        [
                            [6.01175e-02, -2.23368e-05, -8.06644e-02, 2.05692e-02],
                            [0., 0., 0., 0.]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ]
                ],
                [
                    [
                        [
                            [0., 0., 0., 0.],
                            [1.50372e-01, 9.56058e-05, -2.19616e-01, 6.91478e-02]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ],
                    [
                        [
                            [0., 0., 0., 0.],
                            [5.60991e-05, 1.80425e-01, -4.32287e-02, -1.37252e-01]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ],
                    [
                        [
                            [0., 0., 0., 0.],
                            [-2.53036e-01, -8.48563e-02, 3.89817e-01, -5.19248e-02]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ],
                    [
                        [
                            [0., 0., 0., 0.],
                            [1.21760e-01, -4.11758e-01, -7.93568e-02, 3.69355e-01]
                        ],
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]]
                    ]
                ]
            ],
            [
                [
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [2.81016e-01, -2.42157e-02, -1.07727e-01, -1.49073e-01],
                            [0., 0., 0., 0.]
                        ]
                    ],
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [-1.42135e-02, 1.79052e-02, -4.49196e-02, 4.12279e-02],
                            [0., 0., 0., 0.]
                        ]
                    ],
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [-1.24120e-01, -8.81755e-02, 3.46151e-01, -1.33855e-01],
                            [0., 0., 0., 0.]
                        ]
                    ],
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [-2.62499e-01, 1.23683e-01, -2.04571e-01, 3.43387e-01],
                            [0., 0., 0., 0.]
                        ]
                    ]
                ],
                [
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [0., 0., 0., 0.],
                            [4.75062e-01, -2.26065e-01, -2.64496e-01, 1.54989e-02]
                        ]
                    ],
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [0., 0., 0., 0.],
                            [-1.32690e-01, 3.06296e-01, -1.35899e-01, -3.77060e-02]
                        ]
                    ],
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [0., 0., 0., 0.],
                            [-3.04745e-01, -2.66765e-01, 5.24943e-01, 4.65676e-02]
                        ]
                    ],
                    [
                        [[0., 0., 0., 0.], [0., 0., 0., 0.]],
                        [
                            [0., 0., 0., 0.],
                            [2.72915e-02, -1.13118e-01, 7.11694e-02, 1.46571e-02]
                        ]
                    ]
                ]
            ]
        ]
    }

    fn jw_expected() -> Array4<f32> {
        array![
            [
                [
                    [0.8718129, 0., 0., 0.],
                    [0., 0.5933391, 0., 0.],
                    [0., 0., 0.2199310, 0.],
                    [0., 0., 0., -1.6850832]
                ],
                [
                    [1.3431350, 0., 0., 0.],
                    [0., -0.7450928, 0., 0.],
                    [0., 0., 0.5559496, 0.],
                    [0., 0., 0., -1.1539919]
                ]
            ],
            [
                [
                    [-0.4626875, 0., 0., 0.],
                    [0., 1.6424308, 0., 0.],
                    [0., 0., -0.1469197, 0.],
                    [0., 0., 0., -1.0328238]
                ],
                [
                    [-0.6311682, 0., 0., 0.],
                    [0., -0.3771130, 0., 0.],
                    [0., 0., -0.7105603, 0.],
                    [0., 0., 0., 1.7188413]
                ]
            ]
        ]
    }

    fn jb_expected() -> Array4<f32> {
        array![
            [
                [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]
                ],
                [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]
                ]
            ],
            [
                [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]
                ],
                [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]
                ]
            ]
        ]
    }
}
