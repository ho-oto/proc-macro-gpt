use ndarray::{prelude::*, RemoveAxis};

use ad_derive::ad;

use super::op::*;

pub fn softmax<F: NdFloat, D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
    x: Array<F, D>,
    axis: Axis,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    let f = ad!(|x| {
        let ((x, y), __) = fork(x);
        let (y, __) = max_along(y, axis);
        let (x, __) = sub((x, y));
        let (e, __) = exp(x);
        let ((e, s), __) = fork(e);
        let (s, __) = sum_along(s, axis);
        let (r, __) = div((e, s));
        r
    });
    f(x)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test() {
        /*
        x = torch.tensor(
            [[1.93, 1.49, 0.90], [0.68, -1.23, -0.04]],
            dtype=torch.float32
        )
        y0 = torch.softmax(x, 0)
        y1 = torch.softmax(x, 1)
        jx0 = torch.autograd.functional.jacobian(
            lambda x: torch.softmax(x, 0),
            x
        )
        jx1 = torch.autograd.functional.jacobian(
            lambda x: torch.softmax(x, 1),
            x
        )
        */
        let x = array![[1.93, 1.49, 0.90], [0.68, -1.23, -0.04]];
        let y_0_expected = array![
            [0.7772999, 0.9381965, 0.7190996],
            [0.2227001, 0.0618035, 0.2809003]
        ];
        let y_1_expected = array![
            [0.4997393, 0.3218503, 0.1784104],
            [0.6116834, 0.0905783, 0.2977383]
        ];
        let (y_0_actual, _) = softmax(x.clone(), Axis(0));
        let (y_1_actual, _) = softmax(x.clone(), Axis(1));
        assert!((y_0_actual - y_0_expected).abs().iter().all(|x| *x < 1e-5));
        assert!((y_1_actual - y_1_expected).abs().iter().all(|x| *x < 1e-5));
        let mut jx_0_actual = Array::zeros((2, 3, 2, 3));
        let mut jx_1_actual = Array::zeros((2, 3, 2, 3));
        for i in 0..2 {
            for j in 0..3 {
                let mut y_adj = Array::zeros((2, 3));
                y_adj[(i, j)] = 1.0;
                let (_, pb_0) = softmax(x.clone(), Axis(0));
                let (_, pb_1) = softmax(x.clone(), Axis(1));
                let jx_0_elem = pb_0(y_adj.clone());
                let jx_1_elem = pb_1(y_adj);
                jx_0_actual.slice_mut(s![i, j, .., ..]).assign(&jx_0_elem);
                jx_1_actual.slice_mut(s![i, j, .., ..]).assign(&jx_1_elem);
            }
        }
        assert!((jx_0_actual - jx_0_expected())
            .abs()
            .iter()
            .all(|x| *x < 1e-5));
        assert!((jx_1_actual - jx_1_expected())
            .abs()
            .iter()
            .all(|x| *x < 1e-5));
    }

    fn jx_0_expected() -> Array4<f32> {
        array![
            [
                [[0.1731048, 0., 0.], [-0.1731048, 0., 0.]],
                [[0., 0.0579838, 0.], [0., -0.0579838, 0.]],
                [[0., 0., 0.2019953], [0., 0., -0.2019953]]
            ],
            [
                [[-0.1731048, 0., 0.], [0.1731048, 0., 0.]],
                [[0., -0.0579838, 0.], [0., 0.0579838, 0.]],
                [[0., 0., -0.2019953], [0., 0., 0.2019953]]
            ]
        ]
    }

    fn jx_1_expected() -> Array4<f32> {
        array![
            [
                [[0.2499999, -0.1608412, -0.0891587], [0., 0., 0.]],
                [[-0.1608412, 0.2182627, -0.0574214], [0., 0., 0.]],
                [[-0.0891587, -0.0574214, 0.1465801], [0., 0., 0.]]
            ],
            [
                [[0., 0., 0.], [0.2375268, -0.0554053, -0.1821216]],
                [[0., 0., 0.], [-0.0554053, 0.0823739, -0.0269686]],
                [[0., 0., 0.], [-0.1821216, -0.0269686, 0.2090902]]
            ]
        ]
    }
}
