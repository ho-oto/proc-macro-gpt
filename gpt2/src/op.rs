use ndarray::{
    concatenate,
    parallel::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
    prelude::*,
    IntoDimension, RemoveAxis, Slice, Zip,
};
use ndarray_stats::QuantileExt;

pub fn attach<T>(x: T) -> impl FnOnce(()) -> (T, fn(T) -> ()) {
    |()| (x, |_| ())
}

pub fn detach<F: NdFloat, D: Dimension>(
    mut x: Array<F, D>,
) -> ((), impl FnOnce(()) -> Array<F, D>) {
    x.par_mapv_inplace(|_| F::zero());
    ((), move |()| x)
}

pub fn fork<F: NdFloat, D: Dimension>(
    x: Array<F, D>,
) -> (
    (Array<F, D>, Array<F, D>),
    impl FnOnce((Array<F, D>, Array<F, D>)) -> Array<F, D>,
) {
    let y = x.clone();
    ((x, y), move |(mut x_adj, y_adj)| {
        Zip::from(&mut x_adj)
            .and(&y_adj)
            .par_for_each(|x_adj, &y_adj| *x_adj += y_adj);
        x_adj
    })
}

//

fn _sum_along<F: NdFloat, D: Dimension<Smaller = J> + RemoveAxis, J: Dimension<Larger = D>>(
    x: &Array<F, D>,
    axis: &Axis,
) -> Array<F, D> {
    Zip::from(x.lanes(*axis))
        .par_map_collect(|x| x.sum())
        .insert_axis(*axis)
}

fn _broadcast_info<F, D: Dimension>(
    x: &Array<F, D>,
    y: &Array<F, D>,
) -> (D::Pattern, Vec<Axis>, Vec<Axis>) {
    let mut z_dim = D::zeros(x.ndim());
    let mut axes_x = Vec::new();
    let mut axes_y = Vec::new();
    for (i, (&x, &y)) in x.shape().iter().zip(y.shape()).enumerate() {
        z_dim[i] = x.max(y);
        if x != y {
            (if x == 1 { &mut axes_x } else { &mut axes_y }).push(Axis(i));
        }
    }
    (z_dim.into_pattern(), axes_x, axes_y)
}

pub fn add<F: NdFloat, D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
    (mut x, mut y): (Array<F, D>, Array<F, D>),
) -> (
    Array<F, D>,
    impl FnOnce(Array<F, D>) -> (Array<F, D>, Array<F, D>),
) {
    let (z_dim, axes_x, axes_y) = _broadcast_info(&x, &y);
    let z = if z_dim == x.dim() {
        Zip::from(&mut x)
            .and(&y.broadcast(z_dim).unwrap())
            .par_for_each(|x, &y| *x += y);
        x
    } else if z_dim == y.dim() {
        Zip::from(&x.broadcast(z_dim).unwrap())
            .and(&mut y)
            .par_for_each(|&x, y| *y += x);
        y
    } else {
        Zip::from(&x.broadcast(z_dim.clone()).unwrap())
            .and(&y.broadcast(z_dim).unwrap())
            .par_map_collect(|&x, &y| x + y)
    };
    (z, move |z_adj| {
        let mut x_adj = z_adj.clone();
        for axis in axes_x.iter() {
            x_adj = _sum_along(&x_adj, axis)
        }
        let mut y_adj = z_adj;
        for axis in axes_y.iter() {
            y_adj = _sum_along(&y_adj, axis)
        }
        (x_adj, y_adj)
    })
}

pub fn sub<F: NdFloat, D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
    (mut x, mut y): (Array<F, D>, Array<F, D>),
) -> (
    Array<F, D>,
    impl FnOnce(Array<F, D>) -> (Array<F, D>, Array<F, D>),
) {
    let (z_dim, axes_x, axes_y) = _broadcast_info(&x, &y);
    let z = if z_dim == x.dim() {
        Zip::from(&mut x)
            .and(&y.broadcast(z_dim).unwrap())
            .par_for_each(|x, &y| *x -= y);
        x
    } else if z_dim == y.dim() {
        Zip::from(&x.broadcast(z_dim).unwrap())
            .and(&mut y)
            .par_for_each(|&x, y| *y = x - *y);
        y
    } else {
        Zip::from(&x.broadcast(z_dim.clone()).unwrap())
            .and(&y.broadcast(z_dim).unwrap())
            .par_map_collect(|&x, &y| x - y)
    };
    (z, move |z_adj| {
        let mut x_adj = z_adj.clone();
        for axis in axes_x.iter() {
            x_adj = _sum_along(&x_adj, axis)
        }
        let mut y_adj = z_adj;
        for axis in axes_y.iter() {
            y_adj = _sum_along(&y_adj, axis)
        }
        y_adj.par_mapv_inplace(|x| -x);
        (x_adj, y_adj)
    })
}

pub fn mul<F: NdFloat, D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
    (x, y): (Array<F, D>, Array<F, D>),
) -> (
    Array<F, D>,
    impl FnOnce(Array<F, D>) -> (Array<F, D>, Array<F, D>),
) {
    let (z_dim, axes_x, axes_y) = _broadcast_info(&x, &y);
    let mut z = Array::zeros(z_dim.clone());
    Zip::from(x.broadcast(z_dim.clone()).unwrap())
        .and(y.broadcast(z_dim.clone()).unwrap())
        .par_map_assign_into(&mut z, |&x, &y| x * y);
    (z, move |z_adj| {
        let x = x.broadcast(z_dim.clone()).unwrap();
        let y = y.broadcast(z_dim).unwrap();
        let mut x_adj = z_adj.clone();
        Zip::from(&mut x_adj)
            .and(&y)
            .par_for_each(|x_adj, &y| *x_adj *= y);
        for axis in axes_x.iter() {
            x_adj = _sum_along(&x_adj, axis)
        }
        let mut y_adj = z_adj;
        Zip::from(&mut y_adj)
            .and(&x)
            .par_for_each(|y_adj, &x| *y_adj *= x);
        for axis in axes_y.iter() {
            y_adj = _sum_along(&y_adj, axis)
        }
        (x_adj, y_adj)
    })
}

pub fn div<F: NdFloat, D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
    (x, y): (Array<F, D>, Array<F, D>),
) -> (
    Array<F, D>,
    impl FnOnce(Array<F, D>) -> (Array<F, D>, Array<F, D>),
) {
    let (z_dim, axes_x, axes_y) = _broadcast_info(&x, &y);
    let mut z = Array::zeros(z_dim.clone());
    Zip::from(x.broadcast(z_dim.clone()).unwrap())
        .and(y.broadcast(z_dim.clone()).unwrap())
        .par_map_assign_into(&mut z, |&x, &y| x / y);
    (z, move |z_adj| {
        let x = x.broadcast(z_dim.clone()).unwrap();
        let y = y.broadcast(z_dim).unwrap();
        let mut x_adj = z_adj.clone();
        Zip::from(&mut x_adj)
            .and(&y)
            .par_for_each(|x_adj, &y| *x_adj /= y);
        for axis in axes_x.iter() {
            x_adj = _sum_along(&x_adj, axis)
        }
        let mut y_adj = z_adj;
        Zip::from(&mut y_adj)
            .and(&x)
            .and(&y)
            .par_for_each(|y_adj, &x, &y| *y_adj *= -x / y.powi(2));
        for axis in axes_y.iter() {
            y_adj = _sum_along(&y_adj, axis)
        }
        (x_adj, y_adj)
    })
}

pub fn mm<F: NdFloat>(
    (x, y): (Array2<F>, Array2<F>),
) -> (Array2<F>, impl FnOnce(Array2<F>) -> (Array2<F>, Array2<F>)) {
    let z = x.dot(&y);
    (z, move |z_adj| (z_adj.dot(&y.t()), x.t().dot(&z_adj)))
}

pub fn bmm<F: NdFloat>(
    (mut x, mut y): (Array3<F>, Array3<F>),
) -> (Array3<F>, impl FnOnce(Array3<F>) -> (Array3<F>, Array3<F>)) {
    let bmm = |x: &Array3<F>, y: &Array3<F>| {
        let (bx, l, _) = x.dim();
        let (by, _, r) = y.dim();
        assert_eq!(bx, by);
        let mut z = Array::zeros([bx, l, r]);
        Zip::from(z.axis_iter_mut(Axis(0)))
            .and(x.axis_iter(Axis(0)))
            .and(y.axis_iter(Axis(0)))
            .par_for_each(|mut z, x, y| {
                z.assign(&x.dot(&y));
            });
        z
    };
    let z = bmm(&x, &y);
    (z, move |z_adj| {
        x.swap_axes(1, 2);
        y.swap_axes(1, 2);
        (bmm(&z_adj, &y), bmm(&x, &z_adj))
    })
}

//

pub fn add_num<F: NdFloat, D: Dimension>(
    mut x: Array<F, D>,
    c: F,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    x.par_iter_mut().for_each(|x| *x += c);
    let y = x;
    (y, |y_adj| y_adj)
}

pub fn div_num<F: NdFloat, D: Dimension>(
    mut x: Array<F, D>,
    c: F,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    x.par_iter_mut().for_each(|x| *x /= c);
    let y = x;
    (y, move |mut y_adj| {
        y_adj.par_iter_mut().for_each(|y_adj| *y_adj /= c);
        y_adj
    })
}

pub fn sqrt<F: NdFloat, D: Dimension>(
    mut x: Array<F, D>,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    x.par_mapv_inplace(|x| x.sqrt());
    let y = x;
    (y.clone(), move |mut y_adj| {
        Zip::from(&mut y_adj)
            .and(&y)
            .par_for_each(|y_adj, &y| *y_adj /= y + y);
        y_adj
    })
}

pub fn square<F: NdFloat, D: Dimension>(
    x: Array<F, D>,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    let y = Zip::from(&x).par_map_collect(|x| x.powi(2));
    (y, move |mut y_adj| {
        Zip::from(&mut y_adj)
            .and(&x)
            .par_for_each(|y_adj, &x| *y_adj *= x + x);
        y_adj
    })
}

pub fn exp<F: NdFloat, D: Dimension>(
    mut x: Array<F, D>,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    x.par_mapv_inplace(|x| x.exp());
    let y = x;
    (y.clone(), move |mut y_adj| {
        Zip::from(&mut y_adj)
            .and(&y)
            .par_for_each(|y_adj, &y| *y_adj *= y);
        y_adj
    })
}

pub fn log<F: NdFloat, D: Dimension>(
    x: Array<F, D>,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    let y = Zip::from(&x).par_map_collect(|x| x.ln());
    (y, move |mut y_adj| {
        Zip::from(&mut y_adj)
            .and(&x)
            .par_for_each(|y_adj, &x| *y_adj /= x);
        y_adj
    })
}

pub fn gelu<F: NdFloat, D: Dimension>(
    x: Array<F, D>,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    let one = F::one();
    let two = one + one;
    let three = one + two;
    let a = F::from(0.044715).unwrap();
    let b = F::from(0.7978845608028654).unwrap();
    let mut y = Array::zeros(x.dim());
    let mut z = Array::zeros(x.dim());
    Zip::from(&x)
        .and(&mut y)
        .and(&mut z)
        .par_for_each(|&x, y, z| {
            // GELU(x) = 0.5 x (1 + tanh(b (x + ax³)))
            *y = one + (b * (x + a * x.powi(3))).tanh(); // y(x) = (1 + tanh(b (x + ax³)))
            *z = x * (*y) / two; // GELU(x) = 0.5 x y(x)
        });
    (z, move |mut z_adj| {
        Zip::from(&mut z_adj)
            .and(&x)
            .and(&y)
            .par_for_each(|z_adj, &x, &y| {
                // GELU'(x) = 0.5 (y(x) + b y(x) (2 - y(x)) (3ax³ + x))
                *z_adj *= (y + b * y * (two - y) * (three * a * x.powi(3) + x)) / two
            });
        z_adj
    })
}

//

pub fn reshape<F: NdFloat, I: Dimension, J: IntoDimension>(
    x: Array<F, I>,
    new_shape: J,
) -> (
    Array<F, J::Dim>,
    impl FnOnce(Array<F, J::Dim>) -> Array<F, I>,
) {
    let orig_shape = x.dim();
    let y = x.to_shape(new_shape).unwrap().to_owned();
    (y, move |y_adj| {
        y_adj.to_shape(orig_shape).unwrap().to_owned()
    })
}

pub fn swap_axes<F: NdFloat, D: Dimension>(
    mut x: Array<F, D>,
    ax: Axis,
    bx: Axis,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    x.swap_axes(ax.index(), bx.index());
    (x, move |mut x_adj| {
        x_adj.swap_axes(ax.index(), bx.index());
        x_adj
    })
}

pub fn split_at<F: NdFloat, D: Dimension + RemoveAxis>(
    x: Array<F, D>,
    axis: Axis,
    index: usize,
) -> (
    (Array<F, D>, Array<F, D>),
    impl FnOnce((Array<F, D>, Array<F, D>)) -> Array<F, D>,
) {
    let (y, z) = x.view().split_at(axis, index);
    let y = y.to_owned();
    let z = z.to_owned();
    ((y, z), move |(y_adj, z_adj)| {
        let mut x_adj = x;
        Zip::from(x_adj.slice_axis_mut(axis, Slice::from(..index)))
            .and(&y_adj)
            .par_for_each(|x, &y| *x = y);
        Zip::from(x_adj.slice_axis_mut(axis, Slice::from(index..)))
            .and(&z_adj)
            .par_for_each(|x, &z| *x = z);
        x_adj
    })
}

pub fn sum_along<F: NdFloat, D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
    x: Array<F, D>,
    axis: Axis,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    let d = x.len_of(axis);
    let y = _sum_along(&x, &axis);
    (y, move |y_adj| {
        concatenate(axis, &vec![y_adj.view(); d]).unwrap()
    })
}

pub fn sum<F: NdFloat, D: Dimension>(x: Array<F, D>) -> (F, impl FnOnce(F) -> Array<F, D>) {
    let y = x
        .into_par_iter()
        .cloned()
        .reduce(|| F::zero(), |a, b| a + b);
    (y, move |y_adj| {
        let mut x_adj = x;
        x_adj.par_mapv_inplace(|_| y_adj);
        x_adj
    })
}

pub fn max_along<F: NdFloat, D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
    x: Array<F, D>,
    axis: Axis,
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    let i = Zip::from(x.lanes(axis))
        .par_map_collect(|x| x.argmax().unwrap())
        .insert_axis(axis);
    let mut y = Array::zeros(i.dim());
    Zip::from(y.lanes_mut(axis))
        .and(x.lanes(axis))
        .and(i.lanes(axis))
        .par_for_each(|mut y, x, i| y[0] = x[i[0]]);
    (y, move |y_adj| {
        let mut x_adj = Array::zeros(x.dim());
        Zip::from(x_adj.lanes_mut(axis))
            .and(y_adj.lanes(axis))
            .and(i.lanes(axis))
            .par_for_each(|mut x, y, i| x[i[0]] = y[0]);
        x_adj
    })
}

pub fn advanced_indexing_along<
    F: NdFloat,
    D: Dimension<Smaller = S> + RemoveAxis,
    S: Dimension<Larger = D>,
>(
    x: Array<F, D>,
    axis: Axis,
    indices: &[usize],
) -> (Array<F, D>, impl FnOnce(Array<F, D>) -> Array<F, D>) {
    let indices = indices.to_vec();
    let mut d = x.raw_dim();
    d[axis.index()] = indices.len();
    let mut y = Array::zeros(d);
    Zip::from(y.axis_iter_mut(axis))
        .and(&indices)
        .par_for_each(|mut y, &i| y.assign(&x.index_axis(axis, i)));
    (y, move |y_adj| {
        let mut x_adj = x;
        x_adj.par_mapv_inplace(|_| F::zero());
        Zip::indexed(x_adj.axis_iter_mut(axis)).par_for_each(|i, mut x_adj| {
            y_adj
                .axis_iter(axis)
                .enumerate()
                .filter(|(j, _)| indices[*j] == i)
                .for_each(|(_, y_adj)| x_adj.scaled_add(F::one(), &y_adj));
        });
        x_adj
    })
}

pub fn advanced_indexing_zipped<
    F: NdFloat,
    D: Dimension<Smaller = S> + RemoveAxis,
    S: Dimension<Larger = D> + RemoveAxis,
>(
    x: Array<F, D>,
    (axis_a, indices_a): (Axis, &[usize]),
    (axis_b, indices_b): (Axis, &[usize]),
) -> (Array<F, S>, impl FnOnce(Array<F, S>) -> Array<F, D>)
where
    <D as ndarray::Dimension>::Pattern: ndarray::NdIndex<D>,
{
    let mut axis_a = axis_a.index();
    let mut axis_b = axis_b.index();
    assert!(axis_a != axis_b);
    let mut indices_a = indices_a.to_vec();
    let mut indices_b = indices_b.to_vec();
    assert!(indices_a.len() == indices_b.len());
    if axis_a > axis_b {
        (axis_a, axis_b) = (axis_b, axis_a);
        (indices_a, indices_b) = (indices_b, indices_a);
    }
    let mut d = x.raw_dim();
    d[axis_a] = indices_a.len();
    for i in axis_b..(d.ndim() - 1) {
        d[i] = d[i + 1];
    }
    d[x.ndim() - 1] = 1;
    let mut y = Array::zeros(d).remove_axis(Axis(x.ndim() - 1));
    Zip::indexed(y.axis_iter_mut(Axis(axis_a))).par_for_each(|i, mut y| {
        y.assign(
            &x.index_axis(Axis(axis_b), indices_b[i])
                .index_axis(Axis(axis_a), indices_a[i]),
        );
    });
    (y, move |y_adj| {
        let mut x_adj = x;
        x_adj.par_mapv_inplace(|_| F::zero());
        Zip::indexed(x_adj.axis_iter_mut(Axis(axis_a))).par_for_each(|i_a, mut x_adj| {
            Zip::indexed(x_adj.axis_iter_mut(Axis(axis_b - 1))).for_each(|i_b, mut x_adj| {
                y_adj
                    .axis_iter(Axis(axis_a))
                    .enumerate()
                    .filter(|(j, _)| indices_a[*j] == i_a && indices_b[*j] == i_b)
                    .for_each(|(_, y_adj)| x_adj.scaled_add(F::one(), &y_adj));
            });
        });
        x_adj
    })
}

#[allow(dead_code)]
pub fn dbg(_: (), msg: &str) -> ((), impl Fn(()) -> ()) {
    let msg = msg.to_string();
    println!("->{}", msg);
    ((), move |()| println!("<-{}", msg.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ad_derive::ad;
    use ndarray_rand::RandomExt;

    fn rng(seed: u64) -> ndarray_rand::rand::rngs::StdRng {
        ndarray_rand::rand::SeedableRng::seed_from_u64(seed)
    }

    fn uni(low: f64, hight: f64) -> ndarray_rand::rand_distr::Uniform<f64> {
        ndarray_rand::rand_distr::Uniform::new(low, hight)
    }

    fn grad<D: Dimension>(
        f: impl Fn(Array<f64, D>) -> f64,
        a: Array<f64, D>,
        delta: f64,
    ) -> Array<f64, D>
    where
        <D as ndarray::Dimension>::Pattern: ndarray::NdIndex<D>,
    {
        let mut grad = Array::zeros(a.dim());
        let f_a = f(a.clone());
        for (pat, _) in a.indexed_iter() {
            let mut a = a.clone();
            a[pat.clone()] += delta;
            grad[pat] = (f(a) - f_a) / delta
        }
        grad
    }

    fn test_grad<D: Dimension, F>(f: impl Fn(Array<f64, D>) -> (f64, F), a: Array<f64, D>)
    where
        F: FnOnce(f64) -> Array<f64, D>,
        <D as ndarray::Dimension>::Pattern: ndarray::NdIndex<D>,
    {
        let grad_fd = grad(|x| f(x).0, a.clone(), 1e-6);
        let grad_ad = f(a).1(1.0);
        assert!((grad_ad - grad_fd).abs().iter().all(|x| *x < 1e-3));
    }

    fn proj_rand<D: Dimension<Smaller = S> + RemoveAxis, S: Dimension<Larger = D>>(
        a: Array<f64, D>,
        seed: u64,
    ) -> (f64, impl FnOnce(f64) -> Array<f64, D>) {
        let p = Array::random_using(a.dim(), uni(-1.0, 1.0), &mut rng(seed));
        let f = ad!(|x| {
            let (p, __) = attach(p.clone())(());
            let (p, __) = mul((x, p));
            let (p, __) = sum(p);
            p
        });
        f(a)
    }

    #[test]
    fn test_fork() {
        test_grad(
            ad!(|x| {
                let ((x, y), __) = fork(x);
                let (z, __) = mm((x, y));
                let (z, __) = proj_rand(z, 0);
                z
            }),
            Array::random_using((2, 2), uni(-2.0, 2.0), &mut rng(1)),
        );
    }

    fn test_broadcasting_dyadic<F>(
        op: impl Fn((Array3<f64>, Array3<f64>)) -> (Array3<f64>, F),
        seed: u64,
        low: f64,
        hight: f64,
    ) where
        F: FnOnce(Array3<f64>) -> (Array3<f64>, Array3<f64>),
    {
        let mut rng = rng(seed);
        let mut arr = |b, s, t| Array::random_using((b, s, t), uni(low, hight), &mut rng);
        for y in [arr(2, 3, 4), arr(2, 1, 4), arr(1, 1, 1), arr(2, 1, 1)] {
            test_grad(
                ad!(|x| {
                    let (y, __) = attach(y.clone())(());
                    let (z, __) = op((x, y));
                    let (z, __) = proj_rand(z, seed + 42);
                    z
                }),
                arr(2, 3, 4),
            );
            test_grad(
                ad!(|x| {
                    let (y, __) = attach(y.clone())(());
                    let (z, __) = op((y, x));
                    let (z, __) = proj_rand(z, seed + 43);
                    z
                }),
                arr(2, 3, 4),
            );
        }
    }

    #[test]
    fn test_add() {
        let x = array![[1., 2.], [3., 4.]];

        let y = array![[1., 3.], [10., 11.]];
        assert_eq!((&x + &y), add((x.clone(), y)).0);

        let y = array![[1., 3.]];
        assert_eq!((&x + &y), add((x.clone(), y.clone())).0);
        assert_eq!((&y + &x), add((y, x.clone())).0);

        let y = array![[1.], [3.]];
        assert_eq!((&x + &y), add((x.clone(), y.clone())).0);
        assert_eq!((&y + &x), add((y, x)).0);

        for seed in 0..10 {
            test_broadcasting_dyadic(add, seed, -2.0, 2.0);
        }
    }

    #[test]
    fn test_sub() {
        for seed in 0..10 {
            let x = array![[1., 2.], [3., 4.]];

            let y = array![[1., 3.], [10., 11.]];
            assert_eq!((&x - &y), sub((x.clone(), y)).0);

            let y = array![[1., 3.]];
            assert_eq!((&x - &y), sub((x.clone(), y.clone())).0);
            assert_eq!((&y - &x), sub((y, x.clone())).0);

            let y = array![[1.], [3.]];
            assert_eq!((&x - &y), sub((x.clone(), y.clone())).0);
            assert_eq!((&y - &x), sub((y, x)).0);

            test_broadcasting_dyadic(sub, seed, -2.0, 2.0);
        }
    }

    #[test]
    fn test_mul() {
        let x = array![[1., 2.], [3., 4.]];

        let y = array![[1., 3.], [10., 11.]];
        assert_eq!((&x * &y), mul((x.clone(), y)).0);

        let y = array![[1., 3.]];
        assert_eq!((&x * &y), mul((x.clone(), y.clone())).0);
        assert_eq!((&y * &x), mul((y, x.clone())).0);

        let y = array![[1.], [3.]];
        assert_eq!((&x * &y), mul((x.clone(), y.clone())).0);
        assert_eq!((&y * &x), mul((y, x)).0);

        for seed in 0..10 {
            test_broadcasting_dyadic(mul, seed, -2.0, 2.0);
        }
    }

    #[test]
    fn test_div() {
        let x = array![[1., 2.], [3., 4.]];

        let y = array![[1., 3.], [10., 11.]];
        assert_eq!((&x / &y), div((x.clone(), y)).0);

        let y = array![[1., 3.]];
        assert_eq!((&x / &y), div((x.clone(), y.clone())).0);
        assert_eq!((&y / &x), div((y, x.clone())).0);

        let y = array![[1.], [3.]];
        assert_eq!((&x / &y), div((x.clone(), y.clone())).0);
        assert_eq!((&y / &x), div((y, x)).0);

        for seed in 0..10 {
            test_broadcasting_dyadic(div, seed, 2.0, 4.0);
        }
    }

    #[test]
    fn test_mm() {
        let mut rng = rng(0);
        let mut arr = |s, t| Array::random_using((s, t), uni(-10.0, 10.0), &mut rng);
        for (a, b) in [
            (arr(3, 4), arr(4, 3)),
            (arr(4, 3), arr(3, 4)),
            (arr(3, 3), arr(3, 3)),
        ] {
            assert_eq!(a.dot(&b), mm((a.clone(), b.clone())).0);

            test_grad(
                ad!(|x| {
                    let (y, __) = attach(b.clone())(());
                    let (z, __) = mm((x, y));
                    let (z, __) = proj_rand(z, 42);
                    z
                }),
                a.clone(),
            );
            test_grad(
                ad!(|x| {
                    let (y, __) = attach(a.clone())(());
                    let (z, __) = mm((y, x));
                    let (z, __) = proj_rand(z, 43);
                    z
                }),
                b.clone(),
            );
        }
    }

    #[test]
    fn test_bmm() {
        let mut rng = rng(0);
        let mut arr = |b, s, t| Array::random_using((b, s, t), uni(-10.0, 10.0), &mut rng);
        for (a, b) in [
            (arr(2, 3, 4), arr(2, 4, 3)),
            (arr(2, 4, 3), arr(2, 3, 4)),
            (arr(2, 3, 3), arr(2, 3, 3)),
        ] {
            for i in 0..a.len_of(Axis(0)) {
                let s = s![i, .., ..];
                assert_eq!(
                    a.slice(s).dot(&b.slice(s)),
                    bmm((a.clone(), b.clone())).0.slice(s)
                );
            }

            test_grad(
                ad!(|x| {
                    let (y, __) = attach(b.clone())(());
                    let (z, __) = bmm((x, y));
                    let (z, __) = proj_rand(z, 42);
                    z
                }),
                a.clone(),
            );
            test_grad(
                ad!(|y| {
                    let (x, __) = attach(a.clone())(());
                    let (z, __) = bmm((x, y));
                    let (z, __) = proj_rand(z, 43);
                    z
                }),
                b.clone(),
            );
        }
    }

    fn test_unary<F>(op: impl Fn(Array2<f64>) -> (Array2<f64>, F), seed: u64, low: f64, hight: f64)
    where
        F: FnOnce(Array2<f64>) -> Array2<f64>,
    {
        test_grad(
            ad!(|x| {
                let (z, __) = op(x);
                let (z, __) = proj_rand(z, seed + 42);
                z
            }),
            Array::random_using((3, 4), uni(low, hight), &mut rng(seed)),
        );
    }

    #[test]
    fn test_sqrt() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(x.sqrt(), sqrt(x).0);

        for seed in 0..10 {
            test_unary(sqrt, seed, 1.0, 5.0);
        }
    }

    #[test]
    fn test_square() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(x.pow2(), square(x).0);

        for seed in 0..10 {
            test_unary(square, seed, -5.0, 5.0);
        }
    }

    #[test]
    fn test_log() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(x.ln(), log(x).0);

        for seed in 0..10 {
            test_unary(log, seed, 1.0, 5.0);
        }
    }

    #[test]
    fn test_exp() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(x.exp(), exp(x).0);

        for seed in 0..10 {
            test_unary(exp, seed, -1.0, 1.0);
        }
    }

    #[test]
    fn test_gelu() {
        let x = array![-2., -1., 0., 1., 2.];
        let y = array![
            -0.0454022884368896,
            -0.1588079929351807,
            0.0000000000000000,
            0.8411920070648193,
            1.9545977115631104
        ]; // torch: torch.nn.functional.gelu(x, approximate='tanh')
        assert!((y - gelu(x).0).iter().all(|x: &f64| x.abs() < 1e-7));

        for seed in 0..10 {
            test_unary(gelu, seed, -10.0, 10.0);
        }
    }

    #[test]
    fn test_add_num() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!((&x + 1.0), add_num(x, 1.0).0);

        for seed in 0..10 {
            test_unary(|x| add_num(x, 2.0), seed, -10.0, 10.0);
        }
    }

    #[test]
    fn test_div_num() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!((&x / 3.0), div_num(x, 3.0).0);

        for seed in 0..10 {
            test_unary(|x| div_num(x, 3.0), seed, -10.0, 10.0);
        }
    }

    #[test]
    fn test_reshape() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(
            x.clone().to_shape([1, 4, 1]).unwrap(),
            reshape(x, [1, 4, 1]).0
        );

        test_grad(
            ad!(|x| {
                let (z, __) = reshape(x, [1, 3, 20]);
                let (z, __) = proj_rand(z, 42);
                z
            }),
            Array::random_using((3, 4, 5), uni(-10.0, 10.0), &mut rng(0)),
        );
    }

    #[test]
    fn test_swap_axes() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(
            x.clone().permuted_axes([1, 0]),
            swap_axes(x, Axis(0), Axis(1)).0
        );

        test_grad(
            ad!(|x| {
                let (z, __) = swap_axes(x, Axis(0), Axis(2));
                let (z, __) = proj_rand(z, 42);
                z
            }),
            Array::random_using((3, 4, 5), uni(-10.0, 10.0), &mut rng(0)),
        );
    }

    #[test]
    fn test_split_at() {
        let x = array![[1., 2.], [3., 4.]];

        let ((y, z), __) = split_at(x.clone(), Axis(0), 1);
        assert_eq!(x.slice(s![..1, ..]), y);
        assert_eq!(x.slice(s![1.., ..]), z);

        let ((y, z), __) = split_at(x.clone(), Axis(1), 1);
        assert_eq!(x.slice(s![.., ..1]), y);
        assert_eq!(x.slice(s![.., 1..]), z);

        test_grad(
            ad!(|x| {
                let ((a, b), __) = split_at(x, Axis(1), 3);
                let (c, __) = mul((a, b));
                let (t, __) = sum(c);
                t
            }),
            Array::random_using((3, 6), uni(-10.0, 10.0), &mut rng(0)),
        );
    }

    #[test]
    fn test_sum_along() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(array![[4., 6.]], sum_along(x.clone(), Axis(0)).0);
        assert_eq!(array![[3.], [7.]], sum_along(x, Axis(1)).0);

        test_grad(
            ad!(|x| {
                let (z, __) = sum_along(x, Axis(1));
                let (z, __) = proj_rand(z, 42);
                z
            }),
            Array::random_using((3, 4, 5), uni(-10.0, 10.0), &mut rng(0)),
        );
    }

    #[test]
    fn test_sum() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(x.sum(), sum(x).0);

        test_grad(
            ad!(|x| {
                let (t, __) = sum(x);
                t
            }),
            Array::random_using((3, 4, 5), uni(-10.0, 10.0), &mut rng(0)),
        );
    }

    #[test]
    fn test_max_along() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(
            array![[3., 4.]], // torch: x.max(axis=0, keepdims=True).values
            max_along(x.clone(), Axis(0)).0
        );
        assert_eq!(
            array![[2.], [4.]], // torch: x.max(axis=1, keepdims=True).values
            max_along(x.clone(), Axis(1)).0
        );

        test_grad(
            ad!(|x| {
                let (z, __) = max_along(x, Axis(1));
                let (z, __) = proj_rand(z, 42);
                z
            }),
            Array::random_using((3, 10, 5), uni(-10.0, 10.0), &mut rng(0)),
        );
    }

    #[test]
    fn test_advanced_indexing_along() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(
            array![[3., 4.], [1., 2.], [3., 4.], [3., 4.], [1., 2.]], // torch: x[[1,0,1,1,0],:]
            advanced_indexing_along(x.clone(), Axis(0), &[1, 0, 1, 1, 0]).0
        );
        assert_eq!(
            array![[2., 1., 2., 2., 1.], [4., 3., 4., 4., 3.]], // torch: x[:,[1,0,1,1,0]]
            advanced_indexing_along(x, Axis(1), &[1, 0, 1, 1, 0]).0
        );

        test_grad(
            ad!(|x| {
                let (z, __) = advanced_indexing_along(
                    x,
                    Axis(1),
                    &[5, 1, 3, 4, 6, 8, 5, 2, 8, 8, 5, 7, 3, 0, 8, 7, 7, 8, 0, 9],
                );
                let (z, __) = proj_rand(z, 42);
                z
            }),
            Array::random_using((3, 10, 5), uni(-10.0, 10.0), &mut rng(0)),
        );
    }

    #[test]
    fn test_advanced_indexing_zipped() {
        let x = array![[1., 2.], [3., 4.]];
        assert_eq!(
            array![4., 1., 3., 3., 2.], // torch: x[[1,0,1,1,0],[1,0,0,0,1]]
            advanced_indexing_zipped(
                x.clone(),
                (Axis(0), &[1, 0, 1, 1, 0]),
                (Axis(1), &[1, 0, 0, 0, 1])
            )
            .0
        );

        test_grad(
            ad!(|x| {
                let (z, __) = advanced_indexing_zipped(
                    x,
                    (Axis(0), &[3, 1, 3, 2, 2, 0, 0, 0, 1]),
                    (Axis(1), &[0, 0, 0, 1, 1, 2, 0, 1, 2]),
                );
                let (z, __) = proj_rand(z, 42);
                z
            }),
            Array::random_using((5, 4), uni(-10.0, 10.0), &mut rng(0)),
        );
    }
}
