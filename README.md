# proc-macro-gpt

Training GPT-2 using *source-to-source* automatic differentiation implemented with Rust procedural macros.

This hobby project explores how far compile-time metaprogramming in Rust can go when building an automatic differentiation (AD) system.

(The design is heavily inspired by [Zygote.jl](https://github.com/FluxML/Zygote.jl).)

## Macro Design

Because of Rust’s metaprogramming constraints, functions that can be differentiated with this macro must follow certain syntactic rules.

In an ideal compile-time AD system, the process works as follows:

When a function is marked for differentiation, each inner call `f(x)` is replaced with a corresponding **rrule call** `rrule_of_f(x)`.

Each `rrule_of_f` returns a pair:

1. the original output $f(x)$
2. a pullback closure that computes the vector–Jacobian product or the Jacobian-transpose-vector-product $\bar{y}\mapsto J_{f(x)}^\top \bar{y}$

The pullbacks are then composed in reverse order to build the overall backward pass.
If the final output is a scalar, feeding 1 into its pullback gives the full gradient of the function.

However, Rust’s procedural macros operate only on token streams — they cannot observe runtime function calls.
This means we can’t trace the execution graph or safely rewrite `f` into `rrule_of_f` automatically.
To address this, the implementation follows an explicit, convention-based design.

In the implemented `ad!` macro:

* Each differentiable function call must explicitly use its corresponding `rrule_of_*` form.
* The pullback must be bound to a variable following a specific naming pattern: `__`.

These naming conventions allow the macro to recognize rrule calls purely from token patterns.

### Example

The following code:

```rust
ad!(|(x, y)| {
    let (x, __) = foo(x, 1); // Differentiate w.r.t. the first argument only
    let (mut y, __) = bar((x, y)); //  Use a tuple for multiple differentiation arguments
    let mut __; // Uninitialized `__` indicates loop prelude
    for i in 0..10 {
        (y, __) = baz(y, i); //  Update mutable arg and `__`; first elements of LHS/RHS must match
    }
    let (z, __) = (if some_cond(&y) {baz} else {bra})(y); // `if`/`match` not directly supported, but can be written this way
    z
})
```

is expanded into:

```rust
|(x, y)| {
    let (x, __pb1) = foo(x, 1);
    let (mut y, __pb2) = bar((x, y));
    let mut __pb3 = Vec::new();
    for i in 0..10 {
        __pb3.push(
            {
                let __;
                (y, __) = baz(y, i);
                __
            }
        );
    }
    let (z, __pb4) = (if some_cond(&y) {baz} else {bra})(y);
    (
        z,
        move |mut z| {
            let mut y = __pb4(z);
            while let Some(__pb3) = __pb3.pop() {
                y = __pb3(y);
            }
            let mut (x, y) = __pb2(y);
            let mut x = __pb1(x);
            (x, y)
        }
    )
}
```

## GPT-2 Implementation

To verify the effectiveness of this macro, I implemented GPT-2 from scratch using [ndarray](https://github.com/rust-ndarray/ndarray), a widely used n-dimensional array library in Rust, and created code that actually performs training through automatic differentiation.

In this metaprogramming-based approach, there is no need to define a special tensor type or maintain a computation graph — regular `ndarray` arrays can be used directly.

I defined reverse-mode differentiation rules (rrules) for ndarray operations such as arithmetic (with broadcasting), (batched) matrix multiplication, log, exp, gelu, pow2, sqrt, sum, split, permute, reshape, and advanced indexing.

By combining these operations, I implemented the basic components of a decoder-only Transformer and enabled gradient computation for the cross-entropy loss.

### Usage

You can train a small model using the [TinyStories](https://arxiv.org/abs/2305.07759) dataset:

```bash
git clone https://github.com/ho-oto/proc-macro-gpt.git
cd proc-macro-gpt/gpt2
python data.py # or `uv run data.py`
cargo run --release # Running in debug mode will be much slower
```

Example result (after 2000 training steps, input: `One day,`):

```plain
One day, a little girl named Lily went to the park with her mom. She saw a big tree with a big tree. She wanted to climb the tree, but it was too high.

Lily's mom said, "Lily, I
```

## Reference

[Blog post (Japanese)](https://zenn.dev/ho_oto/articles/32a460c64e963b)

## License

MIT
