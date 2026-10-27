---
title: MLP vs CNN
type: map
tags: [neural-networks, linear-algebra, deep-learning]
sources: []
created: 2026-10-13
updated: 2026-10-13
---

# MLP vs CNN

> **One-liner:** An MLP connects every input to every neuron with its own weight; a CNN slides one small shared filter over the input.

Interactive version (rendered formulas, sliding-filter demo, quiz): `explainers/2026-10-13_mlp-vs-cnn.html`.

> ⚠️ Not from `raw/mml-book.pdf` — the book does not cover neural-network architectures. This page is built from standard deep-learning knowledge. A source such as Goodfellow, Bengio & Courville, *Deep Learning* (2016), Ch. 6 and Ch. 9, would ground it.

## 🟢 Intuition (Dummy)
An MLP is a committee where each member reads the whole page at once. A CNN is a magnifying glass with one fixed "pattern detector" that you move across the page. The magnifying glass finds the same pattern (an edge, an eye) anywhere, and it needs to learn only a few numbers.

## 🟡 Mechanics (Practitioner)
**MLP layer:** $\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$, with $\mathbf{W}\in\mathbb{R}^{m\times n}$ — every entry learned. See [[matrix]], [[vector]].

**Convolution layer (2-D):**
$$h_{i,j} = \sigma\Big(\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K_{u,v}\, x_{i+u,\,j+v} + b\Big)$$
The kernel $\mathbf{K}\in\mathbb{R}^{k\times k}$ has no $i,j$ index: the same weights are used at every position (**weight sharing**), and each output sees only a $k\times k$ patch (**local receptive field**).

**Parameter count (first layer, 28×28 image):** MLP with 100 neurons = $784\cdot100+100 = 78{,}500$. CNN with 32 filters of 3×3 = $32\cdot(9+1) = 320$, independent of image size.

| | MLP | CNN |
|---|---|---|
| Connection | dense | local patch |
| Weights | one per connection | one kernel, reused |
| Input | flattened vector | keeps grid shape |
| Shift input | unrelated change | output shifts too (equivariance) |
| Best for | tabular / vector data | images, audio, time series |

Both train the same way: a loss, [[backpropagation]], [[gradient-descent]]. A shared weight's gradient is the sum of its gradients over all positions.

## 🔴 Theory (Expert)
- **Convolution is a constrained linear map.** In 1-D, the conv layer's $\mathbf{W}$ is a banded **Toeplitz** matrix: zeros outside the band (locality), constant along each diagonal (sharing). So a CNN is a special case of an MLP; see [[linear-mapping]].
- **Equivariance:** with shift operator $T_s$, $\mathbf{K} * (T_s\mathbf{x}) = T_s(\mathbf{K} * \mathbf{x})$. The linear maps that commute with all (circular) shifts are exactly the circulant convolutions; these are diagonalised by the discrete Fourier transform (convolution theorem) — compare [[eigendecomposition]].
- **Expressivity vs efficiency:** one-hidden-layer MLPs are universal approximators (Cybenko 1989; Hornik 1991). CNNs are not more expressive; they are a better **inductive bias** for grid data, so they generalise from less data.
- **Multi-channel:** $\mathbf{K}\in\mathbb{R}^{C_\text{out}\times C_\text{in}\times k\times k}$, parameters $C_\text{out}(C_\text{in}k^2+1)$. A 1×1 convolution is an MLP applied per pixel.
- Libraries implement cross-correlation (no kernel flip) and call it convolution; with learned kernels the difference is irrelevant.

## Foundations — what to learn first
Read these in order. Each layer of the list needs the one above it.
1. **Data as numbers:** [[vector]] → [[matrix]] → [[tensor]]
2. **The core operation:** [[inner-product]] (dot product) → [[matrix-multiplication]] → [[linear-mapping]]
3. **One neuron, one layer:** [[bias-vector]] → [[activation-function]] ([[relu]], [[sigmoid-function]]) → [[linear-layer]] → [[forward-pass]]
4. **Stacking layers:** [[neural-network]] → [[multilayer-perceptron]] → [[model-parameters]]
5. **Learning:** [[loss-function]] ([[cross-entropy-loss]]) → [[gradient]] / [[partial-derivative]] → [[chain-rule]] → [[backpropagation]] → [[gradient-descent]] / [[stochastic-gradient-descent]]
6. **Why the CNN's constraints help:** [[overfitting]] → [[regularization]] (weight sharing acts like a built-in regularizer)
7. **CNN-only ideas** (no wiki page yet): convolution / cross-correlation, local receptive field, weight sharing, pooling, translation equivariance — all explained on this page above.

## Connections
- Builds on: [[matrix]], [[linear-mapping]], [[vector]], [[multilayer-perceptron]], [[backpropagation]]
- Leads to: [[convolutional-neural-network]], [[multilayer-perceptron]]
- Used by: [[backpropagation]], [[gradient-descent]]

## Sources
- No ingested source yet (see warning above).
