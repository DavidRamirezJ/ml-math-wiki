---
title: Backpropagation
type: concept
tags: [calculus, deep-learning, optimization, differentiation]
difficulty: advanced
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Backpropagation

> **One-liner:** Backpropagation computes a neural network's gradients cheaply by applying the chain rule backward through the network, reusing work at every layer.

## 🟢 Intuition (Dummy)
A deep network is a tower of functions: the input passes through layer after
layer until it produces a prediction, and an error score says how wrong it was.
To improve, every knob (weight) needs to know "if I nudge you, how much does the
error change?" Computing that one knob at a time, from scratch, would be hugely
wasteful. **Backpropagation** is the clever shortcut: start the blame at the
output error and pass it *backward* through the layers, so each layer reuses the
blame already computed by the layer above it. The total work to get *all* the
gradients ends up about the same as running the network once forward.

## 🟡 Mechanics (Practitioner)
A network is a many-level [[chain-rule|composition]] (5.111):
$$\mathbf y=(f_K\circ f_{K-1}\circ\cdots\circ f_1)(\mathbf x),$$
where each layer (5.113) is $\mathbf f_i=\sigma_i(A_{i-1}\mathbf f_{i-1}+\mathbf b_{i-1})$
with weights $A_{i-1}$, biases $\mathbf b_{i-1}$, and activation $\sigma$
(sigmoid $\tfrac{1}{1+e^{-x}}$, $\tanh$, or ReLU). With squared loss
$L(\boldsymbol\theta)=\lVert\mathbf y-\mathbf f_K(\boldsymbol\theta,\mathbf x)\rVert^2$
and parameters $\boldsymbol\theta=\{A_0,\mathbf b_0,\dots\}$, the chain rule gives
the gradients **layer by layer**:
$$
\frac{\partial L}{\partial\theta_{K-1}}=\frac{\partial L}{\partial \mathbf f_K}\frac{\partial \mathbf f_K}{\partial\theta_{K-1}},\qquad
\frac{\partial L}{\partial\theta_{i}}=\frac{\partial L}{\partial \mathbf f_K}\frac{\partial \mathbf f_K}{\partial \mathbf f_{K-1}}\cdots\frac{\partial \mathbf f_{i+1}}{\partial\theta_i}.
$$
The key efficiency: once $\partial L/\partial\theta_{i+1}$ (equivalently
$\partial L/\partial\mathbf f_{i+1}$) is known, computing $\partial L/\partial\theta_i$
**reuses** it — only the local factors $\partial\mathbf f_{i+1}/\partial\mathbf f_i$
(output-w.r.t.-input) and $\partial\mathbf f_{i+1}/\partial\theta_i$
(output-w.r.t.-parameters) are new. Gradients are passed *backward* through the
network (Fig. 5.9).

**The algorithm:**
1. **Forward pass:** compute and cache each layer's output $\mathbf f_i$ and the loss $L$.
2. **Backward pass:** initialize $\partial L/\partial\mathbf f_K$, then for $i=K-1,\dots,0$ multiply by the cached local Jacobians to get $\partial L/\partial\mathbf f_i$ and read off $\partial L/\partial\theta_i$.
3. **Update:** take a gradient-descent step on each $\theta_i$ (Ch. 7).

## 🔴 Theory (Expert)
- Backpropagation is precisely **reverse-mode [[automatic-differentiation]]**
  applied to the network's computation graph. The "reverse vs. forward" choice is
  *which order* you multiply the Jacobian chain
  $\frac{dy}{dx}=\frac{dy}{db}\frac{db}{da}\frac{da}{dx}$; associativity of matrix
  multiplication makes both valid (5.120–5.121).
- **Why reverse mode for nets.** When the input dimension $\gg$ output dimension
  (typical: many parameters, scalar loss), grouping right-to-left keeps the
  intermediate objects small (row-vector $\times$ matrix), so reverse mode is far
  cheaper than forward mode.
- Cost result: computing the gradient is of the *same order* as computing the
  function — counterintuitive given how long the symbolic derivative expression
  (e.g. 5.110) is.
- Historical lineage (Kelley 1960; Bryson 1961; Dreyfus 1962; Rumelhart, Hinton &
  Williams 1986). Requires differentiable elementary operations; control flow
  (loops, `if`) needs care.

## Connections
- Builds on: [[chain-rule]], [[jacobian]], [[gradient]], [[partial-derivative]]
- Leads to: [[automatic-differentiation]], gradient descent (Ch. 7)
- Used by: training deep neural networks, autoencoders
- People: —

## Sources
- `raw/mml-book.pdf` §5.6.1 (5.111–5.118), printed pp. 159–161 (PDF pp. 165–167).
