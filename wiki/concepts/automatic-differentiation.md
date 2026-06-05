---
title: Automatic Differentiation
type: concept
tags: [calculus, deep-learning, differentiation, numerical-methods]
difficulty: advanced
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Automatic Differentiation

> **One-liner:** Automatic differentiation computes the exact gradient of any program by breaking it into elementary steps and chaining their known derivatives together.

## 🟢 Intuition (Dummy)
You don't need a clever closed-form formula for a function's derivative. Any
computation, no matter how tangled, is built from tiny known steps — add,
multiply, $\sin$, $\exp$. We know the derivative of each tiny step exactly. So
record the steps the program actually took (a **computation graph**), then walk
through it multiplying the little derivatives together using the chain rule. The
machine assembles the exact gradient automatically — no pen-and-paper calculus,
and no error-prone finite-difference guessing. This is what every modern deep
learning framework does under the hood.

## 🟡 Mechanics (Practitioner)
Rewrite a function with **intermediate variables**. For
$f(x)=\sqrt{x^2+\exp(x^2)}+\cos\!\big(x^2+\exp(x^2)\big)$ (5.122):
$$a=x^2,\quad b=\exp(a),\quad c=a+b,\quad d=\sqrt{c},\quad e=\cos(c),\quad f=d+e.$$
This is the **computation graph** (Fig. 5.11). Each edge has an elementary
derivative:
$$\tfrac{\partial a}{\partial x}=2x,\ \tfrac{\partial b}{\partial a}=\exp(a),\
\tfrac{\partial c}{\partial a}=\tfrac{\partial c}{\partial b}=1,\
\tfrac{\partial d}{\partial c}=\tfrac{1}{2\sqrt c},\
\tfrac{\partial e}{\partial c}=-\sin(c),\
\tfrac{\partial f}{\partial d}=\tfrac{\partial f}{\partial e}=1.$$
**Reverse pass** (start at the output, work back), accumulating via the
[[chain-rule]]:
$$
\tfrac{\partial f}{\partial c}=\tfrac{\partial f}{\partial d}\tfrac{\partial d}{\partial c}+\tfrac{\partial f}{\partial e}\tfrac{\partial e}{\partial c},\quad
\tfrac{\partial f}{\partial b}=\tfrac{\partial f}{\partial c}\cdot1,\quad
\tfrac{\partial f}{\partial a}=\tfrac{\partial f}{\partial b}\exp(a)+\tfrac{\partial f}{\partial c}\cdot1,\quad
\tfrac{\partial f}{\partial x}=\tfrac{\partial f}{\partial a}\cdot2x.
$$
The result is exact (to machine precision), and computing it costs about the
same as evaluating $f$ — even though the explicit symbolic derivative (5.110) is
far longer than $f$ itself.

**ML usage:** this is exactly how PyTorch/TensorFlow/JAX compute gradients;
[[backpropagation]] is reverse-mode AD specialized to neural networks.

## 🔴 Theory (Expert)
**Formalization (5.143–5.145).** Let $x_1,\dots,x_d$ be inputs,
$x_{d+1},\dots,x_{D-1}$ intermediates, $x_D=f$ the output, with each
$x_i=g_i(x_{\mathrm{Pa}(x_i)})$ a function of its parent nodes. Then with
$\partial f/\partial x_D=1$,
$$\frac{\partial f}{\partial x_i}=\sum_{x_j:\,x_i\in\mathrm{Pa}(x_j)}\frac{\partial f}{\partial x_j}\frac{\partial g_j}{\partial x_i}.$$
Equation 5.143 is **forward propagation** of values; 5.145 is **backpropagation**
of gradients.

- **AD ≠ symbolic ≠ numerical.** Unlike symbolic differentiation it never builds
  the giant expression; unlike finite differences it is exact, not an
  approximation.
- **Forward vs. reverse mode.** Both compute $\frac{dy}{dx}=\frac{dy}{db}\frac{db}{da}\frac{da}{dx}$;
  they differ in *association order* (5.120 reverse, 5.121 forward). Reverse mode
  is cheap when outputs $\ll$ inputs (scalar loss, many parameters) — the deep
  learning regime.
- Works on any program whose elementary operations are differentiable;
  branches/loops need care. References: Baydin et al. (2018); Griewank & Walther.

## Connections
- Builds on: [[chain-rule]], [[jacobian]], [[derivative]]
- Leads to: [[backpropagation]] (its reverse-mode specialization)
- Used by: deep learning frameworks, gradient-based optimization (Ch. 7)
- People: —

## Sources
- `raw/mml-book.pdf` §5.6 & §5.6.2 (Example 5.14, 5.119–5.145), printed pp. 159–164 (PDF pp. 165–170).
