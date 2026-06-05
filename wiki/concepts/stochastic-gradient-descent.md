---
title: Stochastic Gradient Descent
type: concept
tags: [optimization, gradient-descent, machine-learning, large-scale]
difficulty: intermediate
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Stochastic Gradient Descent

> **One-liner:** Instead of measuring the whole hillside before each step, you sample a few spots, guess the downhill direction, and step — much faster, and the guessing even helps you escape bad ditches.

## 🟢 Intuition (Dummy)
Ordinary [[gradient-descent]] looks at *all* your training data to decide each step. If
you have a million examples, that's a million measurements per step — incredibly slow.
The trick: you don't need a perfect downhill direction, just a *roughly* correct one. So
grab a small random handful of examples ("mini-batch"), estimate the slope from those,
and step. Repeat with a new handful. On average the random guesses point the right way,
so you still get to the bottom — and far faster. A bonus: the randomness jiggles you out
of shallow ditches you might otherwise get stuck in.

## 🟡 Mechanics (Practitioner)
ML objectives are usually a **sum over data points**:
$$ L(\boldsymbol\theta) = \sum_{n=1}^{N} L_n(\boldsymbol\theta), $$
e.g. the negative log-likelihood $L(\boldsymbol\theta)=-\sum_{n=1}^N \log p(y_n\mid \mathbf{x}_n,\boldsymbol\theta)$.

**Batch** gradient descent uses the full sum each step:
$$ \boldsymbol\theta_{i+1} = \boldsymbol\theta_i - \gamma_i \sum_{n=1}^{N}\big(\nabla L_n(\boldsymbol\theta_i)\big)^\top. $$
Evaluating all $N$ gradients is expensive. **SGD** replaces the sum with a sum over a
random subset (mini-batch) — in the extreme, a single example. This works because the
full sum is an **empirical estimate of the expected gradient**, and *any* unbiased
estimate (such as a random subsample) suffices for convergence.

**The trade-off (batch size):**
| Large mini-batch | Small mini-batch |
|---|---|
| accurate gradient, low variance | noisy gradient, high variance |
| stable convergence | can escape bad local optima |
| each step costs more | each step is cheap |
| exploits vectorized matrix ops | — |

**ML usage:** SGD (and adaptive variants like Adam) is **the** training algorithm for
large-scale ML — deep nets on millions of images, topic models, reinforcement learning,
large Gaussian processes. The size of central goal is generalization (Ch. 8), so a
precise minimum isn't even needed, making approximate gradients doubly attractive.

## 🔴 Theory (Expert)
The "stochastic" label means we admit we only have a **noisy approximation** of the
gradient. Convergence is guaranteed by constraining the distribution of the approximate
gradients: it suffices that the estimate is **unbiased**,
$\mathbb{E}[\widehat{\nabla L}] = \nabla L$.

- **Robbins–Monro / Bottou (1998):** when the learning rate $\gamma_i$ decreases at an
  appropriate rate (e.g. $\sum_i\gamma_i=\infty$, $\sum_i\gamma_i^2<\infty$) and under
  mild assumptions, SGD converges **almost surely** to a local minimum.
- The mini-batch gradient is exactly the empirical mean of per-example gradients
  (Section 6.4.1), so reducing batch size raises the **variance** of the estimate — the
  same variance/sample-size relationship as in Monte Carlo mean estimation.
- [[gradient-descent|Momentum]] is especially valuable here, since averaging successive
  noisy estimates reduces variance in the parameter update.

## Connections
- Builds on: [[gradient-descent]], [[gradient]]
- Leads to: neural-network optimization, [[expected-value-and-mean|expected value]]
- Used by: deep learning, [[linear-regression]], large-scale ML
- People: —

## Sources
- `raw/mml-book.pdf` §7.1.3 (Eqs. 7.13–7.15), printed pp. 231–233 (PDF pp. 236–238).
