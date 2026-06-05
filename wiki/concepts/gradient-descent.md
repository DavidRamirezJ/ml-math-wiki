---
title: Gradient Descent
type: concept
tags: [optimization, gradient-descent, machine-learning]
difficulty: beginner
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Gradient Descent

> **One-liner:** To find the bottom of a valley you can't see, keep taking small steps in whatever direction is most steeply downhill from where you stand.

## 🟢 Intuition (Dummy)
Imagine you're on a foggy hillside and want to reach the lowest point. You can't see
far, but you can feel which way the ground tilts under your feet. So you take a step
downhill, feel again, step again. Repeat until the ground is flat — you've reached a
valley bottom.

That's gradient descent. The "objective function" is the landscape, and "training a
model" means rolling a ball down it to the lowest point. One catch: if there are
several valleys, you only reach the one whose slope you happened to be standing on. And
your step size matters — tiny steps take forever, giant steps can launch you across the
valley and back out the other side.

## 🟡 Mechanics (Practitioner)
We want to solve $\min_{\mathbf{x}} f(\mathbf{x})$ for a differentiable
$f:\mathbb{R}^d\to\mathbb{R}$. The **gradient** $\nabla f$ points in the direction of
steepest *ascent*, so we step the opposite way. Starting from a guess $\mathbf{x}_0$,
iterate:
$$ \mathbf{x}_{i+1} = \mathbf{x}_i - \gamma_i\,\big((\nabla f)(\mathbf{x}_i)\big)^\top $$
For a small enough **step-size** $\gamma_i\ge 0$, $f(\mathbf{x}_{i+1})\le f(\mathbf{x}_i)$,
so the sequence $f(\mathbf{x}_0)\ge f(\mathbf{x}_1)\ge\cdots$ converges to a *local*
minimum. (The transpose appears because gradients are row vectors by the book's
convention; the parameters are columns.)

**Step-size = learning rate.** Too small → painfully slow; too large → overshoot,
oscillate, or diverge. Adaptive heuristic: if a step *increases* $f$, undo it and shrink
$\gamma$; if it decreases $f$, try a bigger $\gamma$ next time.

**Momentum** (Eq. 7.11) gives the ball memory so it doesn't bounce between valley walls:
$$ \mathbf{x}_{i+1} = \mathbf{x}_i - \gamma_i\big((\nabla f)(\mathbf{x}_i)\big)^\top + \alpha\,\Delta\mathbf{x}_i,\qquad \Delta\mathbf{x}_i = \mathbf{x}_i-\mathbf{x}_{i-1} $$
with $\alpha\in[0,1]$. This implements a moving average of past gradients, dampening
oscillations and averaging out noise.

**Worked example (Ex. 7.1):** minimize the quadratic
$f(\mathbf{x})=\tfrac12\mathbf{x}^\top\big[\begin{smallmatrix}2&1\\1&20\end{smallmatrix}\big]\mathbf{x}-\big[5,3\big]\mathbf{x}$
with gradient $\nabla f(\mathbf{x})=\mathbf{x}^\top\big[\begin{smallmatrix}2&1\\1&20\end{smallmatrix}\big]-\big[5,3\big]$.
From $\mathbf{x}_0=\big[-3,-1\big]^\top$ with $\gamma=0.085$, one step gives
$\mathbf{x}_1=\big[-1.98,1.21\big]^\top$, then $\mathbf{x}_2=\big[-1.32,-0.42\big]^\top$, converging to the minimum.

**ML usage:** the default training loop for almost every model. The objective is the
loss; the parameters are weights. For large data, use [[stochastic-gradient-descent]].

## 🔴 Theory (Expert)
Gradient descent is a **first-order** method: it ignores curvature. Its asymptotic
convergence rate is poor on **ill-conditioned** problems — long, thin valleys where the
gradient points nearly orthogonal to the direction toward the optimum, causing
characteristic "zigzag" (Fig. 7.3).

For the linear-system least-squares objective $\|A\mathbf{x}-\mathbf{b}\|^2$ (Ex. 7.2,
gradient $2(A\mathbf{x}-\mathbf{b})^\top A$), the convergence speed depends on the
**condition number** $\kappa = \sigma_{\max}(A)/\sigma_{\min}(A)$, the ratio of largest
to smallest [[singular-value-decomposition|singular value]] — the ratio of the most-
to least-curved direction. **Preconditioning** solves $P^{-1}(A\mathbf{x}-\mathbf{b})=0$
with $P^{-1}$ chosen so $P^{-1}A$ is better conditioned yet $P^{-1}$ is cheap.

- **Stationary points** are roots of $\nabla f=0$; classify via the **Hessian**
  $\nabla^2 f$ (positive definite → min, negative definite → max, indefinite → saddle).
- The chapter notes (via the **Abel–Ruffini theorem**, [[niels-henrik-abel]]) that
  degree $\ge 5$ polynomials have no general algebraic root formula — hence iterative
  methods even in 1-D.
- Beyond first order: **Newton/quasi-Newton** (L-BFGS) use Hessian information;
  **conjugate gradient** reuses past directions; **mirror descent** and **natural
  gradient** change the descent metric (§7.4).
- For [[convex-optimization|convex]] $f$, every local minimum is global, removing the
  starting-point dependence entirely.

## Connections
- Builds on: [[gradient]], [[vector]]
- Leads to: [[stochastic-gradient-descent]], [[convex-optimization]]
- Used by: neural-network training, [[linear-regression]], [[quadratic-programming]]
- People: [[niels-henrik-abel]]

## Sources
- `raw/mml-book.pdf` §7.1 (Eqs. 7.4–7.12, Examples 7.1–7.2), printed pp. 227–232 (PDF pp. 232–237).
