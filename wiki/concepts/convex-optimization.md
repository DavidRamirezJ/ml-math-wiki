---
title: Convex Optimization
type: concept
tags: [optimization, convex-optimization, duality]
difficulty: advanced
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Convex Optimization

> **One-liner:** If your landscape is a single smooth bowl, then any valley you find is *the* lowest point — so optimization becomes safe and the answer is guaranteed best.

## 🟢 Intuition (Dummy)
Most hilly landscapes have many valleys, and [[gradient-descent]] only finds whichever
one you start near. But a **convex** landscape is a single bowl — no false bottoms, no
side valleys. Roll a ball anywhere and it lands at the one true minimum. That's why ML
designers love convex objectives: you never worry about getting stuck in the wrong
place. Two shapes matter: a **convex set** (a blob with no dents — any straight line
between two points stays inside) and a **convex function** (a bowl — any chord between
two points lies above the curve).

## 🟡 Mechanics (Practitioner)
**Convex set (Def. 7.2):** $C$ is convex if for all $\mathbf{x},\mathbf{y}\in C$ and
$\theta\in[0,1]$,
$$ \theta\mathbf{x} + (1-\theta)\mathbf{y} \in C. $$
(The segment between any two points stays in the set.)

**Convex function (Def. 7.3):** $f$ with convex domain is convex if for all
$\mathbf{x},\mathbf{y}$ and $\theta\in[0,1]$,
$$ f(\theta\mathbf{x}+(1-\theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}). $$
This is **Jensen's inequality**. A **concave** function is $-f$ for convex $f$.

**The convex program (7.38):**
$$ \min_{\mathbf{x}} f(\mathbf{x})\ \text{ s.t. } g_i(\mathbf{x})\le0\ (\forall i),\ h_j(\mathbf{x})=0\ (\forall j), $$
where $f$ and all $g_i$ are convex functions and the $h_j=0$ define convex sets. Key
payoff: **every local minimum is global**, and we get **strong duality** — the
[[lagrange-multipliers|Lagrangian dual]] optimum equals the primal optimum.

**Checking convexity in practice.** Rather than first principles, use
**convexity-preserving operations** — the same "closure" idea as [[vector-space|vector spaces]]:
- **Nonnegative weighted sums:** if $f_1,f_2$ convex and $\alpha,\beta\ge0$, then
  $\alpha f_1+\beta f_2$ is convex (Ex. 7.4).
- Pointwise max of convex functions, affine composition, etc.

**Worked example (Ex. 7.3):** $f(x)=x\log_2 x$ (negative entropy) is convex for $x>0$.
Midpoint check at $x=2,4$: LHS $f(3)=3\log_2 3\approx4.75$, RHS
$0.5(2\log_2 2)+0.5(4\log_2 4)=1+4=5$, so $4.75\le 5$. ✓

**ML usage:** Many ML losses are deliberately convex (least squares, logistic loss, SVM
hinge) so training has a unique, certifiable optimum. See [[linear-programming]],
[[quadratic-programming]].

## 🔴 Theory (Expert)
**Differentiable characterizations.** If $f$ is differentiable (Section 5.2):
$$ f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla_{\mathbf{x}}f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) \quad\text{(7.31)}, $$
i.e. the function lies above all its tangent planes (first-order condition). If twice
differentiable, $f$ is convex **iff** the **Hessian** $\nabla^2_{\mathbf{x}}f(\mathbf{x})$
is **positive semidefinite** everywhere.

- **Epigraph:** the set of points on or above the graph of $f$ — "filling the bowl with
  water." $f$ is convex **iff** its epigraph is a convex set, the bridge linking convex
  functions and convex sets.
- **Jensen's inequality** generalizes (7.30) to any nonnegative-weighted combination
  (and to expectations: $f(\mathbb{E}[X])\le\mathbb{E}[f(X)]$), all named after
  [[johan-jensen]].
- **Strong duality** rests on convexity allowing the min and max in the
  [[lagrange-multipliers|minimax inequality]] to be swapped with equality (constraint
  qualifications such as Slater's condition assumed).
- A convex set is equivalently the intersection of its **supporting hyperplanes**; this
  dual description of convex functions is formalized by the [[convex-conjugate]].

## Connections
- Builds on: [[gradient]], [[hessian]], [[vector-space]]
- Leads to: [[linear-programming]], [[quadratic-programming]], [[convex-conjugate]]
- Used by: SVMs (Ch. 12), [[lagrange-multipliers]], logistic regression
- People: [[johan-jensen]]

## Sources
- `raw/mml-book.pdf` §7.3 (Defs. 7.2–7.3, Eqs. 7.29–7.38, Examples 7.3–7.4), printed pp. 236–239 (PDF pp. 241–244).
