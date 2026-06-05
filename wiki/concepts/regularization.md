---
title: Regularization
type: concept
tags: [machine-learning, optimization, generalization]
difficulty: intermediate
sources: [mml-ch08-when-models-meet-data]
created: 2026-06-04
updated: 2026-06-04
---

# Regularization

> **One-liner:** Regularization adds a "keep it simple" penalty to training so the model fits the data without contorting itself into something that won't work on new data.

## 🟢 Intuition (Dummy)
Give a student a practice exam and unlimited time and they might *memorize* every answer — perfect
on practice, useless on the real test. Regularization is the teacher saying "your explanation has to
be simple, not just correct on these examples." It charges a fee for complicated, extreme solutions,
nudging the learner toward an explanation that's modest enough to also work on questions it has
never seen.

## 🟡 Mechanics (Practitioner)
Plain [[empirical-risk-minimization]] can drive training loss to zero by memorizing the data —
classic [[overfitting-and-underfitting|overfitting]]. Regularization **biases the search** by adding
a penalty on the parameters (MML §8.2.3). For least squares:
$$ \underbrace{\min_{\boldsymbol\theta}\ \tfrac1N\lVert \mathbf y - X\boldsymbol\theta\rVert^2}_{\text{fit the data}}
\;\;\longrightarrow\;\;
\min_{\boldsymbol\theta}\ \tfrac1N\lVert \mathbf y - X\boldsymbol\theta\rVert^2 \;+\; \lambda\,\lVert\boldsymbol\theta\rVert^2 . $$
- $\lVert\boldsymbol\theta\rVert^2$ is the **regularizer / penalty term**; it pulls $\boldsymbol\theta$ toward the origin (small weights).
- $\lambda\ge 0$ is the **regularization parameter**: it trades off data fit against parameter size.
  $\lambda=0$ recovers ordinary least squares; large $\lambda$ forces $\boldsymbol\theta\to\mathbf 0$.

$\lambda$ is a **hyperparameter** — not learned by the optimizer, but chosen by
[[cross-validation]].

**Worked intuition.** Overfit models tend to have **large** weights (a wiggly curve needs big
coefficients). Penalizing $\lVert\boldsymbol\theta\rVert^2$ makes the optimizer "pay" for those big
weights, so it keeps them small unless the data fit really demands otherwise.

**ML usage.** This $\ell_2$ penalty is **ridge regression** (a.k.a. weight decay in neural nets,
Tikhonov regularization classically). Swapping in $\lVert\boldsymbol\theta\rVert_1$ gives the
**lasso**, which drives weights exactly to zero (feature selection).

## 🔴 Theory (Expert)
- **Regularizer $\equiv$ prior.** Adding $\lambda\lVert\boldsymbol\theta\rVert^2$ is exactly
  [[maximum-a-posteriori-estimation|MAP estimation]] with a zero-mean Gaussian prior
  $p(\boldsymbol\theta)=\mathcal N(\mathbf 0,\Sigma)$: $-\log p(\boldsymbol\theta)$ contributes a
  quadratic term in $\boldsymbol\theta$. The $\ell_1$ lasso corresponds to a Laplace prior. The
  chapter makes this the **regularization $\leftrightarrow$ prior** half of the ERM/MLE parallel
  (§8.2.3, §8.3.2).
- This is **Tikhonov regularization**, born of ill-posed inverse problems (Neumaier 1998); the
  constrained twin is **Ivanov regularization**. It is intimately tied to the bias–variance
  trade-off: more regularization raises bias, lowers variance.
- In [[support-vector-machine]]s the regularizer is equivalent to a **large margin** (Ch. 12) — the
  geometry of generalization made explicit.

## Connections
- Builds on: [[empirical-risk-minimization]], [[loss-function]]
- Leads to: [[overfitting-and-underfitting]], [[maximum-a-posteriori-estimation]]
- Used by: [[linear-regression]] (ridge), [[support-vector-machine]]
- People: —

## Sources
- `raw/mml-book.pdf` §8.2.3 (Eqs. 8.11–8.12, Example 8.3), printed pp. 262–263 (PDF pp. 268–269); §8.2.5 (Tikhonov/Ivanov), p. 265.
