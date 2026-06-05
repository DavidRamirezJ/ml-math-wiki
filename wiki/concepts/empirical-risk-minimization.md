---
title: Empirical Risk Minimization
type: concept
tags: [machine-learning, optimization, supervised-learning]
difficulty: intermediate
sources: [mml-ch08-when-models-meet-data]
created: 2026-06-04
updated: 2026-06-04
---

# Empirical Risk Minimization

> **One-liner:** Pick a family of guess-functions, score how badly each one does on your examples, and keep the one with the smallest average mistake.

## 🟢 Intuition (Dummy)
Imagine you want a rule that predicts someone's salary from their age. You don't know the "true"
rule, so you try out a whole family of candidate rules (say, all straight lines). For each
candidate you look at your real examples and ask: *on average, how far off were you?* The line
that's wrong by the least, on average, across all your examples — that's the one you keep.

That's the whole idea: **average mistake on the data you have, made as small as possible.** The
catch (covered by [[regularization]] and [[cross-validation]]) is that being perfect on the
examples you've *seen* doesn't guarantee being good on examples you *haven't* seen.

## 🟡 Mechanics (Practitioner)
Supervised setup: examples $\mathbf x_n\in\mathbb R^D$ with labels $y_n\in\mathbb R$, $n=1,\dots,N$.
ERM is built from **four design choices** (MML §8.2):

1. **Hypothesis class** — the set of functions the predictor may be. The book uses
   [[linear-mapping|affine functions]]: prepend a constant feature $x^{(0)}=1$ so
   $\mathbf x_n=[1,x^{(1)}_n,\dots,x^{(D)}_n]^\top$ and write
   $$ f(\mathbf x_n,\boldsymbol\theta)=\boldsymbol\theta^\top\mathbf x_n = \theta_0+\sum_{d=1}^{D}\theta_d x^{(d)}_n .$$
2. **[[loss-function]]** — $\ell(y_n,\hat y_n)\ge 0$ scoring a single prediction $\hat y_n=f(\mathbf x_n,\boldsymbol\theta)$.
3. **[[regularization]]** — how to keep the fit from memorizing the training set.
4. **Optimization / search** — how to actually find the minimizer (gradients, Ch. 7).

The object minimized is the **empirical risk** (average training loss, Eq. 8.6):
$$ R_{\text{emp}}(f,X,\mathbf y)=\frac1N\sum_{n=1}^{N}\ell\big(y_n,\,f(\mathbf x_n,\boldsymbol\theta)\big). $$
Minimizing it is **empirical risk minimization.** Using the i.i.d. assumption, $R_{\text{emp}}$ is
a sample estimate of the real target, the **expected (population) risk**
$$ R_{\text{true}}(f)=\mathbb E_{\mathbf x,y}\big[\ell(y,f(\mathbf x))\big], $$
which we can never compute (it needs infinite data) but care about because it measures performance
on *unseen* data.

**Worked example — ordinary least squares (Examples 8.1–8.2).** Squared loss
$\ell(y_n,\hat y_n)=(y_n-\hat y_n)^2$ with the linear predictor gives
$$ \min_{\boldsymbol\theta\in\mathbb R^D}\ \frac1N\sum_{n=1}^N (y_n-\boldsymbol\theta^\top\mathbf x_n)^2
\;=\; \min_{\boldsymbol\theta}\ \frac1N\,\lVert \mathbf y - X\boldsymbol\theta\rVert^2 , $$
with example matrix $X=[\mathbf x_1,\dots,\mathbf x_N]^\top\in\mathbb R^{N\times D}$ and label vector
$\mathbf y\in\mathbb R^N$. This **least-squares problem** has a closed-form solution via the normal
equations (Ch. 9).

**ML usage.** ERM is the backbone of almost all "fit by minimizing a loss" training: linear/logistic
regression, SVMs (where it was popularized), and neural networks (the loss is just non-convex).

## 🔴 Theory (Expert)
- ERM is *not* "probability free": there is an unknown data-generating $p(\mathbf x,y)$, and the
  i.i.d. assumption makes $R_{\text{emp}}\to R_{\text{true}}$ as $N\to\infty$ (law of large numbers).
  Unlike classical statistics, ERM is **agnostic to $p(\mathbf x,y)$** — it never specifies a noise
  model for $y$, and labels may be non-deterministic (§8.2.5).
- The gap $R_{\text{true}}-R_{\text{emp}}$ is the subject of **statistical learning theory**
  (Vapnik): uniform bounds over the hypothesis class control generalization, with capacity measured
  by VC dimension / Rademacher complexity. Richer classes shrink $R_{\text{emp}}$ but loosen the
  bound — the formal face of [[overfitting-and-underfitting]].
- The **performance measure** of a task (accuracy, RMSE, a cost-sensitive metric) and the **loss**
  optimized often differ in practice, for reasons of differentiability and optimization efficiency.
- **Tikhonov regularization** (the $\lambda\lVert\boldsymbol\theta\rVert^2$ penalty) recasts ERM as a
  well-posed inverse problem; the constrained variant is **Ivanov regularization** (§8.2.5).

## Connections
- Builds on: [[vector]], [[linear-mapping]], [[loss-function]]
- Leads to: [[regularization]], [[cross-validation]], [[overfitting-and-underfitting]]
- Used by: [[linear-regression]], [[support-vector-machine]]
- Parallels: [[maximum-likelihood-estimation]] (its probabilistic twin)
- People: [[vladimir-vapnik]]

## Sources
- `raw/mml-book.pdf` §8.2 (Eqs. 8.3–8.13, Examples 8.1–8.2), printed pp. 258–264 (PDF pp. 264–270).
