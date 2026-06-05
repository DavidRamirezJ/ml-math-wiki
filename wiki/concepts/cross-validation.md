---
title: Cross-Validation
type: concept
tags: [machine-learning, generalization, model-selection]
difficulty: beginner
sources: [mml-ch08-when-models-meet-data]
created: 2026-06-04
updated: 2026-06-04
---

# Cross-Validation

> **One-liner:** Cross-validation estimates how well a model will do on new data by repeatedly hiding part of your data, training on the rest, and testing on the hidden part.

## 🟢 Intuition (Dummy)
You want to know if your study method will work on the *real* exam, but you only have a stack of
practice questions. So you hide a few, study the rest, then test yourself on the hidden ones — and
repeat, hiding a different few each time. Average your scores: that's a fair guess of your real-exam
performance, and you used *all* your questions for both studying and testing along the way.

## 🟡 Mechanics (Practitioner)
We care about the **expected risk** on unseen data, but we only have a finite dataset. Split it into
a **training set** (fit the model) and a held-out **validation / test set** (estimate
generalization). One split wastes data and gives a noisy estimate, so use
**$K$-fold cross-validation** (MML §8.2.4):

1. Partition the data into $K$ equal chunks.
2. For each $k=1,\dots,K$: use chunk $k$ as the validation set $\mathcal V^{(k)}$ and the other
   $K-1$ chunks as the training set $\mathcal R^{(k)}$; train predictor $f^{(k)}$ on $\mathcal R^{(k)}$
   and measure its risk on $\mathcal V^{(k)}$.
3. Average the $K$ risks:
$$ \mathbb E_{\mathcal V}\big[R(f,\mathcal V)\big]\ \approx\ \frac1K\sum_{k=1}^{K} R\big(f^{(k)},\mathcal V^{(k)}\big). $$

The risk $R$ is often RMSE for regression. Every example serves once for validation and $K-1$ times
for training.

**Worked example.** $N=100$ points, $K=5$: five rounds, each training on 80 and validating on 20;
average the five RMSEs. The approximation has two error sources: finite training data (so $f^{(k)}$
isn't optimal) and finite validation data (noisy risk estimate).

**Nested cross-validation (§8.6.1)** for **model selection**: an **inner** CV loop picks the best
model / [[regularization|hyperparameter]] using a validation set; an **outer** CV loop then estimates
the generalization error of that chosen model on a test set. Never tune on data you'll report on.

**ML usage.** The default way to choose hyperparameters (regularization strength, polynomial degree,
kernel) and to report honest generalization performance. It is **embarrassingly parallel** — the $K$
runs are independent.

## 🔴 Theory (Expert)
- CV estimates $R_{\text{true}}(f)=\mathbb E_{\mathbf x,y}[\ell(y,f(\mathbf x))]$; it is nearly
  unbiased for the risk of a model trained on $\frac{K-1}{K}N$ points, with variance shrinking in $K$
  (the standard error is $\sigma/\sqrt K$).
- **Leakage:** even *observing* test-set performance and re-training leaks information (Blum & Hardt
  2015); nested CV exists precisely to keep model selection from contaminating the generalization
  estimate. Leave-one-out is the $K=N$ extreme (low bias, high variance, $N$ refits).
- Alternatives: bootstrap and jackknife (Efron & Tibshirani 1993). CV is the resampling cousin of the
  probabilistic [[bayesian-model-selection|marginal-likelihood]] approach to the same problem.

## Connections
- Builds on: [[empirical-risk-minimization]], [[overfitting-and-underfitting]]
- Leads to: [[bayesian-model-selection]], hyperparameter tuning
- Used by: [[regularization]] (choosing $\lambda$), [[linear-regression]], every supervised method
- People: —

## Sources
- `raw/mml-book.pdf` §8.2.4 (Eq. 8.13, Fig. 8.4) & §8.6.1 (Eq. 8.39, Fig. 8.13), printed pp. 263–264, 284 (PDF pp. 269–270, 290).
