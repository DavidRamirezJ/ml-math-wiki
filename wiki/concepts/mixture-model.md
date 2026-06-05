---
title: Mixture Model
type: concept
tags: [density-estimation, probability, machine-learning]
difficulty: beginner
sources: [mml-ch11-density-estimation-gaussian-mixture-models]
created: 2026-06-04
updated: 2026-06-04
---

# Mixture Model

> **One-liner:** A mixture model builds a complicated bumpy probability curve by blending several simple ones together, each weighted by how much it contributes.

## 🟢 Intuition (Dummy)
Imagine you measure the heights of everyone in a school. If you lump kids and teachers together, the
"one hump" bell curve fits badly — there are really *two* groups with two different typical heights. A
mixture model says: instead of forcing one simple shape onto the data, take a few simple shapes (one per
group) and **blend** them. Each shape gets a weight saying "what fraction of the data comes from this
group." Add the weighted shapes up and you get a curve with several bumps — far more flexible than any
single simple curve, perfect for data that comes in clusters.

## 🟡 Mechanics (Practitioner)
A mixture model describes a density $p(x)$ as a **convex combination** of $K$ simple **base** (component) distributions:
$$ p(x) = \sum_{k=1}^{K} \pi_k\, p_k(x), \qquad 0 \le \pi_k \le 1,\quad \sum_{k=1}^{K}\pi_k = 1 .$$
- The $p_k$ come from a chosen family — Gaussians, Bernoullis, Gammas, etc.
- The $\pi_k$ are the **mixture weights** (a probability vector): how much each component contributes.

Because the weights sum to 1 and are nonnegative, $p(x)$ is itself a valid density. The model is **more
expressive** than its base family: it can represent **multimodal** data (several clusters), which a single
unimodal component cannot.

**Worked example (mini).** Two 1-D Gaussian components, $p(x)=0.7\,\mathcal N(x\mid 0,1)+0.3\,\mathcal N(x\mid 5,1)$,
produces a curve with a tall hump near $0$ and a shorter hump near $5$ — impossible with one Gaussian.

**ML usage.** Density estimation, clustering (soft cluster membership), generative modeling, anomaly
detection. The dominant instance is the [[gaussian-mixture-model]].

## 🔴 Theory (Expert)
- A mixture is a **finite convex combination** in the space of probability densities; the weight vector
  $\boldsymbol\pi=[\pi_1,\dots,\pi_K]^\top$ lives on the $(K-1)$-simplex.
- Mixtures admit a **[[latent-variable-model|latent-variable]]** representation: introduce a one-hot
  indicator $z$ with $p(z_k=1)=\pi_k$ and $p(x\mid z_k=1)=p_k(x)$; marginalizing $z$ recovers
  $p(x)=\sum_k \pi_k p_k(x)$. This is what makes the [[expectation-maximization|EM algorithm]] applicable.
- **Identifiability** is only up to permutation of the components (label switching), and mixtures of
  finitely many components are not always identifiable for all base families — a subtlety in inference.
- Finite mixtures are **universal approximators** of densities as $K\to\infty$ (e.g., GMMs approximate any
  smooth density arbitrarily well), the theoretical reason they are so widely used.

## Connections
- Builds on: [[gaussian-distribution]], probability density, convex combination
- Leads to: [[gaussian-mixture-model]], [[latent-variable-model]]
- Used by: [[gaussian-mixture-model]], clustering, [[kernel-density-estimation]] (a degenerate "one component per point" mixture)
- People: —

## Sources
- `raw/mml-book.pdf` §11 intro & §11.1 (Eqs. 11.1–11.2), printed pp. 348–349 (PDF pp. 354–355).
