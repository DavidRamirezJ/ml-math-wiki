---
title: Kernel Density Estimation (KDE)
type: concept
tags: [density-estimation, nonparametric, machine-learning]
difficulty: intermediate
sources: [mml-ch11-density-estimation-gaussian-mixture-models]
created: 2026-06-04
updated: 2026-06-04
---

# Kernel Density Estimation (KDE)

> **One-liner:** Kernel density estimation draws a small smooth bump on top of every data point and adds them all up to get a smooth picture of where the data lives.

## 🟢 Intuition (Dummy)
A histogram chops the number line into bins and counts how many points land in each — but it's blocky and
its look changes a lot depending on where you put the bin edges. KDE fixes the blockiness: instead of
counting into bins, you place a little smooth hill (a "kernel") centered on each data point, then add all
the hills together. Where points are dense, the hills pile up high; where points are sparse, the curve is
low. The result is a smooth estimate of the density, with no arbitrary bin edges — just one knob: how wide
to make each hill.

## 🟡 Mechanics (Practitioner)
Given $N$ i.i.d. samples $x_1,\dots,x_N$, the kernel density estimator is
$$ p(x) = \frac{1}{Nh}\sum_{n=1}^{N} k\!\left(\frac{x - x_n}{h}\right), $$
where:
- $k(\cdot)$ is a **kernel function** — a nonnegative function that integrates to 1 (e.g. a Gaussian or
  uniform shape);
- $h>0$ is the **bandwidth** (smoothing parameter), analogous to a histogram's bin width.

This is **nonparametric**: no fixed number of parameters; the whole dataset *is* the model. Small $h$ →
spiky/overfit; large $h$ → oversmoothed. The bandwidth is the critical hyperparameter, tuned by
cross-validation.

**Worked example (mini).** With a Gaussian kernel $k(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}$, KDE places a
Gaussian of standard deviation $h$ on each point and averages them — a smooth curve following the data's
shape, contrasted in the book against the blocky [[histogram]] of the same 250 points (Fig. 11.13).

**ML usage.** Quick density estimation/visualization, anomaly detection, smoothing, generative baselines —
whenever you want a density without committing to a parametric family like a Gaussian or a [[gaussian-mixture-model|GMM]].

## 🔴 Theory (Expert)
- Independently proposed by **Rosenblatt (1956)** and **Parzen (1962)** (hence "Parzen window").
- KDE can be read as a degenerate [[mixture-model]] with **one component per data point**, each weighted
  $1/N$ and shaped by the kernel — versus a GMM, which uses a small fixed $K$ of *learned* components.
- The bandwidth controls a **bias–variance trade-off**: as $N\to\infty$ with $h\to 0$ and $Nh\to\infty$,
  KDE is a consistent estimator of the true density. The optimal $h$ scales like $N^{-1/(d+4)}$ (the
  curse of dimensionality makes KDE weak in high dimensions).
- Versus parametric estimation: KDE makes minimal assumptions but needs all the data at prediction time and
  degrades in high dimension; parametric models (GMM) compress the data into few parameters but can
  misspecify the density.

## Connections
- Builds on: [[gaussian-distribution]] (common kernel), probability density, [[histogram]]
- Leads to: nonparametric Bayesian methods, smoothing
- Used by: density estimation, anomaly detection, data visualization
- People: [[emanuel-parzen]]

## Sources
- `raw/mml-book.pdf` §11.5 (Eq. 11.74), printed pp. 368–369 (PDF pp. 374–375).
