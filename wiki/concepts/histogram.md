---
title: Histogram
type: concept
tags: [density-estimation, nonparametric, statistics]
difficulty: beginner
sources: [mml-ch11-density-estimation-gaussian-mixture-models]
created: 2026-06-04
updated: 2026-06-04
---

# Histogram

> **One-liner:** A histogram chops your data's range into bins and draws a bar for each bin as tall as the number of points that fell inside it.

## 🟢 Intuition (Dummy)
Suppose you have a pile of test scores and want to see their "shape." Lay out boxes along the score axis
(0–10, 10–20, …), drop each score into its box, then draw a bar over each box whose height is how many
scores landed there. The result — a row of bars — shows you where scores cluster and where they're sparse.
That's a histogram: the simplest way to turn raw numbers into a picture of their distribution. The one
tricky choice is how wide to make the boxes.

## 🟡 Mechanics (Practitioner)
A histogram is a **nonparametric** density estimate: partition the data space into **bins**, count how
many of the $N$ data points fall into each bin, and draw a bar at the bin center with height proportional
to that count. To read it as a density, normalize so the total area is 1: a bin of width $w$ containing
$n_b$ points gets height
$$ \hat p_{\text{bin}} = \frac{n_b}{N\,w} . $$

- **Bin size (bin width)** is the critical hyperparameter. Too wide → underfitting (oversmoothed, hides
  structure); too narrow → overfitting (spiky, noisy). Choose it by cross-validation.
- It is **unsmoothed**: the estimate is piecewise constant and jumps at bin edges; the result also depends
  on where the edges are placed.

**ML usage.** Fast exploratory data analysis, sanity-checking distributions, and a baseline density
estimator. When you need a *smooth* estimate instead of blocky bars, use [[kernel-density-estimation]].

## 🔴 Theory (Expert)
- Proposed by **Karl Pearson (1895)**; the histogram is the oldest nonparametric density estimator.
- It is a special case of binned density estimation: equivalent to a piecewise-constant maximum-likelihood
  density over a fixed bin partition. KDE generalizes it by replacing the hard bin indicator with a smooth
  kernel and removing the dependence on edge placement.
- **Bias–variance trade-off** governs bin width exactly as bandwidth does for [[kernel-density-estimation|KDE]];
  the optimal width shrinks as $N$ grows. In $d$ dimensions the number of bins grows like (bins-per-axis)$^d$,
  so histograms suffer badly from the curse of dimensionality and are rarely used beyond 1–2 dimensions.

## Connections
- Builds on: counting, binning, probability density
- Leads to: [[kernel-density-estimation]]
- Used by: exploratory data analysis, density-estimation baselines
- People: —

## Sources
- `raw/mml-book.pdf` §11.5, printed pp. 368–369 (PDF pp. 374–375).
