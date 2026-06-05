---
title: Probability Distribution
type: concept
tags: [probability, foundations, distributions]
difficulty: beginner
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Probability Distribution

> **One-liner:** A probability distribution is the complete bookkeeping of how likely each possible value of a random variable is.

## 🟢 Intuition (Dummy)
If a [[random-variable|random variable]] is a machine that outputs numbers, the distribution is the *odds sheet* telling you how often each number shows up. For a die it's "1/6 for each face". For something continuous like a person's height, you can't list every value, so instead you describe a smooth curve where the **area** under a stretch of the curve is the chance of landing in that stretch. Total area (or total of all the odds) is always exactly 1 — something must happen.

## 🟡 Mechanics (Practitioner)
**Discrete — probability mass function (pmf).** $P(X=x)\ge 0$ and $\sum_{x\in T}P(X=x)=1$.

**Continuous — probability density function (pdf).** A function $f:\mathbb R^D\to\mathbb R$ is a pdf if (Def 6.1):
$$ f(\mathbf x)\ge 0\quad\text{and}\quad \int_{\mathbb R^D} f(\mathbf x)\,d\mathbf x = 1, $$
with probabilities recovered by integration: $P(a\le X\le b)=\int_a^b f(x)\,dx$. **A density can exceed 1** (only the integral must be 1).

**Cumulative distribution function (cdf)** (Def 6.2): $F_X(\mathbf x)=P(X_1\le x_1,\dots,X_D\le x_D)=\int_{-\infty}^{x_1}\!\!\cdots\!\int_{-\infty}^{x_D} f(\mathbf z)\,d\mathbf z$.

**Many variables at once:**
- **Joint** $p(x,y)=P(X{=}x, Y{=}y)$ — the full table.
- **Marginal** $p(x)=\sum_y p(x,y)$ (or $\int p(x,y)\,dy$) — sum/integrate out the others.
- **Conditional** $p(y\mid x)=\dfrac{p(x,y)}{p(x)}$ — restrict to a row/column and renormalize.

| | Point probability | Interval probability |
|---|---|---|
| Discrete | $P(X=x)$ (pmf) | n/a |
| Continuous | $p(x)$ (pdf) | $P(X\le x)$ (cdf) |

**Worked example (frequency table).** With counts $n_{ij}$, $N$ total: $P(X{=}x_i)=\frac{c_i}{N}$ (column sum / $N$), $P(Y{=}y_j\mid X{=}x_i)=\frac{n_{ij}}{c_i}$.

**ML usage:** discrete distributions model categorical features/labels (handwriting classes, dice-like choices); continuous densities model real-valued data and parameters; the joint/marginal/conditional triad is the substrate of all probabilistic modelling.

## 🔴 Theory (Expert)
- pdf vs. law: the pdf $f$ is a non-negative function integrating to 1; the **law** of $X$ is the *association* of $X$ with $f$ via $P(a\le X\le b)=\int_a^b f$. Distinct concepts the book keeps separate (remark after Def 6.2).
- There exist cdfs with no corresponding pdf (e.g. mixed or singular distributions) — the cdf is the more fundamental object.
- The "uniform distribution" illustrates the discrete/continuous gap (Fig 6.3, Ex 6.3): discrete states need no ordering ($z_1=\text{red}$), continuous densities can have height $>1$ on a short support.
- Computing marginals = high-dimensional sums/integrals, which is generally **#P-hard** — the central computational bottleneck of probabilistic modelling.

## Connections
- Builds on: [[random-variable]]
- Leads to: [[bayes-theorem]], [[expected-value-and-mean]], [[gaussian-distribution]]
- Used by: [[exponential-family]], [[conjugate-prior]], density estimation (Ch.11)
- People: —

## Sources
- `raw/mml-book.pdf` §6.2 (Defs 6.1–6.2, Table 6.1), printed pp. 178–183 (PDF pp. 184–189).
