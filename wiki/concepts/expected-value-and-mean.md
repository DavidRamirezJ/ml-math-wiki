---
title: Expected Value and Mean
type: concept
tags: [probability, statistics, summary-statistics]
difficulty: beginner
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Expected Value and Mean

> **One-liner:** The expected value is the long-run average outcome you'd get if you repeated a random experiment forever.

## 🟢 Intuition (Dummy)
Roll a fair die many, many times and average the results. You won't land on any single face's value, but the average settles near 3.5. That balance point is the **expected value** — the "center of mass" of a distribution. If you slid a finger under a cardboard cutout of the probability bar chart, the expected value is where it balances. The **mean** is just the expected value of the variable itself; the **median** is the middle value, and the **mode** is the most common value.

## 🟡 Mechanics (Practitioner)
**Expected value of a function** $g$ of a random variable $X\sim p(x)$ (Def 6.3):
$$ \mathbb E_X[g(x)]=\int_{\mathcal X} g(x)\,p(x)\,dx\ \text{(continuous)},\qquad \mathbb E_X[g(x)]=\sum_{x\in\mathcal X} g(x)\,p(x)\ \text{(discrete)}. $$

**Mean** = expected value of the identity (Def 6.4); for a vector-valued $X$ it is taken element-wise:
$$ \mathbb E_X[\mathbf x]=\big[\,\mathbb E_{X_1}[x_1],\dots,\mathbb E_{X_D}[x_D]\,\big]^\top\in\mathbb R^D. $$

**Linearity** (the property you use constantly): for $f=ag+bh$,
$$ \mathbb E_X[a\,g(x)+b\,h(x)]=a\,\mathbb E_X[g(x)]+b\,\mathbb E_X[h(x)]. $$

**Other "averages":** **median** = value where $F_X=0.5$ (robust to outliers, hard to generalize to $>1$ dim since there is no ordering of vectors); **mode** = peak(s) of $p(x)$ (a distribution can be multimodal).

**Empirical / sample mean** of data $\mathbf x_1,\dots,\mathbf x_N$ (Def 6.9):
$$ \bar{\mathbf x}=\frac1N\sum_{n=1}^N \mathbf x_n. $$

**Worked example.** Bernoulli with $P(X{=}1)=\mu$: $\mathbb E[x]=1\cdot\mu+0\cdot(1-\mu)=\mu$.

**ML usage:** the expectation operator is everywhere — risk/loss is an expected loss, the marginal likelihood is $\mathbb E_X[p(y\mid x)]$, gradients in training are estimated as sample means (Monte Carlo / SGD), and "the foundational concepts of probability can be derived from the expected value" (p. 187).

## 🔴 Theory (Expert)
- $\mathbb E$ is a **linear operator** on the space of integrable functions of $X$ — this single fact drives the algebra of means and the mean of [[gaussian-distribution|Gaussian]]/exponential-family models.
- $\mathbb E[g(X)]$ via the distribution of $X$ (not of $g(X)$) is the **law of the unconscious statistician**.
- **Iterated expectation / tower rule** (Exercise 6.11): $\mathbb E_X[x]=\mathbb E_Y[\mathbb E_X[x\mid y]]$.
- Mean exists only if the integral converges absolutely — heavy-tailed laws (e.g. Cauchy) have *no* mean, where the median still describes a typical value.
- In high dimensions the mode is computationally hard to find (possibly exponentially many modes), and the median has no canonical definition (no total order on $\mathbb R^D$).

## Connections
- Builds on: [[random-variable]], [[probability-distribution]]
- Leads to: [[covariance-and-variance]], [[bayes-theorem]]
- Used by: [[gaussian-distribution]], [[exponential-family]], empirical risk minimization, Monte Carlo estimation
- People: —

## Sources
- `raw/mml-book.pdf` §6.4.1 (Defs 6.3–6.4, 6.9), printed pp. 186–189, 191–192 (PDF pp. 192–195, 197–198).
