---
title: Sufficient Statistics
type: concept
tags: [probability, statistics, inference]
difficulty: advanced
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Sufficient Statistics

> **One-liner:** A sufficient statistic is a summary of your data that captures everything the data can tell you about an unknown parameter — so you can throw the raw data away.

## 🟢 Intuition (Dummy)
Suppose you flip a coin 1000 times to estimate how biased it is. Do you need to remember the exact *order* of heads and tails? No — just the *count* of heads. That single number contains all the information about the bias; the ordering is irrelevant noise. A **sufficient statistic** is exactly that: a compressed summary that loses nothing about the quantity you're trying to learn. It's the principled answer to "what should I actually store?"

## 🟡 Mechanics (Practitioner)
A **statistic** is any deterministic function of the data (e.g. the sample mean $\hat\mu=\frac1N\sum x_n$). For a model $p(x\mid\theta)$ with unknown $\theta_0$, a vector of statistics $\phi(x)$ is **sufficient** for $\theta_0$ if it carries all the information the sample holds about $\theta_0$.

**Fisher–Neyman factorization (Theorem 6.14):** $\phi(x)$ is sufficient for $\theta$ **iff** the density factors as
$$ p(x\mid\theta)=h(x)\,g_\theta(\phi(x)), $$
where $h(x)$ does not depend on $\theta$, and $g_\theta$ depends on the data **only through** $\phi(x)$.

**How to read it:** if you can peel the density into (a piece ignoring $\theta$) × (a piece touching the data only via $\phi(x)$), then $\phi(x)$ is sufficient.

**Worked example.** For i.i.d. Gaussians $x_n\sim\mathcal N(\mu,\sigma^2)$, the pair $\big(\sum_n x_n,\ \sum_n x_n^2\big)$ is sufficient for $(\mu,\sigma^2)$ — equivalently the sample mean and sample variance. No matter how big $N$ is, those two numbers suffice.

**ML usage:** tells you the minimal quantities to accumulate when fitting a model — crucial for streaming/online learning and memory-bounded estimation. Empirical sufficient statistics are the optimal estimates of population sufficient statistics, which is why ML fitting of [[exponential-family]] models is just statistic-matching.

## 🔴 Theory (Expert)
- The factorization form $g_\theta(\phi(x))$ is precisely the exponential-family kernel $\exp(\langle\theta,\phi(x)\rangle-A(\theta))$ — so sufficiency and the [[exponential-family]] are two views of the same structure.
- **Pitman–Koopman–Darmois:** under i.i.d. sampling, the exponential families are the *only* families with **finite-dimensional** sufficient statistics whose dimension is independent of the sample size $N$. Outside them (non-parametric models), the number of parameters needed grows with data.
- The notion was introduced by **R. A. Fisher**; it underlies the Rao–Blackwell theorem (conditioning an estimator on a sufficient statistic never increases variance) and minimal/complete sufficiency.
- Trivial case: if $p(x\mid\theta)$ doesn't depend on $\theta$, any $\phi$ is sufficient; the interesting case is when $p(x\mid\theta)$ depends on the data *only* through $\phi(x)$.

## Connections
- Builds on: [[probability-distribution]], [[expected-value-and-mean]]
- Leads to: [[exponential-family]], maximum likelihood estimation
- Used by: online learning, [[gaussian-distribution]] / [[conjugate-prior]] updates
- People: [[ronald-fisher]]

## Sources
- `raw/mml-book.pdf` §6.6.2 (Theorem 6.14, Fisher–Neyman), printed pp. 210–211 (PDF pp. 216–217).
