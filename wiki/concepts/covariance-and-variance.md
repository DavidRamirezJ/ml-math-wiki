---
title: Covariance and Variance
type: concept
tags: [probability, statistics, summary-statistics, linear-algebra]
difficulty: intermediate
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Covariance and Variance

> **One-liner:** Variance measures how spread out a random quantity is; covariance measures whether two quantities move together.

## 🟢 Intuition (Dummy)
**Variance** is "how jumpy is this number?" A thermostat-controlled room has low temperature variance; the weather outside has high variance. **Covariance** asks about *pairs*: when one thing goes up, does the other tend to go up too? Height and weight covary positively (taller people are usually heavier); hours of TV and exam scores covary negatively. **Correlation** is just covariance rescaled to land between $-1$ and $+1$ so you can compare across different units.

## 🟡 Mechanics (Practitioner)
**Covariance** of two univariate random variables (Def 6.5):
$$ \mathrm{Cov}[x,y]=\mathbb E\big[(x-\mathbb E[x])(y-\mathbb E[y])\big]=\mathbb E[xy]-\mathbb E[x]\,\mathbb E[y]. $$

**Variance** = self-covariance (Def 6.7); **standard deviation** $\sigma(x)=\sqrt{\mathbb V[x]}$. The **raw-score formula** (one-pass, Eq 6.44):
$$ \mathbb V[x]=\mathbb E[x^2]-(\mathbb E[x])^2\quad\text{("mean of the square minus square of the mean").} $$

**Covariance matrix** of a vector random variable $\mathbf x\in\mathbb R^D$ with mean $\boldsymbol\mu$:
$$ \mathbb V[\mathbf x]=\mathbb E[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top]=\mathbb E[\mathbf x\mathbf x^\top]-\boldsymbol\mu\boldsymbol\mu^\top\in\mathbb R^{D\times D}. $$
It is **symmetric and positive semidefinite**; diagonal = per-coordinate variances, off-diagonal = cross-covariances.

**Correlation** (Def 6.8): $\;\mathrm{corr}[x,y]=\dfrac{\mathrm{Cov}[x,y]}{\sqrt{\mathbb V[x]\mathbb V[y]}}\in[-1,1].$

**Affine transformation** (Eqs 6.50–6.51): for $\mathbf y=A\mathbf x+\mathbf b$,
$$ \mathbb E[\mathbf y]=A\boldsymbol\mu+\mathbf b,\qquad \mathbb V[\mathbf y]=A\,\Sigma\,A^\top. $$

**Empirical covariance** (Def 6.9): $\;\Sigma=\frac1N\sum_{n=1}^N(\mathbf x_n-\bar{\mathbf x})(\mathbf x_n-\bar{\mathbf x})^\top$ (biased; unbiased uses $N-1$).

**Worked example.** $X$ with $\mathbb E[x]=0,\ \mathbb E[x^3]=0$, and $Y=x^2$: $\mathrm{Cov}[x,y]=\mathbb E[x^3]-\mathbb E[x]\mathbb E[x^2]=0$ even though $Y$ fully depends on $X$ — covariance only sees *linear* dependence.

**ML usage:** the covariance matrix is the shape parameter of the [[gaussian-distribution|Gaussian]], the object PCA diagonalizes (Ch.10), and the noise model in linear regression (Ch.9); correlation drives feature selection and whitening.

## 🔴 Theory (Expert)
- **Sums:** $\mathbb V[\mathbf x+\mathbf y]=\mathbb V[\mathbf x]+\mathbb V[\mathbf y]+\mathrm{Cov}[\mathbf x,\mathbf y]+\mathrm{Cov}[\mathbf y,\mathbf x]$; if uncorrelated this collapses to $\mathbb V[\mathbf x]+\mathbb V[\mathbf y]$ (a Pythagorean identity — see [[inner-product]] of random variables).
- **Three expressions for variance** (§6.4.3): standard $\mathbb E[(x-\mu)^2]$, raw-score (numerically unstable when the two large terms nearly cancel), and the pairwise form $\frac{1}{N^2}\sum_{i,j}(x_i-x_j)^2 = 2\big(\frac1N\sum x_i^2-(\frac1N\sum x_i)^2\big)$ — equating "spread about the mean" with "average pairwise distance".
- **Law of total variance:** $\mathbb V[x]=\mathbb E_Y[\mathbb V[x\mid y]]+\mathbb V_Y[\mathbb E[x\mid y]]$.
- Covariance defines an [[inner-product|inner product]] $\langle X,Y\rangle:=\mathrm{Cov}[x,y]$ on zero-mean random variables: symmetric, bilinear, positive definite. Then $\|X\|=\sigma(x)$ and $\cos\theta=\mathrm{corr}[x,y]$ — correlation is literally a cosine, "uncorrelated" means "orthogonal".
- MML assumes covariance matrices are positive *definite* for intuition; degenerate (low-rank, positive semidefinite) cases are skipped.

## Connections
- Builds on: [[expected-value-and-mean]], [[matrix]], [[inner-product]]
- Leads to: [[gaussian-distribution]], [[statistical-independence]]
- Used by: PCA / dimensionality reduction (Ch.10), linear regression noise (Ch.9), whitening
- People: —

## Sources
- `raw/mml-book.pdf` §6.4.1–6.4.4 (Defs 6.5–6.9, Eqs 6.44–6.51), printed pp. 189–194 (PDF pp. 195–200).
