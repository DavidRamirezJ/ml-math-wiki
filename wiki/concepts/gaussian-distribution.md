---
title: Gaussian Distribution
type: concept
tags: [probability, distributions, gaussian]
difficulty: intermediate
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Gaussian Distribution

> **One-liner:** The Gaussian (bell curve) is the distribution that shows up whenever many small random effects add together.

## 🟢 Intuition (Dummy)
Measure the heights of thousands of people and plot a histogram: you get a symmetric hump — common near the average, rare at the extremes. That's the **Gaussian** (a.k.a. **normal**) distribution, the famous bell curve. It appears everywhere because of a deep fact: when you add up lots of independent random nudges, their sum is bell-shaped no matter what the individual nudges looked like (the central limit theorem). It's described by just two numbers: where the hump sits (**mean**) and how wide it is (**variance**).

## 🟡 Mechanics (Practitioner)
**Univariate** density (Eq 6.62):
$$ p(x\mid\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big). $$

**Multivariate** (Eq 6.63), mean vector $\boldsymbol\mu\in\mathbb R^D$, covariance matrix $\Sigma$:
$$ p(\mathbf x\mid\boldsymbol\mu,\Sigma)=(2\pi)^{-D/2}\,|\Sigma|^{-1/2}\exp\!\Big(-\tfrac12(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)\Big). $$
Notation $\mathbf x\sim\mathcal N(\boldsymbol\mu,\Sigma)$. The **standard normal** is $\mathcal N(\mathbf 0,I)$.

**Why it's the workhorse — it's closed under everything:**
- **Marginals are Gaussian** (Eq 6.68): from joint $\mathcal N\big([\boldsymbol\mu_x;\boldsymbol\mu_y],\,[\Sigma_{xx},\Sigma_{xy};\Sigma_{yx},\Sigma_{yy}]\big)$, $\;p(\mathbf x)=\mathcal N(\boldsymbol\mu_x,\Sigma_{xx})$ — just read off the block.
- **Conditionals are Gaussian** (Eqs 6.66–6.67):
$$ \boldsymbol\mu_{x\mid y}=\boldsymbol\mu_x+\Sigma_{xy}\Sigma_{yy}^{-1}(\mathbf y-\boldsymbol\mu_y),\qquad \Sigma_{x\mid y}=\Sigma_{xx}-\Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}. $$
- **Affine maps stay Gaussian:** $\mathbf y=A\mathbf x+\mathbf b\Rightarrow \mathbf y\sim\mathcal N(A\boldsymbol\mu+\mathbf b,\,A\Sigma A^\top)$.
- **Sums of independent Gaussians** (Eq 6.78): $\mathcal N(\boldsymbol\mu_x,\Sigma_x)+\mathcal N(\boldsymbol\mu_y,\Sigma_y)=\mathcal N(\boldsymbol\mu_x+\boldsymbol\mu_y,\Sigma_x+\Sigma_y)$.
- **Product of two densities** (Eqs 6.74–6.76) is a scaled Gaussian with $C=(A^{-1}+B^{-1})^{-1}$, $\mathbf c=C(A^{-1}\mathbf a+B^{-1}\mathbf b)$.

**Worked example (conditioning, Ex 6.6).** $p(x_1,x_2)=\mathcal N\big([0;2],[0.3,-1;-1,5]\big)$. Conditioning on $x_2=-1$:
$$ \mu_{x_1\mid x_2=-1}=0+(-1)(0.2)(-1-2)=0.6,\quad \sigma^2_{x_1\mid x_2=-1}=0.3-(-1)(0.2)(-1)=0.1. $$
(here $\Sigma_{yy}^{-1}=1/5=0.2$), so $p(x_1\mid x_2=-1)=\mathcal N(0.6,0.1)$.

**Sampling** $\mathcal N(\boldsymbol\mu,\Sigma)$: take $\mathbf x\sim\mathcal N(\mathbf 0,I)$, compute the Cholesky factor $\Sigma=AA^\top$, return $\mathbf y=A\mathbf x+\boldsymbol\mu$.

**ML usage:** likelihood & prior in linear regression (Ch.9), mixtures for density estimation (Ch.11), Gaussian processes, variational inference, Kalman filtering, the noise model in countless models.

## 🔴 Theory (Expert)
- **Central limit theorem:** normalized sums of i.i.d. finite-variance variables converge to a Gaussian — the structural reason for its ubiquity.
- It is a maximum-entropy distribution given fixed mean and covariance, and a member of the [[exponential-family]] with sufficient statistics $\phi(x)=[x,x^2]^\top$ and natural parameters $\theta=[\mu/\sigma^2,\,-1/(2\sigma^2)]^\top$.
- The conditional formulas are exactly what the **Kalman filter** computes (Gaussian conditioning on a joint), and what **Gaussian processes** use to condition a distribution over functions on observed data.
- **Reverse/linear-Gaussian inference** (Eqs 6.89–6.91): if $p(\mathbf y)=\mathcal N(A\mathbf x,\Sigma)$ with $A$ full rank, the pseudo-inverse gives $p(\mathbf x)=\mathcal N\big((A^\top A)^{-1}A^\top\mathbf y,\,(A^\top A)^{-1}A^\top\Sigma A(A^\top A)^{-1}\big)$.
- Because it is fully specified by mean and covariance, transformed Gaussians are obtained by transforming just those two — no [[change-of-variables-for-distributions|Jacobian gymnastics]] needed.

## Connections
- Builds on: [[covariance-and-variance]], [[expected-value-and-mean]], [[matrix]]
- Leads to: [[exponential-family]], [[conjugate-prior]], [[change-of-variables-for-distributions]]
- Used by: linear regression (Ch.9), PCA/PPCA (Ch.10), density estimation (Ch.11), Kalman filter, Gaussian processes
- People: [[carl-friedrich-gauss]]

## Sources
- `raw/mml-book.pdf` §6.5 (Eqs 6.62–6.91, Ex 6.6), printed pp. 197–204 (PDF pp. 203–210).
