---
title: "MML Ch.6 — Probability and Distributions"
type: source
tags: [probability, statistics, distributions, bayesian]
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# MML Ch.6 — Probability and Distributions

- **Raw file:** `raw/mml-book.pdf`, Chapter 6, printed pp. 172–224 (PDF pp. 178–230)
- **Type:** book chapter
- **Ingested:** 2026-06-04

## Section coverage

| § | Title | Printed pp. | Wiki page(s) |
|---|---|---|---|
| 6.1 | Construction of a Probability Space | 172–178 | [[random-variable]] |
| 6.2 | Discrete and Continuous Probabilities | 178–183 | [[probability-distribution]] |
| 6.3 | Sum Rule, Product Rule, and Bayes' Theorem | 183–186 | [[bayes-theorem]] |
| 6.4 | Summary Statistics and Independence | 186–197 | [[expected-value-and-mean]], [[covariance-and-variance]], [[statistical-independence]] |
| 6.5 | Gaussian Distribution | 197–204 | [[gaussian-distribution]] |
| 6.6 | Conjugacy and the Exponential Family | 205–214 | [[exponential-family]], [[conjugate-prior]] |
| 6.7 | Change of Variables / Inverse Transform | 214–221 | [[change-of-variables-for-distributions]] |
| 6.8 | Further Reading | 221–224 | — |

## TL;DR (5 bullets)

- A **probability space** $(\Omega, \mathcal A, P)$ plus a **random variable** $X:\Omega\to T$ (a *function*, not a variable) is the machinery that turns messy outcomes into numbers we can compute with; the [[probability-distribution]] is the law $P_X = P\circ X^{-1}$ that the random variable carries to the target space.
- Two rules generate all of probabilistic inference: the **sum rule** (marginalization) and the **product rule** (factorization); chaining them gives [[bayes-theorem]], which inverts a likelihood into a posterior.
- A distribution is summarized by its [[expected-value-and-mean|mean]] and [[covariance-and-variance|(co)variance]]; the covariance is a genuine [[inner-product]] on zero-mean random variables, so correlation is literally the cosine of an angle and "uncorrelated" means "orthogonal".
- The [[gaussian-distribution]] is the workhorse: closed under marginalization, conditioning, affine transformation, and products of densities — which is exactly why linear-Gaussian models (Kalman filter, GPs, linear regression) are tractable.
- The [[exponential-family]] unifies the "named" distributions; it is the unique class with finite-dimensional [[sufficient-statistics]], it gives [[conjugate-prior|conjugate priors]] for cheap closed-form Bayesian updates (e.g. Beta–Binomial), and its log-likelihood is concave.

## Key concepts introduced

- [[random-variable]] — function from outcomes to a target space; the bridge from probability space to numbers.
- [[probability-distribution]] — pmf/pdf/cdf; discrete vs. continuous; joint/marginal/conditional.
- [[bayes-theorem]] — posterior ∝ likelihood × prior; sum & product rules.
- [[expected-value-and-mean]] — the expectation operator, mean, median, mode.
- [[covariance-and-variance]] — variance, covariance matrix, correlation, empirical estimates.
- [[statistical-independence]] — $p(x,y)=p(x)p(y)$; conditional independence; i.i.d.
- [[inner-product|inner product of random variables]] — covariance as an inner product, correlation as cosine (extends [[inner-product]] from Ch.3).
- [[gaussian-distribution]] — univariate & multivariate normal; marginals/conditionals/products.
- [[exponential-family]] — natural parameters, sufficient statistics, log-partition function.
- [[conjugate-prior]] — prior whose posterior keeps its form; Beta–Binomial, Beta–Bernoulli.
- [[sufficient-statistics]] — Fisher–Neyman factorization; all the info a sample carries about $\theta$.
- [[change-of-variables-for-distributions]] — distribution-function technique and the Jacobian recipe.

## Notable results / theorems

- **Sum rule (6.20)** / **product rule (6.22)** / **Bayes' theorem (6.23)**.
- **Affine transformation of mean/covariance (6.50)–(6.51):** $\mathbb E[Ax+b]=A\mu+b$, $\mathbb V[Ax+b]=A\Sigma A^\top$.
- **Law of total variance (6.85c remark):** $\mathbb V[x]=\mathbb E_Y[\mathbb V[x\mid y]]+\mathbb V_Y[\mathbb E[x\mid y]]$.
- **Gaussian conditioning (6.65)–(6.67)** and **marginalization (6.68)** stay Gaussian.
- **Product of two Gaussian densities (6.74)–(6.76)** is a scaled Gaussian.
- **Theorem 6.14 (Fisher–Neyman factorization):** $p(x\mid\theta)=h(x)\,g_\theta(\phi(x))$ iff $\phi$ is sufficient.
- **Theorem 6.15 (probability integral transform):** $Y:=F_X(X)$ is uniform.
- **Theorem 6.16 (multivariate change of variables):** $f(y)=f_x(U^{-1}(y))\,\lvert\det(\partial U^{-1}/\partial y)\rvert$.

## What this changes in the wiki

- First probability/statistics chapter ingested — establishes the whole `probability` tag cluster. No contradictions with existing linear-algebra pages.
- Reuses and extends Ch.3's [[inner-product]] (covariance is an inner product) and Ch.4's Cholesky decomposition (used to sample from $\mathcal N(\mu,\Sigma)$) — link those rather than redefining.
- Adds person pages for [[thomas-bayes]] and [[ronald-fisher]]; [[carl-friedrich-gauss]] already exists (Gaussian distribution) — just linked.
