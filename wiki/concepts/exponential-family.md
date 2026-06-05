---
title: Exponential Family
type: concept
tags: [probability, distributions, statistics]
difficulty: advanced
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Exponential Family

> **One-liner:** The exponential family is the single template that most "named" distributions fit into — and the reason they're so easy to compute with.

## 🟢 Intuition (Dummy)
Statistics textbooks throw a zoo of distributions at you — Gaussian, Bernoulli, Beta, Poisson — and it feels like memorizing unrelated animals. The exponential family is the realization that almost all of them are built from one master recipe. Once you write a distribution in this common form, a whole toolbox of shortcuts unlocks: you only ever need to track a fixed handful of summary numbers from your data, no matter how much data you collect, and finding the "best fit" becomes a well-behaved (concave) optimization.

## 🟡 Mechanics (Practitioner)
A distribution is in the **exponential family** if it can be written (Eq 6.107) as
$$ p(x\mid\boldsymbol\theta)=h(x)\,\exp\big(\langle\boldsymbol\theta,\phi(x)\rangle-A(\boldsymbol\theta)\big), $$
where
- $\phi(x)$ = vector of **[[sufficient-statistics|sufficient statistics]]**,
- $\boldsymbol\theta$ = **natural (canonical) parameters**,
- $A(\boldsymbol\theta)$ = **log-partition function** (normalizer ensuring it integrates to 1),
- $h(x)$ = base measure (can be folded into $\phi$).

Stripped down: $p(x\mid\boldsymbol\theta)\propto\exp(\boldsymbol\theta^\top\phi(x))$.

**Example — Gaussian (Ex 6.13).** With $\phi(x)=[x,\,x^2]^\top$ and $\boldsymbol\theta=\big[\tfrac{\mu}{\sigma^2},\,-\tfrac{1}{2\sigma^2}\big]^\top$:
$$ p(x\mid\boldsymbol\theta)\propto\exp\big(\theta_1 x+\theta_2 x^2\big)\propto\exp\!\Big(-\tfrac{1}{2\sigma^2}(x-\mu)^2\Big). $$

**Example — Bernoulli (Ex 6.14).** $p(x\mid\mu)=\mu^x(1-\mu)^{1-x}$ rewrites as
$$ p(x\mid\mu)=\exp\!\Big[x\log\tfrac{\mu}{1-\mu}+\log(1-\mu)\Big], $$
so $\phi(x)=x$, $\theta=\log\frac{\mu}{1-\mu}$ (the **logit**), $A(\theta)=\log(1+e^{\theta})$, and inverting gives
$$ \mu=\frac{1}{1+e^{-\theta}}\quad\text{(the sigmoid / logistic function).} $$
This is exactly the link between log-odds and probability used in **logistic regression** and as a neural-net activation.

**ML usage:** members include Gaussian, Bernoulli, Binomial, Beta, Poisson, categorical/multinomial, Gamma, Dirichlet. Maximum-likelihood fitting reduces to matching empirical sufficient statistics; the log-likelihood is concave so optimization (Ch.7) is reliable.

## 🔴 Theory (Expert)
- **Pitman–Koopman–Darmois theorem** (independently 1935–36): under repeated i.i.d. sampling, the exponential families are the *only* families admitting **finite-dimensional sufficient statistics** whose dimension does not grow with the sample size. This is the deep reason to care.
- $A(\boldsymbol\theta)$ is convex; its derivatives are cumulants: $\nabla A(\boldsymbol\theta)=\mathbb E[\phi(x)]$, $\nabla^2 A(\boldsymbol\theta)=\mathrm{Cov}[\phi(x)]$ (positive semidefinite ⇒ concave log-likelihood).
- Every member has a **conjugate prior** also in the exponential family (Eq 6.120): $p(\boldsymbol\theta\mid\boldsymbol\gamma)=h_c(\boldsymbol\theta)\exp(\langle[\gamma_1;\gamma_2],[\boldsymbol\theta;-A(\boldsymbol\theta)]\rangle-A_c(\boldsymbol\gamma))$, with sufficient statistics $[\boldsymbol\theta;-A(\boldsymbol\theta)]$. Deriving the canonical conjugate of the Bernoulli (Ex 6.15) *recovers the Beta distribution* — explaining the [[conjugate-prior|Beta–Bernoulli]] conjugacy from first principles.
- The form is precisely the $g_\theta(\phi(x))$ factor in the Fisher–Neyman factorization (Thm 6.14).

## Connections
- Builds on: [[probability-distribution]], [[sufficient-statistics]], [[inner-product]]
- Leads to: [[conjugate-prior]], maximum likelihood / convex optimization (Ch.7)
- Used by: [[gaussian-distribution]], logistic regression, generalized linear models
- People: [[ronald-fisher]]

## Sources
- `raw/mml-book.pdf` §6.6.3 (Eqs 6.107–6.124, Ex 6.13–6.15), printed pp. 211–214 (PDF pp. 217–220).
