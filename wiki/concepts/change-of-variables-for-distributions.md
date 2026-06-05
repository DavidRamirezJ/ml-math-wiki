---
title: Change of Variables for Distributions
type: concept
tags: [probability, distributions, calculus]
difficulty: advanced
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Change of Variables for Distributions

> **One-liner:** When you transform a random variable, this is the rule that tells you the shape of the new distribution.

## 🟢 Intuition (Dummy)
You know how likely each value of $X$ is. Now you squeeze or stretch $X$ through some function — say $Y=X^2$ — and ask: how likely is each value of $Y$? You can't just plug in the old density, because stretching changes how "concentrated" the probability is. Imagine spreading the same amount of butter over a bigger or smaller piece of toast: the total stays 1, but the thickness changes. The change-of-variables formula bookkeeps exactly that thickness adjustment.

## 🟡 Mechanics (Practitioner)
**Discrete case (easy).** For an invertible $U$, $Y=U(X)$:
$$ P(Y=y)=P(X=U^{-1}(y)). $$

**Distribution-function technique (continuous, from first principles).** Find the cdf of $Y$, then differentiate:
$$ F_Y(y)=P(U(X)\le y),\qquad f(y)=\frac{d}{dy}F_Y(y). $$

*Worked example (Ex 6.16).* $X$ with pdf $f(x)=3x^2$ on $[0,1]$, $Y=X^2$:
$$ F_Y(y)=P(X\le y^{1/2})=\int_0^{y^{1/2}}3t^2\,dt=y^{3/2},\qquad f(y)=\tfrac{d}{dy}y^{3/2}=\tfrac32 y^{1/2},\ \ 0\le y\le1. $$

**Change-of-variable formula (univariate, Eq 6.143).** For invertible $U$:
$$ f(y)=f_x\big(U^{-1}(y)\big)\cdot\Big|\tfrac{d}{dy}U^{-1}(y)\Big|. $$
The $|\cdot|$ factor is the local stretch — how much a unit length changes under $U$.

**Multivariate (Theorem 6.16).** For differentiable, invertible $\mathbf y=U(\mathbf x)$:
$$ f(\mathbf y)=f_x\big(U^{-1}(\mathbf y)\big)\cdot\Big|\det\tfrac{\partial}{\partial\mathbf y}U^{-1}(\mathbf y)\Big|, $$
where the absolute Jacobian determinant replaces the scalar derivative (it measures how a unit *volume* is rescaled).

*Worked example (Ex 6.17).* Standard bivariate Gaussian $\mathbf x$, linear map $\mathbf y=A\mathbf x$ with $A=[a,b;c,d]$: inverse $\mathbf x=A^{-1}\mathbf y$, $\det(\partial A^{-1}\mathbf y/\partial\mathbf y)=1/(ad-bc)$, giving
$$ f(\mathbf y)=\tfrac{1}{2\pi}\exp\!\big(-\tfrac12\mathbf y^\top A^{-\top}A^{-1}\mathbf y\big)\,|ad-bc|^{-1}, $$
which is a Gaussian with covariance $\Sigma=AA^\top$.

**ML usage:** the mathematical engine behind **normalizing flows** (transform a simple base density through invertible neural maps to model complex data), the reparameterization trick, and sampling via the inverse cdf.

## 🔴 Theory (Expert)
- The univariate formula derives from the calculus **substitution rule** $\int f(g(x))g'(x)\,dx=\int f(u)\,du$, which itself rests on the chain rule plus the fundamental theorem of calculus.
- The Jacobian determinant arises because differentials (volume cubes) are mapped to parallelepipeds by the linear approximation of $U$ — the same determinant-as-volume-scaling idea from Ch.4.
- **Probability integral transform (Theorem 6.15):** if $F_X$ is strictly monotonic, then $Y:=F_X(X)$ is **uniform** on $[0,1]$. Inverting: feeding uniform samples through $F_X^{-1}$ generates samples from any target distribution (inverse-transform sampling), and the result underlies copulas and goodness-of-fit testing.
- Continuous transformations need the Jacobian factor that the discrete case lacks, precisely because $P(Y=y)=0$ and only the *density* (not a point probability) transforms.

## Connections
- Builds on: [[probability-distribution]], [[matrix]], the Jacobian / determinant (Ch.4–5)
- Leads to: normalizing flows, inverse-transform sampling
- Used by: [[gaussian-distribution]] (affine transforms), generative models
- People: —

## Sources
- `raw/mml-book.pdf` §6.7 (Thms 6.15–6.16, Eqs 6.125–6.151, Ex 6.16–6.17), printed pp. 214–221 (PDF pp. 220–227).
