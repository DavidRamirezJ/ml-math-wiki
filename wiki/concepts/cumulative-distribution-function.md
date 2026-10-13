---
title: Cumulative Distribution Function
type: concept
tags: [probability, distributions, gaussian, activation-function]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, mml-ch06-probability-and-distributions]
created: 2026-10-13
updated: 2026-10-13
---

# Cumulative Distribution Function

> **One-liner:** The CDF answers one simple question: "what's the chance my random number comes out at or below this value?"

## 🟢 Intuition (Dummy)
Imagine lining up everyone in a school by height. Pick any height — say 160 cm — and ask: "what fraction
of the students are 160 cm or shorter?" That fraction is the **cumulative distribution function** at
160. Slide your chosen height upward and the fraction can only grow: from 0 (nobody is shorter than the
shortest) to 1 (everybody is at most as tall as the tallest).

For the bell curve ([[gaussian-distribution]]), the CDF is a smooth S-shaped curve. That S-curve has a
special name, $\Phi$ ("phi"), and it shows up in a surprising place: the [[gelu]] activation function that
GPT-2 uses multiplies every input by $\Phi$ of that input.

## 🟡 Mechanics (Practitioner)
**Definition** (standard; MML Def 6.2 gives the multivariate form — see [[probability-distribution]]):
$$F_X(x)=P(X\le x)$$
- $X$ = a [[random-variable]] (a number that comes out randomly),
- $x$ = a fixed threshold you choose,
- $P(X\le x)$ = the probability that $X$ lands at or below $x$,
- $F_X$ = the CDF of $X$ — a function that turns any threshold into a probability.

For a continuous variable with density (pdf) $p$:
$$F_X(x)=\int_{-\infty}^{x}p(t)\,dt$$
Read it as: "add up (integrate) all the density from minus infinity up to $x$" — the area under the pdf to
the left of $x$.

**The standard normal CDF $\Phi$** (the one GELU uses):
$$\Phi(x)=\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt=\tfrac12\Big[1+\operatorname{erf}\!\big(\tfrac{x}{\sqrt2}\big)\Big]$$
- $\frac{1}{\sqrt{2\pi}}e^{-t^2/2}$ = the bell-curve density $\varphi(t)$ of $\mathcal N(0,1)$ (mean 0, variance 1),
- $\operatorname{erf}$ = the "error function," a standard special function — this is where the "E" in GELU (Gaussian **Error** Linear Unit) comes from.

This integral has no elementary closed form (it needs the special function $\operatorname{erf}$). The book
says it is "common to implement a computationally cheaper approximation," found "via curve fitting," and that
the original GPT-2 was trained with it (book §4.3, p. 105). Inside GELU that approximation amounts to
$\Phi(x)\approx\tfrac12\big(1+\tanh[\sqrt{2/\pi}\,(x+0.044715x^3)]\big)$.

**Handy values** (computed):

| $x$ | $-1$ | $0$ | $1$ | $1.96$ |
|---|---|---|---|---|
| $\Phi(x)$ | 0.1587 | 0.5 | 0.8413 | 0.9750 |

So $\Phi(0)=0.5$ (half the bell is left of the center), and about 68% of the mass lies in $[-1,1]$
($0.8413-0.1587$).

**In LLMs — the book's use (§4.3, p. 105).** The exact GELU is defined as
$$\mathrm{GELU}(x)=x\cdot\Phi(x),$$
"where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian distribution." So
$\Phi$ works as a **soft gate**: at $x=1$ the input is kept at 84%, giving $\mathrm{GELU}(1)\approx0.8413$;
at $x=-1$ only 16% is kept, giving $\mathrm{GELU}(-1)\approx-0.1587$.

```python
import torch
x = torch.tensor([-1.0, 0.0, 1.0])
Phi = 0.5 * (1 + torch.erf(x / 2**0.5))   # standard normal CDF
print(Phi)        # tensor([0.1587, 0.5000, 0.8413])
print(x * Phi)    # exact GELU: tensor([-0.1587, 0.0000, 0.8413])
```
(Sketch, not from the book; the book implements only the tanh approximation.)

## 🔴 Theory (Expert)
- **Characterization.** $F:\mathbb R\to[0,1]$ is a CDF of some random variable iff it is
  (i) non-decreasing ([[monotonic-function]]), (ii) right-continuous, and (iii) $\lim_{x\to-\infty}F(x)=0$,
  $\lim_{x\to+\infty}F(x)=1$. The CDF exists for *every* real random variable — discrete, continuous, or
  mixed — which makes it more fundamental than the pdf.
- **Density as derivative.** Where $F$ is differentiable, $F'(x)=p(x)$. For the standard normal,
  $\Phi'=\varphi$. This is exactly what produces GELU's derivative $\Phi(x)+x\varphi(x)$ ([[derivative]]).
- **Location–scale.** If $X\sim\mathcal N(\mu,\sigma^2)$ then $F_X(x)=\Phi\big(\tfrac{x-\mu}{\sigma}\big)$ —
  every Gaussian CDF is a shifted, stretched $\Phi$ ([[standardization-z-score]]).
- **Symmetry and tails.** $\Phi(-x)=1-\Phi(x)$; Mills-ratio bounds give $1-\Phi(x)\approx\varphi(x)/x$ for
  large $x$, so the gap $1-\Phi(x)$ shrinks faster than any power of $x$.
- **Inverse CDF sampling.** If $U\sim\mathrm{Uniform}(0,1)$ then $F^{-1}(U)$ has CDF $F$ (with $F^{-1}$ the
  generalized inverse). This underlies sampling and the "distribution-function technique" in
  [[change-of-variables-for-distributions]].
- **Sigmoids as CDFs.** The logistic [[sigmoid-function]] is the CDF of the logistic distribution; $\Phi$
  is the CDF of the normal. That's why "probit" vs. "logit" models, and GELU vs. Swish-style gates
  ([[swiglu]]), are near-twins: $\Phi(x)\approx\sigma(1.702x)$.

## Connections
- Builds on: [[random-variable]], [[probability-distribution]], [[gaussian-distribution]]
- Leads to: [[gelu]], [[change-of-variables-for-distributions]]
- Used by: [[gelu]] (the gate $\Phi(x)$), [[activation-function]]
- Related: [[sigmoid-function]], [[standardization-z-score]], [[monotonic-function]], [[derivative]]
- People: [[carl-friedrich-gauss]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book p. 105 ($\mathrm{GELU}(x)=x\cdot\Phi(x)$ with $\Phi$ the standard Gaussian CDF; tanh approximation).
- `raw/mml-book.pdf` §6.2 (Def 6.2, cdf), printed pp. 178–183 (PDF pp. 184–189).
