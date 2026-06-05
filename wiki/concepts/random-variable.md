---
title: Random Variable
type: concept
tags: [probability, foundations]
difficulty: beginner
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Random Variable

> **One-liner:** A random variable is a rule that turns the messy outcome of a random experiment into a number you actually care about.

## 🟢 Intuition (Dummy)
Flip two coins. The raw outcome is something like "heads then tails" — awkward to do math with. But maybe all you care about is *how many heads* came up: 0, 1, or 2. A random variable is the little machine that reads the raw outcome and spits out that number. It is "random" only because its *input* (the experiment) is random; the machine itself is a fixed, deterministic lookup table.

The name is famously misleading: a random variable is **neither random nor a variable — it is a function.**

## 🟡 Mechanics (Practitioner)
A **probability space** has three pieces (Kolmogorov's axioms):
- **Sample space** $\Omega$ — set of all possible outcomes (e.g. $\{hh, ht, th, tt\}$).
- **Event space** $\mathcal A$ — collection of subsets of $\Omega$ we can assign probability to (often the power set for discrete problems).
- **Probability measure** $P$ — assigns each event $A\in\mathcal A$ a number $P(A)\in[0,1]$ with $P(\Omega)=1$.

A **random variable** is a function
$$ X : \Omega \to T $$
into a **target space** $T$ (the values/states we care about). The probability transfers to $T$ via the **pre-image**:
$$ P_X(S) = P(X\in S) = P\big(X^{-1}(S)\big) = P(\{\omega\in\Omega : X(\omega)\in S\}),\quad S\subseteq T. $$
This induced law $P_X = P\circ X^{-1}$ is the [[probability-distribution]] of $X$.

**Worked example (two biased coins, $P(\$)=0.3$).** Let $X$ = number of \$ drawn, $T=\{0,1,2\}$. Then $X(\$\$)=2,\ X(\$\pounds)=X(\pounds\$)=1,\ X(\pounds\pounds)=0$, and
$$ P(X{=}2)=0.3^2=0.09,\quad P(X{=}1)=2(0.3)(0.7)=0.42,\quad P(X{=}0)=0.7^2=0.49. $$

**ML usage:** every quantity we model — a class label, a pixel value, a model parameter, a prediction — is a random variable. $T$ finite/countable ⇒ **discrete**; $T=\mathbb R$ or $\mathbb R^D$ ⇒ **continuous**.

## 🔴 Theory (Expert)
- For continuous $T$, the naive power-set event space is too large; $\mathcal A$ must be a **Borel $\sigma$-algebra** (closed under complements and countable unions/intersections) and $P$ a **measure** on it. MML deliberately side-steps the measure-theoretic construction (p. 180–181) and "pretends" continuous spaces behave like finite discrete ones.
- Measurability: $X$ must be a **measurable function**, i.e. $X^{-1}(S)\in\mathcal A$ for every measurable $S\subseteq T$, so that $P_X(S)$ is well-defined.
- For continuous $X$, $P(X=x)=0$ for every single $x$ (a set of measure zero); probability lives on *intervals/regions*, described by a [[probability-distribution|density]].
- The book flags the rampant abuse of "probability distribution" to mean the pmf, the pdf, **and** the cdf — disambiguate by context.

## Connections
- Builds on: [[vector]]
- Leads to: [[probability-distribution]], [[expected-value-and-mean]], [[bayes-theorem]]
- Used by: [[gaussian-distribution]], [[exponential-family]], essentially every probabilistic ML model
- People: —

## Sources
- `raw/mml-book.pdf` §6.1, printed pp. 172–178 (PDF pp. 178–184). Example 6.1 (biased-coin) verbatim.
