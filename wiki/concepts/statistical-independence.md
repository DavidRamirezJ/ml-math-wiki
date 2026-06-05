---
title: Statistical Independence
type: concept
tags: [probability, statistics]
difficulty: intermediate
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Statistical Independence

> **One-liner:** Two things are independent if learning one tells you nothing new about the other.

## 🟢 Intuition (Dummy)
Flip a coin in London and roll a die in Tokyo. Knowing the coin came up heads gives you zero hint about the die — those events are **independent**. Now consider the cloudiness of the sky and whether it rains: knowing it's cloudy *does* change your guess about rain, so those are **dependent**. **Conditional** independence is subtler: two things might look related, but once you know a third fact, the link disappears (ice-cream sales and drownings are linked — until you account for "it's summer," which explains both).

## 🟡 Mechanics (Practitioner)
**Independence** (Def 6.10): $X,Y$ are statistically independent iff
$$ p(x,y)=p(x)\,p(y). $$
Equivalent consequences when independent:
$$ p(y\mid x)=p(y),\quad p(x\mid y)=p(x),\quad \mathbb V[x+y]=\mathbb V[x]+\mathbb V[y],\quad \mathrm{Cov}[x,y]=0. $$

**Crucial caveat:** $\mathrm{Cov}[x,y]=0$ does **not** imply independence — covariance only detects *linear* dependence. Example: $X$ zero-mean with $\mathbb E[x^3]=0$ and $Y=X^2$ are dependent yet have zero covariance.

**Conditional independence** (Def 6.11): $X\perp\!\!\!\perp Y\mid Z$ iff
$$ p(x,y\mid z)=p(x\mid z)\,p(y\mid z)\quad\text{for all } z, $$
equivalently $p(x\mid y,z)=p(x\mid z)$ — "given $z$, knowing $y$ adds nothing about $x$."

**i.i.d.:** data $X_1,\dots,X_N$ are **independent and identically distributed** — all mutually independent and drawn from the same distribution. This is the standard modelling assumption that makes likelihoods factorize: $p(x_1,\dots,x_N\mid\theta)=\prod_n p(x_n\mid\theta)$.

**Worked example.** Two coin draws *with replacement* are independent, so $P(\$\$)=P(\$)P(\$)=0.3\times0.3=0.09$ (Example 6.1).

**ML usage:** the i.i.d. assumption underlies almost every loss/likelihood; conditional independence is the language of **graphical models** (Ch.8.5) — edges encode dependence, missing edges encode conditional independence.

## 🔴 Theory (Expert)
- For $>2$ variables, "independent" means **mutually** independent: *every* subset factorizes (strictly stronger than pairwise independence).
- Independence is the special case $X\perp\!\!\!\perp Y\mid\emptyset$ of conditional independence.
- Zero covariance ⇏ independence, but for *jointly Gaussian* variables it *does* (a special, much-used property of the [[gaussian-distribution]]).
- Conditional independence structures factorize huge joint distributions into manageable products — the entire computational payoff of Bayesian networks and Markov random fields.

## Connections
- Builds on: [[probability-distribution]], [[covariance-and-variance]]
- Leads to: [[bayes-theorem]], graphical models (Ch.8.5)
- Used by: i.i.d. likelihoods, [[gaussian-distribution]] (sums of independent Gaussians), Naive Bayes
- People: —

## Sources
- `raw/mml-book.pdf` §6.4.5 (Defs 6.10–6.11, Eqs 6.53–6.57), printed pp. 194–195 (PDF pp. 200–201).
