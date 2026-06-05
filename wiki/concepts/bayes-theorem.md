---
title: Bayes' Theorem
type: concept
tags: [probability, bayesian, inference]
difficulty: intermediate
sources: [mml-ch06-probability-and-distributions]
created: 2026-06-04
updated: 2026-06-04
---

# Bayes' Theorem

> **One-liner:** Bayes' theorem is the rule for updating what you believe after you see new evidence.

## 🟢 Intuition (Dummy)
You are waiting for a late friend. Before any data you guess "probably stuck in traffic" — that's your **prior** belief. Then you learn it's pouring rain. Rain makes traffic *more* likely to explain the lateness, so you update toward "traffic." Bayes' theorem is just the exact arithmetic of that update: start with a belief, see some evidence, combine them into a sharper, evidence-informed belief (the **posterior**). It is "probabilistic inverse" — it lets you reason backward from observed effects to hidden causes.

## 🟡 Mechanics (Practitioner)
Everything follows from two rules on a joint [[probability-distribution]] $p(x,y)$:

**Sum rule (marginalization):**
$$ p(x)=\sum_{y\in\mathcal Y} p(x,y)\quad\text{(discrete)},\qquad p(x)=\int_{\mathcal Y} p(x,y)\,dy\quad\text{(continuous)}. $$

**Product rule (factorization):** $\;p(x,y)=p(y\mid x)\,p(x)=p(x\mid y)\,p(y)$.

Equating the two factorizations gives **Bayes' theorem:**
$$ \underbrace{p(x\mid y)}_{\text{posterior}}=\frac{\overbrace{p(y\mid x)}^{\text{likelihood}}\;\overbrace{p(x)}^{\text{prior}}}{\underbrace{p(y)}_{\text{evidence}}},\qquad p(y)=\int p(y\mid x)\,p(x)\,dx = \mathbb E_X[p(y\mid x)]. $$

- **Prior** $p(x)$: belief about latent $x$ before data (must be nonzero wherever $x$ is plausible).
- **Likelihood** $p(y\mid x)$: how data $y$ arises given $x$ (a function of $x$, *not* a distribution in $x$).
- **Evidence / marginal likelihood** $p(y)$: normalizer; independent of $x$; key to Bayesian model selection.
- **Posterior** $p(x\mid y)$: the quantity of interest — everything we know about $x$ after seeing $y$.

**Worked example (two bags, Exercise 6.4).** Heads (prob 0.6) → bag 1 (4 mango / 2 apple); tails (0.4) → bag 2 (4 mango / 4 apple). Given a mango, probability it came from bag 2:
$$ P(B_2\mid M)=\frac{P(M\mid B_2)P(B_2)}{P(M\mid B_1)P(B_1)+P(M\mid B_2)P(B_2)}=\frac{(1/2)(0.4)}{(2/3)(0.6)+(1/2)(0.4)}=\frac{0.2}{0.6}=\tfrac13. $$

**ML usage:** the foundation of Bayesian inference — posterior over parameters, Bayesian linear regression (Ch.9), model selection via evidence (Ch.8.6), latent-variable models.

## 🔴 Theory (Expert)
- These two rules are not arbitrary: the **Cox–Jaynes theorem** shows that any system of "plausibilities" obeying consistency, honesty, and reproducibility must obey the rules of probability (up to a monotonic rescaling) — so probability *is* extended logic.
- The evidence integral $p(y)=\int p(y\mid x)p(x)\,dx$ is usually intractable (high-dimensional), motivating MCMC, variational inference, and [[conjugate-prior|conjugate priors]] that make it closed-form.
- Collapsing the posterior to a point statistic (MAP, posterior mean) discards information; keeping the *full* posterior yields decisions robust to uncertainty (e.g. data-efficient model-based RL, Deisenroth et al. 2015).
- Bayesian vs. frequentist: the formula is uncontroversial; the interpretation of $p(x)$ as "degree of belief" (Bayesian) vs. "limiting frequency" (frequentist) is the philosophical split.

## Connections
- Builds on: [[probability-distribution]], [[random-variable]]
- Leads to: [[conjugate-prior]], [[expected-value-and-mean]]
- Used by: Bayesian linear regression (Ch.9), probabilistic modelling (Ch.8.4), [[gaussian-distribution]] posteriors
- People: [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §6.3 (Eqs 6.20, 6.22, 6.23), printed pp. 183–186 (PDF pp. 189–192). Two-bag example from Exercise 6.4.
