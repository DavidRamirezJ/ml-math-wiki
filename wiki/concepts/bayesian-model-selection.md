---
title: Bayesian Model Selection
type: concept
tags: [machine-learning, model-selection, bayesian, probabilistic-modeling]
difficulty: advanced
sources: [mml-ch08-when-models-meet-data]
created: 2026-06-04
updated: 2026-06-04
---

# Bayesian Model Selection

> **One-liner:** Choose between competing models by how well each one *predicted the data you actually got* — a rule that automatically prefers the simplest model that still works.

## 🟢 Intuition (Dummy)
You have two theories. A "simple" theory makes a narrow, confident prediction; a "flexible" theory
can explain almost anything, so its prediction is spread thin across many possibilities. Now the data
arrives. If it lands where the simple theory pointed, the simple theory wins — it bet big and was
right. The flexible theory hedged, so even when the data fits, it only assigned a little probability
to it. This is **Occam's razor**, made automatic: the model that explains the data without being
needlessly flexible comes out on top, with no extra "penalize complexity" rule bolted on.

## 🟡 Mechanics (Practitioner)
We want the best model class, not just the best parameters. More flexible classes (higher-degree
polynomials, bigger nets) can fit more datasets but risk [[overfitting-and-underfitting|overfitting]];
[[cross-validation]] is one answer, the **marginal likelihood** is the probabilistic one (MML §8.6).

The central quantity is the **model evidence / marginal likelihood** — the probability the model $M_k$
assigns to the data, with parameters integrated out (Eq. 8.44):
$$ p(\mathcal D\mid M_k)=\int p(\mathcal D\mid\boldsymbol\theta_k)\,p(\boldsymbol\theta_k\mid M_k)\,d\boldsymbol\theta_k . $$
Treat model selection as Bayesian inference over models (hierarchical generative process, Eqs.
8.40–8.42: $M_k\sim p(M)$, $\boldsymbol\theta_k\sim p(\boldsymbol\theta\mid M_k)$,
$\mathcal D\sim p(\mathcal D\mid\boldsymbol\theta_k)$). The model posterior (Eq. 8.43) and MAP model are
$$ p(M_k\mid\mathcal D)\ \propto\ p(M_k)\,p(\mathcal D\mid M_k), \qquad
M^*=\arg\max_{M_k} p(M_k\mid\mathcal D). $$
With a uniform prior over models, this just picks the model with the **largest evidence**.

**Bayes factor (§8.6.3).** To compare two models, the **posterior odds** factor into prior odds times
the **Bayes factor** (Eq. 8.46):
$$ \underbrace{\frac{p(M_1\mid\mathcal D)}{p(M_2\mid\mathcal D)}}_{\text{posterior odds}}
=\underbrace{\frac{p(M_1)}{p(M_2)}}_{\text{prior odds}}\;\underbrace{\frac{p(\mathcal D\mid M_1)}{p(\mathcal D\mid M_2)}}_{\text{Bayes factor}} . $$
With equal priors, choose $M_1$ if the Bayes factor exceeds 1.

**Worked intuition (Fig. 8.14).** Each model spreads a fixed total probability of 1 over all possible
datasets. A simple model concentrates it on few datasets (tall, narrow); a complex one spreads it
thin (low, wide). For data in the "simple" region, the simple model assigns higher evidence and wins.

**ML usage.** Selecting polynomial degree, mixture components, kernel type, latent dimension, network
architecture. When the evidence integral is closed-form (conjugate priors), use it directly (Ch. 9);
otherwise approximate (numerical integration, Monte Carlo) or fall back to cheap proxies — **AIC**
$=\log p(\mathbf x\mid\boldsymbol\theta)-M$ and **BIC** $\approx \log p(\mathbf x\mid\boldsymbol\theta)-\tfrac12 M\log N$ (Eqs. 8.48–8.49, $M$ = #params, $N$ = #data).

## 🔴 Theory (Expert)
- **Why evidence beats likelihood.** The plain likelihood always rewards more parameters (it overfits);
  the marginal likelihood does **not**, because the parameters are integrated out — it embodies an
  *automatic* trade-off between fit and complexity (Occam's razor) with no explicit penalty term.
- **Jeffreys–Lindley paradox:** with a diffuse (very spread-out) prior, the evidence of a complex
  model is dragged down, so the Bayes factor can over-favor the simpler model. Evidence is sensitive
  to the prior, unlike the likelihood.
- **Occam's razor is about function complexity, not parameter count** (Rasmussen & Ghahramani 2001):
  it still operates for nonparametric models with infinitely many parameters (Gaussian processes).
- The bottleneck is the **intractable evidence integral** (8.44), shared with [[bayesian-inference]];
  conjugate priors give closed forms, otherwise approximate.

## Connections
- Builds on: [[bayesian-inference]], [[bayes-theorem]], [[overfitting-and-underfitting]]
- Leads to: [[cross-validation]] (its resampling counterpart), [[conjugate-prior]]
- Used by: [[linear-regression]] (evidence for degree selection), [[gaussian-mixture-model]]
- People: [[william-of-occam]], [[thomas-bayes]]

## Sources
- `raw/mml-book.pdf` §8.6 (Eqs. 8.39–8.49, Figs. 8.13–8.15), printed pp. 283–288 (PDF pp. 289–294).
