---
title: Arthur P. Dempster (1929–)
type: person
tags: [statistics, em-algorithm, people]
sources: [mml-ch11-density-estimation-gaussian-mixture-models]
created: 2026-06-04
updated: 2026-06-04
---

# Arthur P. Dempster (1929–)

> **Why they matter to ML:** Co-author of the 1977 paper that named and unified the **EM algorithm**, the workhorse for fitting latent-variable and mixture models across machine learning.

## Life (brief)
Arthur Pentland Dempster is a Canadian-American statistician, long a professor of statistics at Harvard
University. He is known both for the EM algorithm and for Dempster–Shafer theory of evidence (belief
functions), an alternative framework for reasoning under uncertainty.

## Key contributions
- **The EM algorithm (Dempster, Laird & Rubin, 1977).** A general iterative method for maximum-likelihood
  (and MAP) estimation when data are incomplete or have latent variables: alternate an Expectation step
  (compute the expected complete-data log-likelihood under the latent posterior) and a Maximization step
  (re-estimate parameters). Maps to → [[expectation-maximization]], and its canonical application here →
  [[gaussian-mixture-model]].
- **Dempster–Shafer theory.** A generalization of Bayesian probability using belief and plausibility
  functions; influential in AI uncertainty reasoning (no dedicated page yet).

## Connections
- Concepts: [[expectation-maximization]], [[gaussian-mixture-model]], [[latent-variable-model]]
- Related people: [[thomas-bayes]] (posterior reasoning underlying the E-step)

## Sources
- `raw/mml-book.pdf` §11.3, printed p. 360 (PDF p. 366) — "the EM algorithm was proposed by Dempster et al. (1977)."
