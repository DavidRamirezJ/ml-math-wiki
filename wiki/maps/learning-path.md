---
title: Learning Path — The Atomic Concepts
type: map
tags: [learning-path, foundations, curriculum]
sources: [mml-ch02-linear-algebra, mml-ch05-vector-calculus, mml-ch06-probability-and-distributions, mml-ch07-continuous-optimization]
created: 2026-06-18
updated: 2026-06-18
---

# Learning Path — The Atomic Concepts

> **One-liner:** Master ~20 irreducible "atoms" and every advanced ML method becomes
> assembly, not new material.

## What "atomic" means here

The **atomic concepts** are the foundations at the *bottom* of the dependency graph — the ideas
that have no prerequisites of their own and that every other page is built from. They are distinct
from the *most-mentioned* pages (popularity). For example [[principal-component-analysis]] is
referenced by 30 pages, but it is **not** atomic — it is a composite built from eigen-structure,
projection, and variance. The atoms are what PCA is *made of*.

The whole of *Mathematics for Machine Learning* rests on **four pillars**. Learn the atoms of each
pillar and the rest of the wiki is recombination.

---

## Pillar 1 — Linear Algebra · *the language of data*

Data are vectors; operations on data are matrices. The most foundational pillar.

- [[vector]] — the single most foundational object; everything is a vector.
- [[vector-space]] — the arena vectors live in; defines "add & scale."
- [[matrix]] — both a data table **and** a function; the most-referenced concept page.
- [[linear-mapping]] — the deep truth: every matrix is a structure-preserving function.
- [[linear-independence]] + [[basis-and-rank]] — what *dimension* and *redundancy* mean.
- [[inner-product]] — gives length, angle, distance, similarity (one of the highest in-degree pages).

People: [[carl-friedrich-gauss]]

## Pillar 2 — Vector Calculus · *the language of learning*

Learning = adjusting numbers to reduce error. Calculus says which way to adjust.

- [[derivative]] — rate of change; the seed of everything downstream.
- [[partial-derivative]] — the derivative along one input when there are many.
- [[gradient]] — the bundle of all partial slopes → steepest-ascent direction. Powers
  [[gradient-descent]] and [[backpropagation]].
- [[chain-rule]] — the one rule that makes training deep models possible.

People: [[carl-gustav-jacob-jacobi]], [[ludwig-otto-hesse]], [[brook-taylor]]

## Pillar 3 — Probability · *the language of uncertainty*

Real data is noisy; probability is how models reason about what they don't know.

- [[random-variable]] — turns a random outcome into a number you can compute with.
- [[probability-distribution]] — the full bookkeeping of how likely each value is.
- [[expected-value-and-mean]] + [[covariance-and-variance]] — center and spread, the two
  numbers every model estimates.
- [[bayes-theorem]] — the belief-update rule; engine of all probabilistic inference.
- [[gaussian-distribution]] — the default distribution; one of the most depended-on pages.

People: [[thomas-bayes]], [[carl-friedrich-gauss]], [[pierre-simon-laplace]]

## Pillar 4 — Optimization · *the language of "best"*

Every trained model is the answer to "find the parameters that minimize error."

- [[loss-function]] — turns "how wrong?" into one number to minimize.
- [[gradient-descent]] — the universal recipe: step downhill, repeat.
- [[lagrange-multipliers]] — how to optimize *under constraints* (foundation of SVMs).

People: [[joseph-louis-lagrange]], [[augustin-louis-cauchy]]

---

## The honest shortlist — attack these first, in this order

The true bedrock. Every advanced page ([[principal-component-analysis]],
[[support-vector-machine]], [[backpropagation]], [[bayesian-linear-regression]]) is just these
atoms combined:

1. [[vector]]
2. [[matrix]]
3. [[linear-mapping]]
4. [[inner-product]]
5. [[derivative]]
6. [[gradient]]
7. [[chain-rule]]
8. [[random-variable]]
9. [[probability-distribution]]
10. [[bayes-theorem]]
11. [[gradient-descent]]

## From atoms to methods (what they assemble into)

- Linear-algebra atoms → [[eigenvalues-and-eigenvectors]] → [[eigendecomposition]] →
  [[singular-value-decomposition]] → [[principal-component-analysis]]
- Calculus atoms → [[gradient-descent]] → [[backpropagation]] → trained neural nets
- Probability atoms → [[maximum-likelihood-estimation]] → [[bayesian-inference]] →
  [[bayesian-linear-regression]], [[gaussian-mixture-model]]
- Optimization + linear algebra → [[lagrange-multipliers]] → [[support-vector-machine]]

## Connections

- Builds on: nothing — this **is** the foundation layer.
- Leads to: every concept page in the wiki.
- See also: [[index]] (full catalog), [[overview]] *(pending)*

## Sources

- `raw/mml-book.pdf` Part I (Chs. 2–7) — the mathematical-foundations half of the book.
- Dependency structure derived from inbound-link analysis of `wiki/concepts/`.
