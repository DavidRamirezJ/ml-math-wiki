---
title: Johan Jensen (1859–1925)
type: person
tags: [convex-analysis, inequalities, history]
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Johan Jensen (1859–1925)

> **Why they matter to ML:** Jensen's inequality is the defining property of convex functions and appears throughout ML — from convex losses to the ELBO in variational inference.

## Life (brief)
Danish mathematician and engineer (full name Johan Ludwig William Valdemar Jensen). A
largely self-taught mathematician who worked for a telephone company while making
contributions to analysis and complex function theory.

## Key contributions
- **Jensen's inequality.** For a convex function $f$ and weights $\theta\in[0,1]$,
  $f(\theta x+(1-\theta)y)\le\theta f(x)+(1-\theta)y$; more generally
  $f(\mathbb{E}[X])\le\mathbb{E}[f(X)]$. This *is* the definition of convexity used in the
  book (Eq. 7.30). Maps to [[convex-optimization]].

## Connections
- Concepts: [[convex-optimization]]
- Related people: [[werner-fenchel]], [[adrien-marie-legendre]]

## Sources
- `raw/mml-book.pdf` §7.3, Remark after Eq. 7.30, printed p. 239 (PDF p. 244).
