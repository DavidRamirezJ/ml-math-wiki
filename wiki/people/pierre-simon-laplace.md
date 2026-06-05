---
title: Pierre-Simon Laplace (1749–1827)
type: person
tags: [mathematician, linear-algebra, probability]
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Pierre-Simon Laplace (1749–1827)

> **Why they matter to ML:** his cofactor expansion is the textbook recursive way to compute determinants, and his broader work on probability and Bayesian reasoning underpins much of statistical machine learning.

## Life (brief)
French scholar often called "the Newton of France," active across mathematics, astronomy, and
physics through the Revolutionary and Napoleonic eras. His five-volume *Mécanique céleste* and
*Théorie analytique des probabilités* were landmark works; he made foundational contributions to
celestial mechanics, differential equations, and probability theory.

## Key contributions
- **Laplace expansion (cofactor expansion):** the recursive formula
  $\det(A)=\sum_k(-1)^{k+j}a_{kj}\det(A_{k,j})$ reducing an $n\times n$ [[determinant]] to
  $(n-1)\times(n-1)$ ones (Thm 4.2 in this chapter). Maps to the [[determinant]] page.
- **Probability and Bayesian inference** (outside this chapter): the Laplace transform, Bayesian
  probability with uniform priors, and the Laplace approximation — all central to later ML
  mathematics (forward link to Ch.6).

## Connections
- Concept: [[determinant]]
- Related: [[characteristic-polynomial]], [[carl-friedrich-gauss]]

## Sources
- `raw/mml-book.pdf` §4.1 (Thm 4.2, Example 4.3), printed pp. 101–102 (PDF pp. 106–107).
