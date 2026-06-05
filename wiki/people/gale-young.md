---
title: Gale Young (1912–1990)
type: person
tags: [mathematician, linear-algebra]
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# Gale Young (1912–1990)

> **Why they matter to ML:** co-author of the theorem proving that the truncated SVD gives the best low-rank approximation — the optimality guarantee that justifies PCA and SVD-based compression.

## Life (brief)
American mathematician. With the physicist Carl Eckart he published the 1936 result on low-rank
matrix approximation that bears both their names; the theorem became foundational in numerical linear
algebra, psychometrics, and later machine learning.

## Key contributions
- **The Eckart–Young theorem (1936):** the rank-$k$ truncation of a matrix's
  [[singular-value-decomposition]] is the optimal rank-$k$ approximation, with error equal to the
  first discarded singular value $\sigma_{k+1}$ (Thm 4.25). Maps to [[low-rank-approximation]].

## Connections
- Concept: [[low-rank-approximation]], [[singular-value-decomposition]]
- Related: [[carl-eckart]], [[principal-component-analysis]]

## Sources
- `raw/mml-book.pdf` §4.6 (Thm 4.25, Eckart and Young, 1936), printed pp. 131–132 (PDF pp. 136–137).
