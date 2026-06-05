---
title: André-Louis Cholesky (1875–1918)
type: person
tags: [mathematician, linear-algebra, numerical-methods]
sources: [mml-ch04-matrix-decompositions]
created: 2026-06-04
updated: 2026-06-04
---

# André-Louis Cholesky (1875–1918)

> **Why they matter to ML:** his matrix factorization is the standard, numerically stable way to handle covariance matrices — used everywhere from Gaussian sampling to the reparameterization trick in deep generative models.

## Life (brief)
French military officer, geodesist, and mathematician. He developed his factorization method around
1910 while working on geodetic surveying problems that required solving large systems of normal
equations. He was killed in action in 1918, near the end of World War I, and his method was published
posthumously (1924) by a fellow officer.

## Key contributions
- **The Cholesky factorization** $A=LL^\top$ for symmetric positive definite matrices, with $L$
  lower-triangular and positive on the diagonal. Today called the **[[cholesky-decomposition]]** — a
  "matrix square root" and a symmetry-exploiting special case of LU decomposition. It is the default
  tool for solving SPD systems, sampling from multivariate Gaussians, and computing log-determinants.

## Connections
- Concept: [[cholesky-decomposition]]
- Related: [[symmetric-positive-definite-matrix|positive definite matrix]], [[determinant]], [[gaussian-elimination]]

## Sources
- `raw/mml-book.pdf` §4.3 (Thm 4.18), printed pp. 114–115 (PDF pp. 119–120).
