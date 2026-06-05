---
title: Brook Taylor (1685–1731)
type: person
tags: [mathematician, calculus]
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Brook Taylor (1685–1731)

> **Why they matter to ML:** His series expansion is how we replace any smooth function by a polynomial — the basis of linearization, Newton's method, and the Laplace approximation.

## Life (brief)
English mathematician, born in Edmonton, Middlesex. A Fellow of the Royal
Society and one of its secretaries, he worked in the generation immediately after
Newton. His 1715 treatise *Methodus Incrementorum Directa et Inversa* introduced
the finite-difference calculus and the result now bearing his name, and also laid
early groundwork for the calculus of variations.

## Key contributions
- **Taylor's theorem / Taylor series** — representing a smooth function as an
  infinite sum of polynomial terms built from its derivatives at a point. Today
  the family of approximations $T_n(x)=\sum_{k=0}^n \frac{f^{(k)}(x_0)}{k!}(x-x_0)^k$
  and the multivariate generalization. Maps to [[taylor-series]]. (The special
  case $x_0=0$ is the *Maclaurin series*, after Colin Maclaurin.)

## Connections
- Concepts: [[taylor-series]], and via its terms [[gradient]] and [[hessian]]
- Related work: linearization (5.148), Newton's method and Laplace approximation (Ch. 6–7)

## Sources
- `raw/mml-book.pdf` §5.1.1, §5.8 (Defs 5.3–5.4, 5.7), printed pp. 142–145, 165–170.
