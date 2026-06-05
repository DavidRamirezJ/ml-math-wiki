---
title: Adrien-Marie Legendre (1752–1833)
type: person
tags: [optimization, convex-analysis, history]
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Adrien-Marie Legendre (1752–1833)

> **Why they matter to ML:** The Legendre transform re-describes a convex function by its slopes — the basis of the convex conjugate and a key duality tool in ML.

## Life (brief)
French mathematician active during and after the Revolution. He contributed broadly to
number theory, statistics, and analysis, and (independently of Gauss) published the
**method of least squares** in 1805.

## Key contributions
- **Legendre transform.** A transformation taking a convex differentiable function
  $f(x)$ to a function of its tangent slopes $s=\nabla f(x)$. Generalized by Fenchel to
  the **Legendre–Fenchel transform / convex conjugate**. Maps to [[convex-conjugate]].
- **Method of least squares.** Foundational for regression and the squared-error
  objectives optimized by [[gradient-descent]] (see Ch. 9). (Famously contested with
  [[carl-friedrich-gauss]].)

## Connections
- Concepts: [[convex-conjugate]], [[convex-optimization]]
- Related people: [[werner-fenchel]] (generalized the transform), [[joseph-louis-lagrange]], [[carl-friedrich-gauss]]

## Sources
- `raw/mml-book.pdf` §7.3.3, printed pp. 242–243 (PDF pp. 247–248).
