---
title: Werner Fenchel (1905–1988)
type: person
tags: [optimization, convex-analysis, history]
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Werner Fenchel (1905–1988)

> **Why they matter to ML:** He turned the Legendre transform into the general convex conjugate that works even for non-smooth losses — the modern duality workhorse.

## Life (brief)
German-born mathematician who emigrated to Denmark, a foundational figure of modern
**convex analysis**. His work on convex functions and conjugacy underpins the field that
Rockafellar and others later systematized.

## Key contributions
- **Legendre–Fenchel transform / convex conjugate.** Extends the classical
  [[adrien-marie-legendre|Legendre transform]] to functions that need be neither convex
  nor differentiable, using a supremum:
  $f^*(\mathbf{s})=\sup_{\mathbf{x}}(\langle\mathbf{s},\mathbf{x}\rangle-f(\mathbf{x}))$.
  Maps to [[convex-conjugate]].
- **Fenchel duality.** A duality framework for convex problems built on conjugate
  functions, complementary to [[lagrange-multipliers|Lagrangian duality]].

## Connections
- Concepts: [[convex-conjugate]], [[convex-optimization]]
- Related people: [[adrien-marie-legendre]], [[joseph-louis-lagrange]]

## Sources
- `raw/mml-book.pdf` §7.3.3, printed pp. 242–246 (PDF pp. 247–251).
