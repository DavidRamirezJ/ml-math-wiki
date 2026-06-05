---
title: Niels Henrik Abel (1802–1829)
type: person
tags: [algebra, history]
sources: [mml-ch07-continuous-optimization]
created: 2026-06-04
updated: 2026-06-04
---

# Niels Henrik Abel (1802–1829)

> **Why they matter to ML:** His impossibility theorem explains why we *must* optimize numerically — there's no algebraic formula for the roots of general high-degree equations.

## Life (brief)
Norwegian mathematician who, despite dying of tuberculosis at 26 in poverty, made
profound contributions to algebra and analysis. The Abel Prize is named in his honor.

## Key contributions
- **Abel–Ruffini theorem (Abel, 1826).** There is in general **no algebraic (radical)
  solution** for polynomial equations of degree $\ge 5$. The book cites this to motivate
  iterative numerical optimization: even setting a derivative to zero need not yield a
  closed-form stationary point, so we use [[gradient-descent]] instead. Relates to
  [[gradient-descent]].

## Connections
- Concepts: [[gradient-descent]]
- Related people: —

## Sources
- `raw/mml-book.pdf` §7 (margin note), printed p. 227 (PDF p. 232).
