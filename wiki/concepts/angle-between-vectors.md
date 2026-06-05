---
title: Angle Between Vectors
type: concept
tags: [analytic-geometry, inner-product, foundations]
difficulty: beginner
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Angle Between Vectors

> **One-liner:** The angle between two vectors measures how aligned their directions are, computed from their inner product.

## 🟢 Intuition (Dummy)
Two arrows have an angle between them: zero if they point the same way, a right angle if they are
perpendicular, a straight angle if they point opposite. The inner product gives us this angle
without a protractor. If you divide the inner product by the two lengths, you always get a number
between $-1$ and $+1$ — and that number is the cosine of the angle. Close to $+1$ means "nearly
the same direction," near $0$ means "perpendicular."

## 🟡 Mechanics (Practitioner)
By the **Cauchy–Schwarz inequality**, for nonzero $x,y$ in an [[inner-product]] space,
$$ -1 \le \frac{\langle x,y\rangle}{\|x\|\,\|y\|} \le 1, $$
so there is a unique angle $\omega\in[0,\pi]$ with
$$ \cos\omega = \frac{\langle x,y\rangle}{\|x\|\,\|y\|}. $$

**Worked example (§3.4).** $x=\big[1,1\big]^\top$, $y=\big[1,2\big]^\top$, dot product:
$$ \cos\omega = \frac{x^\top y}{\sqrt{x^\top x}\,\sqrt{y^\top y}} = \frac{3}{\sqrt{2}\,\sqrt{5}} = \frac{3}{\sqrt{10}}, \qquad \omega = \arccos\tfrac{3}{\sqrt{10}} \approx 0.32\text{ rad} \approx 18^\circ. $$

**Inner product matters.** With a different inner product the angle changes: for
$x=\big[1,1\big]^\top$, $y=\big[-1,1\big]^\top$ the dot product gives $90^\circ$ (orthogonal), but
the inner product with $A=\big[\begin{smallmatrix}2&0\\0&1\end{smallmatrix}\big]$ gives
$\cos\omega = -\tfrac13$, i.e. $\omega \approx 109.5^\circ$ — no longer orthogonal.

**ML usage.** **Cosine similarity** $\cos\omega = \frac{x^\top y}{\|x\|\|y\|}$ is the workhorse
similarity score for text/word embeddings and recommendation systems, where direction matters more
than magnitude.

## 🔴 Theory (Expert)
- The definition is well-posed *precisely because* of Cauchy–Schwarz; without it the ratio could
  fall outside $[-1,1]$ and $\arccos$ would be undefined.
- Angles, like lengths and distances, are **relative to the inner product**. Orthogonality with
  respect to one inner product need not hold for another (see [[orthogonality]]).
- The cosine equals the correlation coefficient when vectors are mean-centered — linking geometry
  to statistics (Ch. 6).

## Connections
- Builds on: [[inner-product]], [[norm]]
- Leads to: [[orthogonality]], [[orthogonal-projection]] ($\|\pi_U(x)\| = |\cos\omega|\,\|x\|$)
- Used by: cosine similarity, embeddings, attention scores
- People: [[augustin-louis-cauchy]], [[hermann-schwarz]]

## Sources
- `raw/mml-book.pdf` §3.4 (Eqs 3.24–3.28, Examples 3.6–3.7), printed pp. 76–78 (PDF pp. 81–83).
