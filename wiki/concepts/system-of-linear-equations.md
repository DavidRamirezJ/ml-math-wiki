---
title: System of Linear Equations
type: concept
tags: [linear-algebra, foundations]
difficulty: beginner
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# System of Linear Equations

> **One-liner:** Several straight-line equations that must all be true at once — and you hunt for the values that satisfy every one of them.

## 🟢 Intuition (Dummy)
Imagine two recipes that share ingredients, and you want a shopping list that satisfies both at
once. Each constraint is an equation; together they're a *system*. Geometrically, each equation
with two unknowns is a **line** on a graph. A solution is a point sitting on *all* the lines at once.
Two lines can cross at one point (one answer), lie exactly on top of each other (infinitely many
answers), or run parallel and never meet (no answer). That's the whole story.

## 🟡 Mechanics (Practitioner)
General form with $m$ equations and $n$ unknowns:

$$
\begin{aligned}
a_{11}x_1 + \cdots + a_{1n}x_n &= b_1 \\
&\;\;\vdots \\
a_{m1}x_1 + \cdots + a_{mn}x_n &= b_m
\end{aligned}
$$

Compactly, $A\mathbf{x} = \mathbf{b}$ with coefficient [[matrix]] $A \in \mathbb{R}^{m\times n}$, unknowns
$\mathbf{x}\in\mathbb{R}^n$, right-hand side $\mathbf{b}\in\mathbb{R}^m$. Solved by [[gaussian-elimination]].

**Worked example (unique solution).**
$$
\begin{aligned}
x_1 + x_2 + x_3 &= 3\\
x_1 - x_2 + 2x_3 &= 2\\
x_2 + x_3 &= 2
\end{aligned}
\;\Rightarrow\; (x_1,x_2,x_3) = (1,1,1)
$$

**No solution example.** Replace the third equation with $2x_1 + 3x_3 = 1$: adding the first two
gives $2x_1+3x_3=5$, contradicting $=1$. No point satisfies all three.

## 🔴 Theory (Expert)
**Trichotomy theorem:** a real linear system has **exactly one of**: no solution, a unique solution,
or infinitely many solutions — *never* a finite number $>1$. (p. 21)

- **Geometric view:** 2 unknowns → each equation is a line; the solution set is the intersection
  (point, line, or $\varnothing$). 3 unknowns → each equation is a plane; intersection is a plane,
  line, point, or $\varnothing$.
- **Structure of the solution set:** if $\mathbf{x}_p$ is one ("particular") solution, the full set is
  $\mathbf{x}_p + \ker(A)$ — a [[affine-space|affine subspace]] whose direction space is the null space
  $\ker(A)$. Infinitely many solutions ⇔ $\ker(A)\neq\{\mathbf 0\}$ and the system is consistent.
- **Free variables** parameterize the infinite case, e.g. setting $x_3 = a\in\mathbb{R}$ yields a
  one-parameter family $\big(\tfrac52-\tfrac32a,\;\tfrac12+\tfrac12a,\;a\big)$.
- **Link to ML:** when no exact solution exists (overdetermined, noisy data), [[linear-regression]]
  finds the least-squares best fit instead (Ch. 9).

## Connections
- Builds on: [[vector]], [[matrix]]
- Leads to: [[gaussian-elimination]], [[basis-and-rank]], [[affine-space]]
- Used by: [[linear-regression]]
- People: [[carl-friedrich-gauss]]

## Sources
- `raw/mml-book.pdf` §2.1, pp. 19–21 (PDF pp. 24–26)
