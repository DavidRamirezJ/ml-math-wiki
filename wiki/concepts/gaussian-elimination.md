---
title: Gaussian Elimination
type: concept
tags: [linear-algebra, algorithm]
difficulty: intermediate
sources: [mml-ch02-linear-algebra]
created: 2026-06-04
updated: 2026-06-04
---

# Gaussian Elimination

> **One-liner:** A bookkeeping recipe of row operations that grinds any linear system into a "staircase" form you can read the answer straight off of.

## 🟢 Intuition (Dummy)
Solving equations by hand is fiddly. Gaussian elimination is a *fixed procedure* that always
works: you're allowed three harmless moves — swap two equations, scale an equation, or add one
equation to another — none of which change the answer set. You keep applying them until the
equations line up in a neat staircase, where the last equation has one unknown, the next has two,
and so on. Then you read the solution off from the bottom up. It's named after [[carl-friedrich-gauss]].

## 🟡 Mechanics (Practitioner)
**Three elementary (row) transformations** (preserve the solution set):
1. Swap two rows. 2. Multiply a row by $\lambda \neq 0$. 3. Add a multiple of one row to another.

Work on the **augmented matrix** $[A\,|\,\mathbf{b}]$. Reduce to:
- **Row-echelon form (REF):** all-zero rows at the bottom; each **pivot** (first nonzero entry in
  a row) sits strictly to the right of the pivot above → "staircase."
- **Reduced row-echelon form (RREF):** REF *plus* every pivot $=1$ and is the only nonzero entry
  in its column. **Gaussian elimination = the algorithm that produces RREF.**

**Basic vs. free variables:** variables at pivot columns are *basic*; the rest are *free* (they
parameterize infinitely many solutions).

**Solving $A\mathbf{x}=\mathbf{b}$ — three steps (general solution):**
1. Find one **particular solution** (express $\mathbf b$ via pivot columns).
2. Find all solutions of the homogeneous $A\mathbf{x}=\mathbf 0$ (the [[basis-and-rank|null space]]).
3. General solution = particular $+$ any combination of homogeneous solutions.

Worked example from the book (Eq. 2.47): for a $4\times5$ system the general solution is
$\mathbf x = [2,0,-1,1,0]^\top + \lambda_1[2,1,0,0,0]^\top + \lambda_2[2,0,-1,2,1]^\top$.

**Computing an inverse:** row-reduce $[A\,|\,I_n] \rightsquigarrow [I_n\,|\,A^{-1}]$ (§2.3.3).

## 🔴 Theory (Expert)
- **The Minus-1 Trick** (§2.3.3): given RREF (no zero rows), pad to $n\times n$ by inserting rows
  $[0\cdots0\,{-1}\,0\cdots0]$ so the diagonal holds $1$ or $-1$; the columns carrying a $-1$ on the
  diagonal form a **basis of the null space** $\ker(A)$ (= solution space of $A\mathbf x = 0$,
  introduced as [[basis-and-rank|kernel/null space]], detailed in §2.7.3).
- **Where it's used:** computing determinants (§4.1), testing [[linear-independence]], finding a
  matrix inverse (§2.2.2), computing [[basis-and-rank|rank]], and finding a basis (§2.6.1).
- **Cost & limits:** operation count scales **cubically** in the number of equations — fine for
  thousands of variables, impractical for millions. For huge systems use **iterative methods**
  (Jacobi, Gauß–Seidel, SOR, Richardson) or **Krylov methods** (conjugate gradients, GMRES, BiCG),
  which iterate $\mathbf x^{(k+1)} = C\mathbf x^{(k)} + \mathbf d$ to shrink the residual.
- When no exact solution exists, use the least-squares normal equations
  $\mathbf x = (A^\top A)^{-1}A^\top \mathbf b$ (needs linearly independent columns) — the bridge to
  [[linear-regression]] (Ch. 9). The book warns: explicitly forming inverses/pseudo-inverses is
  numerically discouraged.

## Connections
- Builds on: [[matrix]], [[system-of-linear-equations]]
- Leads to: [[linear-independence]], [[basis-and-rank]], [[linear-regression]]
- Used by: matrix inversion, rank, determinant (Ch. 4)
- People: [[carl-friedrich-gauss]]

## Sources
- `raw/mml-book.pdf` §2.3, Def 2.6, printed pp. 27–35 (PDF pp. 32–40). Validated 2026-06-04.
