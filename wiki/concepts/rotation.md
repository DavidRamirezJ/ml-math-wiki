---
title: Rotation
type: concept
tags: [analytic-geometry, linear-algebra]
difficulty: intermediate
sources: [mml-ch03-analytic-geometry]
created: 2026-06-04
updated: 2026-06-04
---

# Rotation

> **One-liner:** A rotation spins vectors around a fixed point by an angle, changing their direction but never their length.

## 🟢 Intuition (Dummy)
Spin a clock hand around its center: the tip traces a circle, the hand stays the same length, it
just points a new way. A rotation does that to every vector at once, turning the whole space by a
fixed angle about the origin. By convention a positive angle turns counterclockwise. Nothing is
stretched or squashed — distances and angles between vectors are perfectly preserved. That is why
rotations matter in graphics and robotics: a robot arm's joints rotate to reposition without
deforming the parts.

## 🟡 Mechanics (Practitioner)
A **rotation** is a length- and angle-preserving linear map — an [[orthogonal-matrix|orthogonal]]
automorphism of a Euclidean space, with $\det = +1$.

**In $\mathbb R^2$.** Rotating the standard basis by $\theta$ sends
$e_1\mapsto[\cos\theta,\sin\theta]^\top$, $e_2\mapsto[-\sin\theta,\cos\theta]^\top$, so
$$ R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}. $$

**In $\mathbb R^3$.** Rotate about each axis (the named axis is fixed):
$$ R_1(\theta)=\begin{bmatrix}1&0&0\\0&\cos\theta&-\sin\theta\\0&\sin\theta&\cos\theta\end{bmatrix},\
   R_2(\theta)=\begin{bmatrix}\cos\theta&0&\sin\theta\\0&1&0\\-\sin\theta&0&\cos\theta\end{bmatrix},\
   R_3(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta&0\\\sin\theta&\cos\theta&0\\0&0&1\end{bmatrix}. $$

**In $\mathbb R^n$ — Givens rotation (Def 3.11).** Fix $n-2$ dimensions and rotate within the
$(i,j)$-plane: $R_{ij}(\theta)$ is the identity except
$$ r_{ii}=\cos\theta,\quad r_{ij}=-\sin\theta,\quad r_{ji}=\sin\theta,\quad r_{jj}=\cos\theta. $$

**Worked example.** Rotating $x=\big[1,0\big]^\top$ by $90^\circ$:
$R(90^\circ)=\big[\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\big]$, so
$R x = \big[0,1\big]^\top$ — the vector turns onto the vertical axis, length still 1.

**ML usage.** Givens rotations are building blocks of QR decomposition and eigenvalue algorithms;
rotations also model data augmentation, pose in robotics/vision, and basis changes in PCA.

## 🔴 Theory (Expert)
- Rotations are the orthogonal matrices with $\det = +1$ — the **special orthogonal group** $SO(n)$.
- **Properties (§3.9.4):** preserve distances $\|x-y\|=\|R x - R y\|$ and angles; in $\ge 3$
  dimensions rotations generally **do not commute** ($R_1 R_2 \ne R_2 R_1$), so order matters. In
  $\mathbb R^2$ they commute, $R(\phi)R(\theta)=R(\theta)R(\phi)$, forming an **Abelian group**.
- A general $n$-D rotation factors into a product of Givens rotations.
- Reflections are the other orthogonal maps ($\det=-1$); together rotations and reflections make up
  the full orthogonal group $O(n)$.

## Connections
- Builds on: [[orthogonal-matrix]], [[linear-mapping]], [[angle-between-vectors]]
- Leads to: QR decomposition, eigenvalue algorithms (Ch. 4), basis change for PCA (Ch. 10)
- Used by: computer graphics, robotics, data augmentation
- People: —

## Sources
- `raw/mml-book.pdf` §3.9 (Eqs 3.74–3.81, Def 3.11), printed pp. 91–94 (PDF pp. 96–99).
