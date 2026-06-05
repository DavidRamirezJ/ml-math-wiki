---
title: Margin
type: concept
tags: [classification, geometry, support-vector-machine, generalization]
difficulty: intermediate
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Margin

> **One-liner:** The margin is the width of the empty buffer zone between the dividing line and the nearest data point on either side.

## 🟢 Intuition (Dummy)
Picture the dividing line as a road and the data points as houses. The margin is how far back from the
road the closest house sits — and we want the road built so the nearest houses on both sides are as far
away as possible. A wide buffer means a small wobble in where we drew the road still classifies everyone
correctly, so the classifier is more trustworthy on houses it has never seen.

## 🟡 Mechanics (Practitioner)
The **margin** is the distance from the [[separating-hyperplane]] to the closest example. To compute it,
take a point $x_a$ and its [[orthogonal-projection]] $x_a'$ onto the plane. Since $w$ is normal to the
plane, $x_a = x_a' + r\,\dfrac{w}{\|w\|}$ for some signed distance $r$.

There is a scale ambiguity ($w$ and $b$ can be rescaled freely), so we must fix a scale. Two equivalent choices:

1. **Normalize $w$:** require $\|w\|=1$ and maximize $r$ directly:
   $$ \max_{w,b,r}\ r \quad\text{s.t.}\quad y_n(\langle w,x_n\rangle+b)\ge r,\ \ \|w\|=1,\ \ r>0. $$
2. **Normalize the margin:** scale the data so the closest example satisfies $\langle w,x_a\rangle+b=1$.
   Substituting the projection relation and using $\langle w,w\rangle=\|w\|^2$ gives the clean result
   $$ r = \frac{1}{\|w\|}. $$

So **maximizing the margin = minimizing $\|w\|$** (equivalently $\tfrac12\|w\|^2$). The full distance
between the two margin boundaries $\langle w,x\rangle+b=\pm1$ is $2/\|w\|$.

**Worked example.** If after training $w=[3,4]^\top$ then $\|w\|=5$ and the margin is $r=1/5=0.2$. Halving
$w$ to $[1.5,2]^\top$ doubles the *apparent* margin — which is why the scale must be pinned down.

**ML usage.** The quantity the SVM maximizes; large margin is the SVM's inductive bias and is what makes it
[[overfitting-and-underfitting|generalize]] well.

## 🔴 Theory (Expert)
**Theorem 12.1** proves the two scalings above give the *same* optimizer: maximizing $r$ subject to
$\|w\|=1$ is equivalent to $\min_{w,b}\tfrac12\|w\|^2$ subject to $y_n(\langle w,x_n\rangle+b)\ge 1$. The
proof reparametrizes $w'=w/\|w\|$, divides the constraint by $r>0$ (valid under separability), and observes
$\|w''\| = 1/r$, so maximizing $r$ is minimizing $\|w''\|$.

The margin is central to **statistical learning theory**: [[vladimir-vapnik|Vapnik]] and Chervonenkis
showed that when the margin is large the effective complexity (VC-dimension-like capacity) of the
function class is small, bounding the generalization error independently of the input dimension. This is
why a large-margin separator is preferred among the infinitely many separators (Steinwart & Christmann,
2008). The margin also equals the **projection error** incurred when projecting the closest example onto
the hyperplane.

## Connections
- Builds on: [[orthogonal-projection]], [[inner-product]], [[norm]], [[separating-hyperplane]]
- Leads to: [[hard-margin-svm]], [[soft-margin-svm]], [[support-vector-machine]]
- Used by: SVM, margin-based generalization bounds
- People: [[vladimir-vapnik]]

## Sources
- `raw/mml-book.pdf` §12.2.1–12.2.3, **Thm 12.1**, printed pp. 374–379 (PDF pp. 380–385).
