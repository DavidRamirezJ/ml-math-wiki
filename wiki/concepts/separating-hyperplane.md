---
title: Separating Hyperplane
type: concept
tags: [classification, geometry, support-vector-machine]
difficulty: intermediate
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Separating Hyperplane

> **One-liner:** A separating hyperplane is a flat dividing wall that puts one class on one side and the other class on the other.

## 🟢 Intuition (Dummy)
In a 2D drawing, a hyperplane is just a straight line; in 3D it is a flat sheet; in higher dimensions
it is the same idea — a flat wall one dimension lower than the room it lives in. To classify, you ask
only: *which side of the wall is this point on?* Everything on one side is "positive," everything on
the other is "negative." The wall also has a built-in arrow telling you which side is which.

## 🟡 Mechanics (Practitioner)
A hyperplane is an [[affine-space|affine subspace]] of dimension $D-1$ in $\mathbb R^D$. Define the
**linear classifier function** parametrized by $w\in\mathbb R^D$ and $b\in\mathbb R$:
$$ f(x)=\langle w,x\rangle + b. $$
The **separating hyperplane** is the level set
$$ \{x\in\mathbb R^D : \langle w,x\rangle + b = 0\}. $$
Classify a test point by the sign:
$$ \hat y = \begin{cases} +1 & \langle w,x\rangle + b \ge 0\\ -1 & \langle w,x\rangle + b < 0.\end{cases} $$
A correct fit on training data satisfies $\langle w,x_n\rangle+b\ge 0$ when $y_n=+1$ and $<0$ when
$y_n=-1$, combined compactly as
$$ y_n(\langle w,x_n\rangle + b) \ge 0. $$

**$w$ is the normal vector.** For any two points $x_a,x_b$ on the plane, $f(x_a)-f(x_b)=\langle w, x_a-x_b\rangle=0$,
so $w$ is [[orthogonality|orthogonal]] to every direction lying in the plane. The scalar $b$ is the
**intercept** (offset from the origin). Distances to the plane are computed by [[orthogonal-projection]]
along $w/\|w\|$.

**ML usage.** The decision surface of the SVM, the perceptron, logistic regression, and (per-neuron)
linear layers. The SVM's job is to choose *which* separating hyperplane — the max-[[margin]] one.

## 🔴 Theory (Expert)
$\{x:\langle w,x\rangle+b=0\}$ is a coset of the $(D-1)$-dimensional subspace $w^\perp$; it is the
solution set of one linear equation, hence an affine subspace of codimension 1. Rescaling
$(w,b)\mapsto(\lambda w,\lambda b)$, $\lambda>0$, leaves the set and the classification *unchanged*,
revealing a one-parameter scale ambiguity — exactly the ambiguity the SVM resolves by fixing the
[[margin]] scale ($\|w\|=1$, or unit margin). The signed quantity $f(x)/\|w\|$ is the **signed distance**
from $x$ to the hyperplane; its absolute value is what the margin maximizes. Linear separability — the
existence of some $(w,b)$ with $y_n f(x_n)>0$ for all $n$ — is not guaranteed; when it fails one uses
slack ([[soft-margin-svm]]) or a feature map ([[kernel-trick]]).

## Connections
- Builds on: [[affine-space]], [[inner-product]], [[orthogonality]], [[orthogonal-projection]]
- Leads to: [[margin]], [[hard-margin-svm]], [[support-vector-machine]]
- Used by: SVM, perceptron, logistic regression, linear classifiers
- People: —

## Sources
- `raw/mml-book.pdf` §12.1, printed pp. 372–373 (PDF pp. 378–379). Hyperplanes as affine subspaces: §2.8.
