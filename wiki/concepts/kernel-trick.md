---
title: Kernel Trick
type: concept
tags: [classification, support-vector-machine, kernels, rkhs, nonlinear]
difficulty: advanced
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Kernel Trick

> **One-liner:** The kernel trick lets a straight-line method draw curved boundaries by secretly measuring similarity in a richer space — without ever building that space.

## 🟢 Intuition (Dummy)
A straight line can't separate dots arranged in a bullseye (one color in the center, the other in a ring).
But if you lift the dots onto a hill — pushing the center ones up — a flat sheet *can* slice them apart. The
kernel trick does this lifting **implicitly**: instead of actually computing new "lifted" coordinates (which
could be huge or even infinite), it uses a shortcut formula that directly returns *how similar two points are
after lifting*. Because the SVM only ever needs similarities between examples, that shortcut is all it needs —
and suddenly a linear method produces curved decision boundaries.

## 🟡 Mechanics (Practitioner)
The [[dual-support-vector-machine|dual SVM]] uses examples **only through inner products**
$\langle x_i,x_j\rangle$. Replace each by a **kernel**
$$ k(x_i,x_j) = \langle \phi(x_i),\phi(x_j)\rangle_{\mathcal H}, $$
where $\phi:\mathcal X\to\mathcal H$ is a (possibly nonlinear, possibly infinite-dimensional) feature map into
a Hilbert space $\mathcal H$. You never form $\phi$ explicitly — you just evaluate $k$. The dual becomes
$$ \min_\alpha\ \tfrac12\sum_{i,j} y_i y_j \alpha_i\alpha_j\, k(x_i,x_j) - \sum_i \alpha_i,\quad \text{s.t. } \sum_i y_i\alpha_i=0,\ 0\le\alpha_i\le C, $$
and prediction is $f(x)=\sum_{n\in SV}\alpha_n y_n\, k(x_n,x)+b$.

**Popular kernels** (for $x_i\in\mathbb R^D$):
- **Linear:** $k(x,x')=\langle x,x'\rangle$.
- **Polynomial (degree $d$):** $k(x,x')=(\langle x,x'\rangle + c)^d$.
- **Gaussian RBF:** $k(x,x')=\exp(-\gamma\|x-x'\|^2)$ — its feature space is *infinite-dimensional*.
- **Rational quadratic.**

The **Gram / kernel matrix** $K_{ij}=k(x_i,x_j)$ must be **symmetric positive semidefinite**: $z^\top K z\ge0$
for all $z$. Choose the kernel and its parameters by nested [[cross-validation]].

**ML usage.** Turns the linear SVM into a flexible nonlinear classifier (Fig. 12.10). The same trick works for
any algorithm written in terms of inner products (kernel PCA, Gaussian processes, kernel ridge regression).

## 🔴 Theory (Expert)
A function $k:\mathcal X\times\mathcal X\to\mathbb R$ is a **kernel** iff it is symmetric and positive
semidefinite; equivalently (Mercer/Moore–Aronszajn) there exists a Hilbert space and feature map realizing
$k(x,x')=\langle\phi(x),\phi(x')\rangle$. Every kernel has a **unique reproducing-kernel Hilbert space (RKHS)**
$\mathcal H$ (Aronszajn, 1950), with **canonical feature map** $\phi(x)=k(\cdot,x)$ and reproducing property
$\langle f,k(\cdot,x)\rangle=f(x)$. The generalization $\langle\cdot,\cdot\rangle\to k(\cdot,\cdot)$ is the
**kernel trick**. Hidden bonus: $\mathcal X$ need not be $\mathbb R^D$ at all — kernels are defined on strings,
graphs, sets, and distributions, so the SVM applies to non-vector data. 

**Disambiguation:** this "kernel" (RKHS) is unrelated to the linear-algebra *kernel = null space*, and to the
*smoothing kernel* in [[kernel-density-estimation]]. Computationally, a kernel often costs far less than forming
$\phi$ explicitly (e.g. the polynomial kernel's expansion has combinatorially many terms; RBF's is infinite).

## Connections
- Builds on: [[inner-product]], [[dual-support-vector-machine]], [[symmetric-positive-definite-matrix]]
- Leads to: kernel PCA, Gaussian processes, kernel ridge regression
- Used by: [[support-vector-machine]], nonlinear classification
- People: —

## Sources
- `raw/mml-book.pdf` §12.4 (Eqs. 12.52–12.53), printed pp. 388–390 (PDF pp. 394–396).
