---
title: Norm
type: concept
tags: [analytic-geometry, inner-product, foundations, deep-learning]
difficulty: beginner
sources: [mml-ch03-analytic-geometry, llm-appD-training-loop-bells-and-whistles]
created: 2026-06-04
updated: 2026-10-13
---

# Norm

> **One-liner:** A norm is a rule that measures how "long" a vector is — a single non-negative number standing in for its size.

## 🟢 Intuition (Dummy)
Think of a vector as an arrow starting at the origin. Its *length* is how far the tip is from the
start. A norm is just the recipe for that length. The familiar recipe is the straight-line distance
(Pythagoras). But there are other sensible recipes: if you can only walk along city blocks
(no diagonals), the "distance" is how far you walk along the streets — that's the Manhattan norm.
Different recipes, all called norms, share three common-sense rules: scaling an arrow scales its
length, the direct route is never longer than a detour, and only the zero arrow has length zero.

## 🟡 Mechanics (Practitioner)
**Definition 3.1.** A **norm** on a vector space $V$ is a function $\|\cdot\| : V \to \mathbb R$,
$x \mapsto \|x\|$, satisfying for all $\lambda\in\mathbb R$ and $x,y\in V$:

- **Absolutely homogeneous:** $\|\lambda x\| = |\lambda|\,\|x\|$
- **Triangle inequality:** $\|x+y\| \le \|x\| + \|y\|$
- **Positive definite:** $\|x\| \ge 0$, and $\|x\| = 0 \iff x = \mathbf 0$

**Two common norms on $\mathbb R^n$:**
$$ \|x\|_1 := \sum_{i=1}^n |x_i| \quad(\text{Manhattan / }\ell_1), \qquad
   \|x\|_2 := \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^\top x} \quad(\text{Euclidean / }\ell_2). $$

The Euclidean norm is the default throughout MML. The set of unit-norm vectors ($\|x\|=1$) is a
diamond for $\ell_1$ and a circle for $\ell_2$ (Fig. 3.3).

**Worked example.** For $x = \big[3, 4\big]^\top$: $\|x\|_1 = |3|+|4| = 7$ and
$\|x\|_2 = \sqrt{9+16} = 5$.

**ML usage.** Norms measure error magnitudes and regularize models: $\ell_2$ ("ridge")
penalizes large weights; $\ell_1$ ("lasso") drives weights to exactly zero, giving sparsity.
Loss functions like mean-squared error are squared $\ell_2$ norms of residuals.

### In LLMs: measuring the gradient for gradient clipping
*Build a Large Language Model (From Scratch)*, §D.3, uses the $\ell_2$ norm to decide when a gradient is
"too big" during training (see [[gradient-clipping]]). The book states it for a vector
$\mathbf v = [v_1, v_2, \dots, v_n]$ (book p. 317):
$$\|\mathbf v\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$
- $\mathbf v$ = the gradient, flattened into one long vector; each $v_i$ is the loss's [[partial-derivative]] with respect to one parameter.
- $n$ = the number of trainable parameters (about 162 million for the book's untied `GPTModel` with `context_length=256`; "124M" counts the weights with weight tying, book §4.6).
- In words: square every component, add them up, take the square root — Pythagoras in $n$ dimensions.

The book adds that "this calculation method is also applied to matrices" (book p. 318): square every entry,
sum, square-root. (Strictly, that matrix version is the **Frobenius norm** — see 🔴 Theory below.)

**Clipping rule.** If $\|\mathbf G\|_2 > \text{max\_norm}$, rescale by $c = \text{max\_norm}/\|\mathbf G\|_2$, so
$\mathbf G' = c\,\mathbf G$ has norm exactly max_norm. By absolute homogeneity, $\|c\mathbf G\| = |c|\,\|\mathbf G\|$ —
this is the first norm axiom above doing real work.

**Book example (book p. 318)** with $\mathbf G = \begin{bmatrix}1 & 2\\3 & 4\end{bmatrix}$ and max_norm $=1$:

> ⚠️ **Book typo:** the book prints $\|\mathbf G\|_2 = \sqrt{1^2 + 2^2 + 2^2 + 4^2} = \sqrt{25} = 5$ and scales by $1/5$.
> The entry 3 was copied as 2. The true value is $\sqrt{1 + 4 + 9 + 16} = \sqrt{30} \approx 5.477$, so the correct
> factor is $1/\sqrt{30} \approx 0.183$ and $\mathbf G' \approx \begin{bmatrix}0.183 & 0.365\\0.548 & 0.730\end{bmatrix}$.
> With the book's $1/5$, $\|\mathbf G'\| = \sqrt{30}/5 \approx 1.095 \neq 1$. (The book's norm $\sqrt{25}=5$ would be correct only for
> $\begin{bmatrix}1&2\\2&4\end{bmatrix}$.) This is an arithmetic slip, not a disagreement about the definition.

In PyTorch, `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` computes one $\ell_2$ norm over
**all** parameter gradients together and scales them by the same factor. In the book's demo the largest single
gradient entry drops from `tensor(0.0411)` to `tensor(0.0185)` (book p. 319), i.e. a common factor of about 0.45.

## 🔴 Theory (Expert)
- A norm always **induces a metric** $d(x,y) = \|x-y\|$ (see [[metric]]) and the unit ball
  $\{x : \|x\|\le 1\}$ is convex (a direct consequence of the triangle inequality and homogeneity).
- Every **[[inner-product]] induces a norm** $\|x\| := \sqrt{\langle x,x\rangle}$, but not conversely:
  the Manhattan norm is **not** induced by any inner product (it fails the parallelogram law
  $\|x+y\|^2 + \|x-y\|^2 = 2\|x\|^2 + 2\|y\|^2$, which holds iff a norm comes from an inner product).
- The family $\|x\|_p = \big(\sum_i |x_i|^p\big)^{1/p}$ for $p\ge 1$ generalizes these; $p=\infty$
  gives the max norm $\max_i |x_i|$. On finite-dimensional spaces all norms are **equivalent**
  (each is bounded by constant multiples of any other), so they induce the same topology — though
  not the same geometry or the same optimal solutions.
- **Matrix norms (LLM context).** For a matrix $\mathbf G\in\mathbb R^{m\times n}$, the "square every entry" recipe
  used in the LLM book is the **Frobenius norm** $\|\mathbf G\|_F = \sqrt{\sum_{i,j} G_{ij}^2} = \sqrt{\operatorname{tr}(\mathbf G^\top\mathbf G)}$,
  which equals the vector $\ell_2$ norm of $\operatorname{vec}(\mathbf G)$ and is induced by the Frobenius inner product
  $\langle A,B\rangle = \operatorname{tr}(A^\top B)$. It differs from the **spectral (operator) 2-norm** $\|\mathbf G\|_2 = \sigma_{\max}(\mathbf G)$,
  the largest singular value (see [[singular-value-decomposition]]). For $\begin{bmatrix}1&2\\3&4\end{bmatrix}$:
  Frobenius $=\sqrt{30}\approx 5.477$, spectral $\approx 5.465$. The book's notation $|\mathbf G|_2$ means the Frobenius one.
- **Clipping as projection.** Norm clipping $\mathbf g \mapsto \mathbf g\cdot\min(1, r/\|\mathbf g\|_2)$ is the Euclidean (nearest-point)
  projection onto the convex ball $\{\mathbf x : \|\mathbf x\|_2 \le r\}$ — it preserves direction and changes only length. (This is projection onto a convex set, not the [[orthogonal-projection]] onto a subspace.)

## Connections
- Builds on: [[vector]], [[vector-space]]
- Leads to: [[metric]], [[inner-product]], [[angle-between-vectors]]
- Used by: [[orthogonal-projection]] (projection error), [[regularization]], [[loss-function]], [[gradient-clipping]] (LLM training), [[weight-decay]]
- People: —

## Sources
- `raw/mml-book.pdf` §3.1 (Definition 3.1, Examples 3.1–3.2) and §3.3, printed pp. 71–72, 75 (PDF pp. 76–77, 80).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.3 Gradient clipping, book pp. 317–319 (L2 norm p. 317; matrix example p. 318; PyTorch demo p. 319); §D.4, book p. 320 (clipping in `train_model`).
