---
title: Gram Matrix
type: concept
tags: [linear-algebra, inner-product, attention, kernels]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Gram Matrix

> **One-liner:** A Gram matrix is a table that lists the similarity (dot product) between every pair of vectors in a collection.

## 🟢 Intuition (Dummy)
Imagine a class of six students, and you want to know how alike each pair of them is. You make a
6-by-6 grid: row "Ana", column "Ben" holds the Ana–Ben similarity score. The grid is mirrored across
its diagonal (Ana–Ben is the same as Ben–Ana), and the diagonal itself holds each student compared
with themselves.

That grid is a Gram matrix. In the LLM book it is the very first attention table you ever build:
before any learning happens, each word is compared with every other word using a
[[inner-product|dot product]], and the 6-by-6 grid of scores tells you which words "look alike".

## 🟡 Mechanics (Practitioner)

### Definition
Stack the vectors as the rows of a matrix $X$. Then
$$
G = XX^\top,\qquad G_{ij} = \mathbf x^{(i)}\cdot\mathbf x^{(j)} = \sum_{k=1}^{d} x^{(i)}_k\,x^{(j)}_k .
$$
Symbol by symbol:
- $X \in \mathbb R^{T\times d}$ = a table with $T$ rows (the vectors) and $d$ columns (their coordinates).
- $X^\top$ = the [[matrix-transpose|transpose]] of $X$ (rows turned into columns), shape $d\times T$.
- $G = XX^\top$ = a $T\times T$ table (by [[matrix-multiplication]]).
- $G_{ij}$ = the entry in row $i$, column $j$: the dot product of vector $i$ with vector $j$.
- $x^{(i)}_k$ = coordinate number $k$ of vector $i$; the sum multiplies matching coordinates and adds them.

Two facts you can see right away:
- **Symmetric:** $G_{ij} = G_{ji}$, because a dot product doesn't care about order.
- **Diagonal = squared lengths:** $G_{ii} = \lVert\mathbf x^{(i)}\rVert^2$ (see [[norm]]).

### The book's Gram matrix: simplified self-attention (§3.3.2, pp. 61–63)
Inputs: Your [0.43, 0.15, 0.89], journey [0.55, 0.87, 0.66], starts [0.57, 0.85, 0.64],
with [0.22, 0.58, 0.33], one [0.77, 0.25, 0.10], step [0.05, 0.80, 0.55] ($T=6$, $d=3$).
```python
attn_scores = inputs @ inputs.T      # (6,3) @ (3,6) -> (6,6), same result as a double for loop
```
```
[[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
 [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
 [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
 [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
 [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
 [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]]
```
Checks you can do by hand:
- Mirror symmetry: entry [0,1] = entry [1,0] = 0.9544 (the dot product of "Your" and "journey").
- Diagonal: $G_{11} = 0.43^2 + 0.15^2 + 0.89^2 = 0.1849 + 0.0225 + 0.7921 = 0.9995$.
- "journey" and "starts" are almost the same vector, so $G_{23} = 1.4754$ is nearly as big as $G_{22} = 1.4950$.

The book then applies row-wise [[softmax]] to this matrix to get the [[attention-weights]] (p. 63).
The book points out that Python for loops are slow and that `inputs @ inputs.T` gives the same result (p. 62).

### Per-head Gram matrices: the batched-matmul demo (§3.6.2, pp. 88–89)
A 4-D tensor `a` of shape (batch, heads, tokens, head_dim) = (1, 2, 3, 4).
`a @ a.transpose(2, 3)` computes one 3×3 Gram matrix **per head** at once:
```
head 0: [[1.3208, 1.1631, 1.2879],      head 1: [[0.4391, 0.7003, 0.5903],
         [1.1631, 2.2150, 1.8424],               [0.7003, 1.3737, 1.0620],
         [1.2879, 1.8424, 2.0402]]               [0.5903, 1.0620, 0.9912]]
```
Both are symmetric. The book shows this equals `first_head @ first_head.T` and
`second_head @ second_head.T` done separately (p. 89). Check (ours):
$0.2745^2+0.6584^2+0.2775^2+0.8573^2 \approx 1.3208$. See [[batched-matrix-multiplication]].

### Trainable attention is *not* a Gram matrix
Once trainable weights arrive (§3.4), the scores are $QK^\top = (XW_q)(XW_k)^\top$. Since $W_q \ne W_k$
in general, this matrix is **not symmetric**: how much "journey" attends to "Your" can differ from how
much "Your" attends to "journey". The Gram matrix is the special case $W_q = W_k = I$ (the identity).
See [[scaled-dot-product-attention]].

## 🔴 Theory (Expert)
- **Positive semidefinite (PSD).** For any $\mathbf c \in \mathbb R^T$:
  $\mathbf c^\top G\,\mathbf c = \mathbf c^\top XX^\top\mathbf c = \lVert X^\top\mathbf c\rVert^2 \ge 0$.
  So all eigenvalues of $G$ are $\ge 0$. $G$ is positive *definite* exactly when the rows of $X$ are
  [[linear-independence|linearly independent]]. See [[symmetric-positive-definite-matrix]] and [[eigenvalues-and-eigenvectors]].
- **Rank.** $\operatorname{rank}(G) = \operatorname{rank}(X) \le \min(T, d)$. In the book's example $T=6$ but
  $d=3$, so the 6×6 attention-score matrix has rank at most 3: at least three of its eigenvalues are 0
  (we checked that $\det(X^\top X)\approx0.957\ne0$, so the rank is exactly 3). This low-rank structure is
  why attention with small head dimension is sometimes described as a low-rank interaction (see [[basis-and-rank]], [[low-rank-approximation]]).
- **Trace.** $\operatorname{tr}(G) = \sum_i\lVert\mathbf x^{(i)}\rVert^2$; for the book's inputs, $0.9995+1.4950+1.4570+0.4937+0.6654+0.9450 = 6.0556$ (see [[trace]]).
- **Same eigenvalues as $X^\top X$.** The nonzero eigenvalues of $XX^\top$ ($T\times T$) and $X^\top X$ ($d\times d$, the scatter matrix behind the [[data-covariance-matrix]]) are the same: they are the squared [[singular-value-decomposition|singular values]] of $X$. This is the trick behind computing [[principal-component-analysis|PCA]] in whichever space is smaller.
- **General inner products and kernels.** For any [[inner-product]] $\langle\cdot,\cdot\rangle$, $G_{ij} = \langle\mathbf x^{(i)},\mathbf x^{(j)}\rangle$ is a Gram matrix. Replacing the inner product with a kernel $k(\mathbf x^{(i)},\mathbf x^{(j)})$ gives the **kernel matrix** used by the [[kernel-trick]] and [[support-vector-machine|SVMs]]; a function is a valid kernel exactly when every such matrix is PSD (Mercer's condition; MML ch. 12). Unnormalized attention $\exp(\mathbf q\cdot\mathbf k/\sqrt{d_k})$ can be read through the same lens (outside the book).
- **Gram–Schmidt** (different object, same name): [[gram-schmidt-orthogonalization]] turns a basis into an orthonormal one; the Gram matrix of the result is the identity.

## Connections
- Builds on: [[inner-product]] (dot product), [[matrix]], [[matrix-transpose]], [[matrix-multiplication]], [[norm]]
- Leads to: [[symmetric-positive-definite-matrix]], [[attention-scores]], [[scaled-dot-product-attention]], [[kernel-trick]]
- Related: [[angle-between-vectors]], [[singular-value-decomposition]], [[principal-component-analysis]], [[batched-matrix-multiplication]], [[basis-and-rank]]
- Used by: [[self-attention]], [[multi-head-attention]], [[support-vector-machine]]
- People: [[sebastian-raschka]]

> **From [[llm-appA-introduction-to-pytorch]]:** §A.2.3 gives the smallest Gram-matrix example in the book. For $\mathbf X = \begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$, `tensor2d.matmul(tensor2d.T)` (or `tensor2d @ tensor2d.T`) prints `tensor([[14, 32], [32, 77]])` (book p. 261). Read $\mathbf X\mathbf X^\top$ as "multiply $\mathbf X$ by its [[matrix-transpose|transpose]]". Entry $(i,j)$ is the dot product of row $i$ with row $j$: $1\cdot1+2\cdot2+3\cdot3 = 14$, $1\cdot4+2\cdot5+3\cdot6 = 32$, $4\cdot4+5\cdot5+6\cdot6 = 77$. The result is symmetric ($32$ appears twice), as every Gram matrix is. The book shows the operation; the Gram-matrix reading is this wiki's annotation.

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.1–3.3.2 ("Understanding dot products" sidebar; `attn_scores = inputs @ inputs.T`; 6×6 score tensor), book pp. 58–64.
- Same PDF §3.6.2 (batched matrix multiplication demo `a @ a.transpose(2, 3)`), book pp. 88–89.
- Same PDF §3.4.2 (matrix form $QK^\top$, Fig. 3.18), book pp. 71–72.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2.3 (`.T`, `.matmul`, `@`; output `[[14, 32], [32, 77]]`), book pp. 260–261 (PDF pp. 282–283).
