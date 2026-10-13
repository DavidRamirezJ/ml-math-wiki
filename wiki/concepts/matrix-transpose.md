---
title: Matrix Transpose
type: concept
tags: [linear-algebra, matrix, pytorch, llm]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-ch05-pretraining-on-unlabeled-data, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Matrix Transpose

> **One-liner:** Transposing a matrix flips it over its diagonal, so its rows become its columns and its columns become its rows.

## 🟢 Intuition (Dummy)
Picture a spreadsheet where each row is a person and each column is a
measurement (height, weight, age). Now tilt it so each *row* is a measurement
and each *column* is a person. Same numbers, same pairings — you just turned
the table on its side. That's a transpose.

It's not the same as "reshaping." Reshaping pours the numbers, in reading
order, into a box of a different shape. Transposing keeps every number tied
to its row-and-column partner and just swaps which one counts as the row.
Raschka's book makes this point with a tiny example where both operations
give a 3×2 result — but with the numbers in different places (book p. 261).

Why care? In LLMs, you constantly need "every row of this matrix dotted with
every row of that matrix" — for example, every query against every key in
[[attention-mechanism|attention]]. The transpose is what lines the rows up so
plain matrix multiplication can do it in one shot.

## 🟡 Mechanics (Practitioner)

### Definition
$$(\mathbf X^\top)_{ij}=X_{ji},\qquad \mathbf X\in\mathbb R^{m\times n}\ \Rightarrow\ \mathbf X^\top\in\mathbb R^{n\times m}.$$
Read it symbol by symbol:
- $\mathbf X$ — the original [[matrix]], with $m$ rows and $n$ columns ($\mathbb R^{m\times n}$ = "an $m$-by-$n$ grid of real numbers").
- $^\top$ — the "transpose" symbol (a superscript T).
- $(\mathbf X^\top)_{ij}$ — the entry in row $i$, column $j$ of the transposed matrix.
- $X_{ji}$ — the entry of the original in row $j$, column $i$: the indices swap.
- The shape flips: $m\times n$ becomes $n\times m$.

### The book's example (§A.2.3, book pp. 260–261)
```python
tensor2d = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])     # shape [2, 3]
tensor2d.reshape(3, 2)   # [[1, 2], [3, 4], [5, 6]]   -- reading order kept
tensor2d.T               # [[1, 4], [2, 5], [3, 6]]   -- rows become columns
tensor2d @ tensor2d.T    # [[14, 32], [32, 77]]
```
$$\mathbf X=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad
\mathbf X^\top=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix},\qquad
\mathbf X\mathbf X^\top=\begin{bmatrix}14&32\\32&77\end{bmatrix}.$$
- `.T` "flips it across its diagonal." Compare with `.reshape(3, 2)`: same
  shape, different arrangement — **transpose ≠ reshape**.
- `.matmul(...)` and `@` both do [[matrix-multiplication]].
- Checking $\mathbf X\mathbf X^\top$ by hand (our check): each entry is an
  [[inner-product|inner (dot) product]] of two **rows** of $\mathbf X$:
  - $(1,1)$: $1\cdot1+2\cdot2+3\cdot3=14$
  - $(1,2)=(2,1)$: $1\cdot4+2\cdot5+3\cdot6=32$
  - $(2,2)$: $4\cdot4+5\cdot5+6\cdot6=77$
  - Shapes: $[2,3]\times[3,2]\to[2,2]$.

  This "rows dotted with rows" matrix is the [[gram-matrix]] of $\mathbf X$.

### Where LLMs use it
**1. Attention scores (§3.4, book pp. 66–72).** With queries
$\mathbf Q\in\mathbb R^{T\times d}$ and keys $\mathbf K\in\mathbb R^{T\times d}$
(one row per token), all pairwise scores are
$$\boldsymbol\Omega=\mathbf Q\mathbf K^\top\in\mathbb R^{T\times T},\qquad \Omega_{ij}=\mathbf q^{(i)}\cdot\mathbf k^{(j)}.$$
$T$ = number of tokens (6 in the book's "Your journey starts with one step"
example); $d$ = the query/key size (2 in the toy example); $\Omega_{ij}$ = how
much token $i$ "looks at" token $j$. In code: `attn_scores = queries @ keys.T`
([[scaled-dot-product-attention]], [[attention-scores]]). For batched
multi-head tensors the same idea uses `.transpose(2, 3)` to swap only the
last two axes (§3.6; see [[tensor-reshaping-and-transposing]]).

**2. Loading OpenAI's GPT-2 weights (§5.5, book pp. 165–166).** OpenAI's
checkpoint stores each linear layer's weight as (in, out) for $\mathbf y=\mathbf x\mathbf W+\mathbf b$.
PyTorch's `nn.Linear` stores its weight as **(out, in)** and computes
$$\mathbf y=\mathbf x\mathbf W^\top+\mathbf b.$$
Here $\mathbf x$ = the input row vector, $\mathbf W$ = the stored weight
(out×in), $\mathbf b$ = the [[bias-vector]], $\mathbf y$ = the output. So every
weight matrix gets a `.T` on the way in:
```python
q_w, k_w, v_w = np.split(
    (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
gpt.trf_blocks[b].att.W_query.weight = assign(
    gpt.trf_blocks[b].att.W_query.weight, q_w.T)
# ... same .T for W_key, W_value, out_proj, and both feed-forward layers
```
The biases are 1-D vectors, so they need **no** transpose. For GPT-2 124M,
the fused `c_attn` weight is $768\times2304$ and splits into three
$768\times768$ pieces (shapes derived from the config, not printed by the
book). The book's `assign` helper raises an error on any shape mismatch — a
safety net that catches a forgotten `.T` on non-square layers such as the
$768\to3072$ feed-forward weight. (Square $768\times768$ layers would *not*
be caught: a missing `.T` there gives wrong weights silently, which shows up
only as incoherent generated text — our observation; the book says only that
`assign` alerts on mismatches and a wrong mapping yields incoherent text.) The PyTorch (out, in) layout is also the
hint for Exercise 3.1 (§3.4, p. 73; see [[linear-layer]]).

**3. Weight tying (§5.5, p. 166).** GPT-2 reuses the token-embedding matrix
$\mathbf W_{\text{te}}\in\mathbb R^{50257\times768}$ as the output layer:
logits $=\mathbf h\,\mathbf W_{\text{te}}^\top\in\mathbb R^{50257}$, where
$\mathbf h\in\mathbb R^{768}$ is the final hidden state ([[weight-tying]]). The
transpose turns "token → vector" into "vector → score per token."

## 🔴 Theory (Expert)
- **Algebraic rules.** $(\mathbf A^\top)^\top=\mathbf A$;
  $(\mathbf A+\mathbf B)^\top=\mathbf A^\top+\mathbf B^\top$;
  $(c\mathbf A)^\top=c\mathbf A^\top$; and the order-reversing
  $(\mathbf A\mathbf B)^\top=\mathbf B^\top\mathbf A^\top$. The last one explains
  the two layouts above: $(\mathbf x\mathbf W^\top)^\top=\mathbf W\mathbf x^\top$,
  so "row vector times $\mathbf W^\top$" and "matrix times column vector" are
  the same [[linear-mapping]].
- **Dot product as a transpose.** For column vectors,
  $\mathbf u\cdot\mathbf v=\mathbf u^\top\mathbf v$ — the standard
  [[inner-product]] on $\mathbb R^n$.
- **Symmetry and Gram matrices.** $(\mathbf X\mathbf X^\top)^\top=\mathbf X\mathbf X^\top$,
  so the book's $\begin{bmatrix}14&32\\32&77\end{bmatrix}$ must be symmetric. It is also
  positive semidefinite: $\mathbf v^\top\mathbf X\mathbf X^\top\mathbf v=\lVert\mathbf X^\top\mathbf v\rVert^2\ge0$
  ([[symmetric-positive-definite-matrix]]). Attention's $\mathbf Q\mathbf K^\top$
  is **not** symmetric in general, because $\mathbf W_q\neq\mathbf W_k$ — token
  $i$ can attend to $j$ more than $j$ attends to $i$.
- **The adjoint view.** The transpose is the matrix of the adjoint map:
  $\langle\mathbf A\mathbf x,\mathbf y\rangle=\langle\mathbf x,\mathbf A^\top\mathbf y\rangle$.
  This is why transposes appear everywhere in backpropagation: for
  $\mathbf Y=\mathbf X\mathbf W$ with upstream gradient $\bar{\mathbf Y}$, the
  gradients are $\bar{\mathbf X}=\bar{\mathbf Y}\mathbf W^\top$ and
  $\bar{\mathbf W}=\mathbf X^\top\bar{\mathbf Y}$
  ([[automatic-differentiation]], [[backpropagation]]).
- **Memory layout.** In PyTorch, `.T`/`.transpose` does not move data; it
  swaps the *strides* and returns a view. The result is usually
  non-contiguous, which is why `.view()` can fail after a transpose and the
  book calls `.contiguous()` before reshaping attention heads (§3.6; §A.2.3
  notes that `.view()` needs contiguous memory).
- **Rank is preserved.** $\operatorname{rank}(\mathbf X^\top)=\operatorname{rank}(\mathbf X)$
  (row rank = column rank; [[basis-and-rank]]).

> The algebraic rules, adjoint/backprop formulas, stride explanation, and the
> (in, out) vs (out, in) rationale are standard background; the book shows the
> `.T` calls but does not derive them.

## Connections
- Builds on: [[matrix]], [[vector]], [[inner-product]]
- Leads to: [[gram-matrix]], [[matrix-multiplication]], [[tensor-reshaping-and-transposing]], [[symmetric-positive-definite-matrix]]
- Used by: [[scaled-dot-product-attention]], [[attention-scores]], [[multi-head-attention]], [[linear-layer]], [[pretrained-weights]], [[weight-tying]], [[backpropagation]]
- Related: [[linear-mapping]], [[basis-and-rank]], [[tensor]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2.3 "Common PyTorch tensor operations" (`.T`, `.reshape`, `.matmul`, `@`), book pp. 260–261 (PDF pp. 282–283).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5 "Loading pretrained weights from OpenAI" (Listing 5.5, `.T` on weights, weight tying), book pp. 164–166 (PDF pp. 186–188).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (`queries @ keys.T`, `nn.Linear` stores the weight transposed), book pp. 66–73; §3.6 (`.transpose(1, 2)`, `.contiguous()`), book pp. 86–89.
