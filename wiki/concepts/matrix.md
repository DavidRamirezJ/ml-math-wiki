---
title: Matrix
type: concept
tags: [linear-algebra, foundations]
difficulty: beginner
sources: [mml-ch02-linear-algebra, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-06-04
updated: 2026-10-13
---

# Matrix

> **One-liner:** A matrix is a rectangular grid of numbers — and that grid can both *store a system of equations* and *act as a function that transforms vectors*.

## 🟢 Intuition (Dummy)
A matrix is just a table of numbers with rows and columns, like a spreadsheet. Why bother?
Two reasons. First, it's a tidy way to write down a whole [[system-of-linear-equations]] at once.
Second — and this is the deep part — a matrix is also a **machine**: feed it a [[vector]] and it
spits out a new vector ([[linear-mapping]]). So one grid of numbers does double duty: it *stores
data* and it *does things to data*. Keep straight which role a given matrix is playing.

## 🟡 Mechanics (Practitioner)
An $(m,n)$ matrix has $m$ rows and $n$ columns, $A \in \mathbb{R}^{m\times n}$, entries $a_{ij}$.

**Addition** (same shape, element-wise): $(A+B)_{ij} = a_{ij}+b_{ij}$.

**Multiplication** $C = AB$ with $A\in\mathbb{R}^{m\times n}, B\in\mathbb{R}^{n\times k}, C\in\mathbb{R}^{m\times k}$:
$$ c_{ij} = \sum_{l=1}^{n} a_{il}\,b_{lj} $$
Row $i$ of $A$ "dot" column $j$ of $B$. **Neighboring dimensions must match** ($n=n$); inner
dimension collapses, outer dimensions survive. ⚠️ Not commutative: $AB \neq BA$ in general
(Example 2.3: same $A,B$ give a $2\times2$ *and* a $3\times3$ result).

**Animation — each entry is a row · column dot product** (Example 2.3, $C = AB$):

![Matrix multiplication animation: each entry c_ij is the dot product of row i of A and column j of B](../../assets/matrix-multiply.gif)

The amber **row of $A$** times the blue **column of $B$** accumulates, term by term, into the
green **target cell of $C$** — repeated for all four entries. Regenerate with
`python3 assets/matrix-multiply-anim.py`.

**Identity** $I_n$: 1's on the diagonal, 0's elsewhere; $I_mA = AI_n = A$.

**Inverse** $A^{-1}$ (square only): the $B$ with $AB = I = BA$. Exists ⇔ $A$ is *regular /
invertible / nonsingular*; when it exists it is unique. For $2\times2$:
$$ A^{-1} = \frac{1}{a_{11}a_{22}-a_{12}a_{21}}\begin{bmatrix} a_{22} & -a_{12}\\ -a_{21} & a_{11}\end{bmatrix}, \quad \text{iff } a_{11}a_{22}-a_{12}a_{21}\neq 0 $$
(that denominator is the *determinant*, Ch. 4).

**Transpose** $A^\top$: $b_{ij}=a_{ji}$ (flip rows↔columns). **Symmetric** if $A=A^\top$ (square only).

**In ML:** weight matrices in neural nets, covariance matrices, data matrices (rows = samples).
⚠️ Element-wise product $c_{ij}=a_{ij}b_{ij}$ is the **Hadamard product**, *not* matrix multiplication.

### In LLMs (Raschka, *Build a LLM from Scratch*)
In an LLM, nearly every object is a matrix (or a stack of matrices, a [[tensor]]). Both roles from
the 🟢 tier show up: matrices that **store data** (a batch of token vectors) and matrices that
**do things** (weight matrices).

> ⚠️ Notation difference (not a contradiction): this page and MML write column vectors and
> $A\mathbf{x}$. The LLM book follows PyTorch and writes **row vectors** multiplied from the left,
> $\mathbf{x}W$ or $XW$. The two are transposes of each other: $\mathbf{x}W = (W^\top\mathbf{x}^\top)^\top$.

**1. Tensors, transpose and `@` in PyTorch (§A.2, book pp. 260–261).**
```python
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])   # shape [2, 3]
tensor2d.reshape(3, 2)       # [[1,2],[3,4],[5,6]]
tensor2d.T                   # [[1,4],[2,5],[3,6]]  (transpose != reshape!)
tensor2d @ tensor2d.T        # [[14,32],[32,77]]
```
Check with $c_{ij}=\sum_l a_{il}b_{lj}$: $c_{11} = 1\cdot1+2\cdot2+3\cdot3 = 14$,
$c_{12} = 1\cdot4+2\cdot5+3\cdot6 = 32$, $c_{22} = 16+25+36 = 77$. `.matmul` and `@` are the same
operation. The reshape and the transpose are both $3\times2$, but the numbers sit in different
places. A *design matrix* is the same idea for data: `X_train` is $5\times2$, rows = examples,
columns = features (§A.6, book p. 270).

**2. The embedding table is a matrix you index by row (§2.7–2.8).** `torch.nn.Embedding(6, 3)`
holds $W\in\mathbb{R}^{6\times3}$. Token ID 3 returns row 3: $[-0.4015,\ 0.9666,\ -1.1481]$
(book pp. 42–43). A batch of IDs `[2, 3, 5, 1]` returns the $4\times3$ matrix of those rows. GPT-2
uses $50{,}257\times768$ (tokens) and $1024\times768$ (positions). See [[embedding-layer]].

**3. Weight matrices = parameters.** "In a neural network layer that is represented by a
2,048 × 2,048–dimensional matrix (or tensor) of weights, each element of this matrix is a
parameter", so $2{,}048\times2{,}048 = 4{,}194{,}304$ parameters (book p. 94). In the appendix-A
network, `model.layers[0].weight.shape` is `torch.Size([30, 50])` and the whole 50→30→20→3 net
has $(50\cdot30+30)+(30\cdot20+20)+(20\cdot3+3) = 2{,}213$ trainable parameters (book p. 267).
See [[model-parameters]].

**4. Attention is three matrix products (§3.3–3.4).** Stack the 6 token vectors of "Your journey
starts with one step" into $X\in\mathbb{R}^{6\times3}$ (row 2, "journey", is $[0.55, 0.87, 0.66]$). Then
$$\Omega = XX^\top,\qquad A = \operatorname{softmax}_{\text{row}}(\Omega),\qquad Z = AX$$
- $X$ = input matrix, one row per token ($T=6$ rows, $d=3$ columns).
- $X^\top$ = its transpose ($3\times6$), so $XX^\top$ is $6\times6$: entry $(i,j)$ is the [[inner-product|dot product]] of token $i$ with token $j$.
- $\Omega$ = attention scores; it is **symmetric** here, e.g. $\Omega_{12} = \Omega_{21} = 0.9544$ (book pp. 62–63).
- $\operatorname{softmax}_{\text{row}}$ = apply [[softmax]] to each row, so every row of $A$ sums to 1.
- $Z = AX$ = the $6\times3$ matrix of [[context-vector|context vectors]]; row 2 is $[0.4419, 0.6515, 0.5683]$ (book p. 63).

With trainable weights (§3.4): $Q = XW_q,\ K = XW_k,\ V = XW_v$ with $W_q\in\mathbb{R}^{3\times2}$, and
$Z = \operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$. With seed 123, $\mathbf{q}^{(2)} = \mathbf{x}^{(2)}W_q = [0.4306,\ 1.4551]$
(book p. 66). See [[query-key-value]], [[scaled-dot-product-attention]].

**5. Triangular matrices as masks (§3.5).** `torch.tril(torch.ones(6, 6))` keeps the lower
triangle (row $i$ = $i$ ones, then zeros); `torch.triu(..., diagonal=1)` marks the entries strictly
above the diagonal, which are filled with $-\infty$ before the softmax (book pp. 75–77). See
[[causal-attention-mask]].

**6. Batched matrix multiplication (§3.6).** For a 4-D tensor of shape
(batch, heads, tokens, head_dim) = (1, 2, 3, 4), `a @ a.transpose(2, 3)` multiplies the last two
axes **for every head at once**. Head 0's result starts $[[1.3208, 1.1631, 1.2879], \dots]$, the same
as `first_head @ first_head.T` (book p. 89). Check: $1.3208 \approx 0.2745^2+0.6584^2+0.2775^2+0.8573^2$.
See [[batched-matrix-multiplication]].

**7. PyTorch stores `nn.Linear` weights transposed.** `nn.Linear(d_in, d_out).weight` has shape
$(d_{out}, d_{in})$ and computes $\mathbf{y} = \mathbf{x}W^\top + \mathbf{b}$ (Exercise 3.1 hint, book p. 73). OpenAI's
GPT-2 checkpoint stores $(d_{in}, d_{out})$, so loading needs `.T` everywhere, and the fused query/key/value
weight `c_attn` ($768\times2304$) is cut into three $768\times768$ blocks with `np.split(..., 3, axis=-1)`
(book p. 165). See [[matrix-transpose]].

**8. Matrix sizes in GPT-2 124M (§4.1).** Token table $50{,}257\times768$, position table
$1{,}024\times768$, output head `nn.Linear(768, 50257, bias=False)` (book pp. 95–97). The output
head maps each 768-dim hidden row to 50,257 [[logits]].

**9. Speed (§A.9).** Exercise A.4 (book p. 282) asks at what size a GPU beats a CPU at `a @ b`.
An $n\times n$ product costs $O(n^3)$ multiply-adds, all independent per entry, which is why
massively parallel [[gpu-computing|GPUs]] win once $n$ is large enough to hide the transfer overhead.

**10. Low-rank updates (§E.1, book pp. 322–324).** [[low-rank-adaptation|LoRA]] replaces a full update
$\Delta W$ by a product of two thin matrices:
$$W_{\text{updated}} = W + AB,\qquad A\in\mathbb{R}^{d\times r},\ B\in\mathbb{R}^{r\times d}$$
- $W$ = frozen pretrained weight; $A, B$ = small trainable matrices; $r$ = the **inner dimension** (a [[hyperparameter]]).
- Because matrix multiplication is distributive, $\mathbf{x}(W + AB) = \mathbf{x}W + \mathbf{x}AB$, so the two paths can be kept separate.

## 🔴 Theory (Expert)
Key algebraic identities (all from §2.2.2):
$$ (AB)^{-1}=B^{-1}A^{-1}, \quad (AB)^\top = B^\top A^\top, \quad (A^\top)^\top = A, \quad (A+B)^\top = A^\top+B^\top $$
Note $(A+B)^{-1} \neq A^{-1}+B^{-1}$ in general. If $A$ invertible, so is $A^\top$, and $(A^{-1})^\top = (A^\top)^{-1} =: A^{-\top}$.

- Matrix multiplication is **associative** and **distributive** but not commutative.
- The set of invertible $n\times n$ matrices under multiplication forms the **general linear
  group** $GL(n,\mathbb{R})$ (Def 2.8) — a [[vector-space|group]] that is *not* Abelian.
- A matrix represents either a **collection of vectors** (columns) or a **[[linear-mapping]]**
  (via Def 2.19, a transformation matrix w.r.t. ordered bases). Same object, two readings.
- The product $A\mathbf{x}$ is a **linear combination of the columns of $A$** weighted by the
  entries of $\mathbf{x}$ — the bridge to [[linear-independence]] and [[basis-and-rank]].
- **LLM view of the same facts.** (i) Dually, $\mathbf{x}^\top W$ is a linear combination of the
  *rows* of $W$; with a one-hot $\mathbf{x}$ this picks one row, which is the
  [[one-hot-encoding]] ⇔ embedding-lookup equivalence. (ii) $XX^\top$ is a [[gram-matrix]]:
  symmetric and positive semidefinite, which is why the untrained score matrix $\Omega$ in §3.3
  is symmetric; $QK^\top = XW_qW_k^\top X^\top$ is generally *not* symmetric. (iii) An attention
  weight matrix $A$ is **row-stochastic** (non-negative, rows sum to 1), so each row of $Z=AX$ is a
  [[convex-combination]] of the rows of $X$. (iv) Block multiplication,
  $X[W^{(1)}\,|\,\cdots\,|\,W^{(h)}] = [XW^{(1)}\,|\,\cdots\,|\,XW^{(h)}]$, justifies one big
  $W_q$ split into heads ([[weight-splitting-multi-head-attention]]) and the fused `c_attn` split.
- **Rank bound.** $\operatorname{rk}(AB)\le\min(\operatorname{rk}A,\operatorname{rk}B)\le r$, so LoRA's
  update lives in a rank-$\le r$ set; by Eckart–Young the best rank-$r$ approximation of a given
  $\Delta W$ is its truncated [[singular-value-decomposition]] (classical analogue, not in the book).
  Associativity, $(\mathbf{x}A)B = \mathbf{x}(AB)$, lets the code compute `x @ A @ B` without ever forming
  the $d\times d$ product (see [[low-rank-approximation]]).

## Connections
- Builds on: [[vector]], [[vector-space]]
- Leads to: [[gaussian-elimination]], [[linear-mapping]], [[basis-and-rank]], [[tensor]], [[matrix-multiplication]], [[matrix-transpose]]
- Used by: [[system-of-linear-equations]], all of ML (weights, covariances), [[embedding-layer]], [[self-attention]], [[scaled-dot-product-attention]], [[batched-matrix-multiplication]], [[linear-layer]], [[low-rank-adaptation]]
- Related: [[gram-matrix]], [[causal-attention-mask]], [[model-parameters]], [[one-hot-encoding]]
- People: —

## Sources
- `raw/mml-book.pdf` §2.2, Defs 2.1–2.5, printed pp. 22–26 (PDF pp. 27–31). Validated 2026-06-04.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book pp. 41–44 and §2.8, book pp. 43–49 (embedding weight matrix, row lookup).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book pp. 57–64 ($XX^\top$, row softmax, $AX$); §3.4, book pp. 65–73 ($XW_q$, `nn.Linear` transposed weights); §3.5, book pp. 75–81 (`tril`/`triu` masks); §3.6, book pp. 86–90 (batched matmul, weight splits).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 94–97 (2,048×2,048 example, GPT-2 matrix shapes); §5.5, book pp. 164–166 (OpenAI weight transposes, `np.split`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2, book pp. 258–261 (tensors, `.T`, `@`); §A.5, book p. 267 (`[30, 50]` weight, 2,213 params); §A.6, book p. 270 (design matrix); §A.9, book p. 282 (Exercise A.4).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1, book pp. 322–324 ($W + AB$, distributive law).
