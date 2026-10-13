---
title: Tensor (Rank and Shape)
type: concept
tags: [linear-algebra, pytorch, data-structures, llm]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-appA-introduction-to-pytorch, llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Tensor (Rank and Shape)

> **One-liner:** A tensor is a box of numbers with any number of axes: a single number, a list, a table, or a stack of tables, and so on.

## 🟢 Intuition (Dummy)
Start with things you know:
- A single number, like a temperature `2`. That is a **scalar**.
- A list of numbers, like `(3, 1, 3)`. That is a **[[vector]]**.
- A table with rows and columns. That is a **[[matrix]]**.
- A pile of tables, like the pages of a spreadsheet workbook. There is no special name, so
  people just say "3D tensor". Keep going and you get 4D, 5D, … tensors (book p. 258).

"Tensor" is the one word that covers all of these. The number of axes is called the
tensor's **rank** (or **order**) (book p. 258). To find the rank, count how many index
numbers you need to point at one entry: none for a scalar, one for a list position,
two (row and column) for a table, three for "page, row, column".

A trap with words: a "3D vector" such as `(3, 1, 3)` has 3 *entries* but only *one* axis,
so it is still a rank-1 tensor (Figure A.6, book p. 258).

In practice, a tensor is just a **data container**. Each axis usually means something.
In an LLM a batch of text looks like this: axis 1 = which example in the batch,
axis 2 = which token position, and later axis 3 = which embedding coordinate.

NumPy users: a PyTorch tensor is basically a NumPy array with two superpowers added,
automatic gradients and GPU support (book p. 258).

## 🟡 Mechanics (Practitioner)

### Notation for each rank
- Scalar: $s\in\mathbb{R}$. Rank 0, no index.
- Vector: $\mathbf{v}\in\mathbb{R}^{n}$, entries $v_i$. Rank 1, one index.
- Matrix: $\mathbf{A}\in\mathbb{R}^{m\times n}$, entries $A_{ij}$. Rank 2, two indices.
- Rank-3 tensor: $\mathcal{T}\in\mathbb{R}^{d_1\times d_2\times d_3}$, entries $\mathcal{T}_{ijk}$. Three indices.

How to read "$\mathcal{T}\in\mathbb{R}^{d_1\times d_2\times d_3}$": "$\mathcal{T}$ is a block of real numbers
($\mathbb{R}$) with $d_1$ slots along axis 1, $d_2$ along axis 2 and $d_3$ along axis 3". The list
$(d_1, d_2, d_3)$ is the **shape**; its length (3) is the **rank**; the product $d_1d_2d_3$ is the
total number of entries (`.numel()`).

### Creating tensors (Listing A.1, book p. 259)
```python
import torch
tensor0d = torch.tensor(1)                         # scalar,  shape []
tensor1d = torch.tensor([1, 2, 3])                 # vector,  shape [3]
tensor2d = torch.tensor([[1, 2], [3, 4]])          # matrix,  shape [2, 2]
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])        # 3D,      shape [2, 2, 2]
```
The nesting depth of the Python list sets the rank (the shapes in the comments follow from
the nesting; the book does not print them).

### Data types (book pp. 259–260)
- Python ints become `torch.int64`; Python floats become `torch.float32`.
- float32 is the default because it is precise enough for most deep learning, needs less memory
  and compute than 64-bit, and GPUs are optimised for 32-bit maths (book p. 259).
- Convert with `.to`: `tensor1d.to(torch.float32)`. (The same `.to` also moves a tensor to a
  device, e.g. `.to("cuda")`, book p. 280.)

### Everyday operations, with the book's numbers (book pp. 260–261)
Take
$$\mathbf{X}=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix},\qquad \texttt{X.shape}=\texttt{torch.Size([2, 3])}.$$

| Operation | Code | Result |
|---|---|---|
| Reshape | `X.reshape(3, 2)` or `X.view(3, 2)` | $\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}$ |
| Transpose | `X.T` | $\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}$ |
| Matrix product | `X.matmul(X.T)` or `X @ X.T` | $\begin{bmatrix}14&32\\32&77\end{bmatrix}$ |

**Reshape vs transpose.** Both give a 3×2 result, but the entries land in different places.
Reshape keeps the flat reading order 1,2,3,4,5,6 and just regroups it; transpose flips across the
diagonal:
$$(\mathbf{X}^\top)_{ij}=X_{ji}$$
- $(\mathbf{X}^\top)_{ij}$ = entry in row $i$, column $j$ of the transpose; $X_{ji}$ = entry in row $j$, column $i$ of the original. Rows become columns. See [[matrix-transpose]] and [[tensor-reshaping-and-transposing]].

**Checking the product by hand.**
$$(\mathbf{X}\mathbf{X}^\top)_{ij}=\sum_{k=1}^{3}X_{ik}X_{jk}$$
- $i, j$ pick two rows of $\mathbf{X}$; $\sum_{k=1}^{3}$ adds over the 3 columns; so each entry is the [[inner-product|dot product]] of row $i$ with row $j$.
- $(1,1)$: $1\cdot1+2\cdot2+3\cdot3=14$; $(1,2)$: $1\cdot4+2\cdot5+3\cdot6=32$; $(2,2)$: $4\cdot4+5\cdot5+6\cdot6=77$.
- Shapes: $[2,3]\times[3,2]\to[2,2]$. This "rows dotted with rows" pattern is exactly how attention scores $\mathbf{Q}\mathbf{K}^\top$ are computed later (see [[gram-matrix]], [[attention-scores]]).

`.view` vs `.reshape`: the book calls `.view()` "the more common command for reshaping" (book p. 260).
`.view` needs the data to be contiguous in memory and fails otherwise; `.reshape` always works and
copies the data if it must (book p. 261). Both exist because PyTorch first followed the Lua Torch
syntax and later, by popular request, added NumPy-like syntax; PyTorch adopts most of the NumPy
array API (book pp. 258, 260–261).

### Tensors in the LLM pipeline (Ch. 2)
The book's first real tensors are the input–target batches of token IDs (book pp. 37–41):

```python
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4,
                                  stride=4, shuffle=False)
inputs, targets = next(iter(dataloader))
# inputs:  tensor([[   40,   367,  2885,  1464],
#                  [ 1807,  3619,   402,   271], ...])   shape [8, 4]
```
- Rank 2, shape $[B, T]=[8, 4]$: $B$ = [[mini-batch|batch size]], $T$ = [[context-length]] (tokens per row), entries = integer [[token-id|token IDs]].
- After the [[embedding-layer]] each ID becomes a 256-dimensional vector, so the tensor grows to rank 3: shape `[8, 4, 256]` = (batch, tokens, [[embedding-dimension]]) (book p. 47).
- Typical shapes later on: attention weights `[batch, heads, tokens, tokens]` (rank 4) and output [[logits]] `[batch, tokens, vocab_size]`.

The book's own definition at this point: a tensor is "a multidimensional array" (book p. 37).

### Devices (book pp. 279–280)
```python
tensor_1 = torch.tensor([1., 2., 3.]).to("cuda")
tensor_2 = torch.tensor([4., 5., 6.]).to("cuda")
print(tensor_1 + tensor_2)   # tensor([5., 7., 9.], device='cuda:0')
```
A tensor lives on one device (CPU or a GPU like `cuda:0`), operations run there, and all
operands must be on the same device or PyTorch raises a `RuntimeError` (book p. 280).
Elementwise addition here is $\mathbf{a}+\mathbf{b}=(a_1+b_1,\ a_2+b_2,\ a_3+b_3)=(5,7,9)$.

## 🔴 Theory (Expert)
- **Two meanings of "tensor".** In mathematics and physics, a tensor is a multilinear map
  (an element of $V_1\otimes\cdots\otimes V_k$) whose components transform in a specific way under a
  change of basis. In deep learning, "tensor" means only the component array in a fixed basis, i.e. an
  element of $\mathbb{R}^{d_1\times\cdots\times d_k}$, which is itself a [[vector-space]] of dimension
  $\prod_i d_i$. PyTorch tensors are the second kind.
- **Rank ≠ matrix rank.** Tensor rank/order (number of axes) is unrelated to the linear-algebra
  rank of a matrix (number of linearly independent columns; see [[basis-and-rank]]). A $1000\times1000$
  matrix is always a rank-2 tensor, but its matrix rank can be anything from 0 to 1000. (Multilinear
  algebra also has a third notion, CP-rank, which is again different.)
- **Memory layout and strides.** A tensor is a flat storage buffer plus a shape and a stride per axis;
  element $(i_1,\dots,i_k)$ sits at offset $\sum_a i_a s_a$. In row-major (C) order on a $m\times n$ matrix
  the strides are $(n,1)$, so element $(i,j)$ is at $in+j$. Reshape to $p\times q$ keeps flat position
  $k$ and moves it to $(\lfloor k/q\rfloor, k\bmod q)$. A transpose just swaps strides without moving data,
  which makes the result **non-contiguous**; that is why `.view` can fail after `.T` while `.reshape`
  silently copies (book p. 261).
- **Batched linear algebra.** `@` on rank ≥ 3 tensors treats the leading axes as batch axes and
  multiplies the last two, with [[broadcasting]] over mismatched leading dimensions. This is the engine
  of [[batched-matrix-multiplication]] in multi-head attention.
- **Precision.** float32 uses 4 bytes per entry, float64/int64 use 8, which is why float32 halves
  memory relative to float64 (general fact, not stated in the book). See [[floating-point-precision]]
  and [[parameter-count-and-memory]].
- **Autograd hook.** Tensors with `requires_grad=True` record the operations applied to them, building
  the [[computational-graph]] used by [[automatic-differentiation]] (Appendix A.3–A.4).

### From llm-ch03-coding-attention-mechanisms
Ch. 3 uses tensors of rank 2, 3 and 4 for attention:
- **Rank 2:** the input matrix of 6 tokens × 3 dimensions for "Your journey starts with one step" (book p. 57) and the $6 \times 6$ score and weight matrices (book pp. 62–63).
- **Rank 3:** `batch = torch.stack((inputs, inputs), dim=0)` gives shape `(2, 6, 3)` — 2 texts, 6 tokens each, 3 dimensions per token (book p. 80). [[causal-attention]] returns `(2, 6, 2)`.
- **Rank 4:** [[multi-head-attention]] reshapes `(b, num_tokens, d_out)` into `(b, num_heads, num_tokens, head_dim)` with `.view` and `.transpose(1, 2)` (book pp. 86–89; see [[tensor-reshaping-and-transposing]]).
- Listing 3.3 also introduces `register_buffer`: a tensor (the causal mask) that moves with the module to the GPU but is not a trained parameter (book p. 81; see [[causal-attention-mask]]).

## Connections
- Builds on: [[vector]], [[matrix]], [[vector-space]]
- Operations: [[matrix-multiplication]], [[matrix-transpose]], [[tensor-reshaping-and-transposing]], [[broadcasting]], [[batched-matrix-multiplication]], [[inner-product]], [[gram-matrix]]
- Leads to: [[token-embedding]], [[embedding-layer]], [[attention-scores]], [[logits]], [[mini-batch]], [[pytorch-dataset-and-dataloader]]
- Tools: [[pytorch]], [[gpu-computing]], [[floating-point-precision]], [[automatic-differentiation]]
- Contrast: [[basis-and-rank]] (matrix rank is a different "rank")
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2 "Understanding tensors" (A.2.1–A.2.3), book pp. 258–261 (ranks, Figure A.6, Listing A.1, dtypes, reshape/view/T/matmul with outputs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.1, book p. 252 (tensor library as one of PyTorch's three components).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book pp. 37–41 (tensor as multidimensional array; input/target tensors of shape 8×4); §2.7–2.8, book p. 47 (shape [8, 4, 256]).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9.1, book pp. 279–280 (devices, `.to("cuda")`, same-device rule).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.1, book p. 57 (6×3 input); §3.3.2, book pp. 62–63; §3.5.3, book pp. 80–82 (`torch.stack`, shape (2, 6, 3), `register_buffer`); §3.6.2, book pp. 86–89 (4-D head tensors).
