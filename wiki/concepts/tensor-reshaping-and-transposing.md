---
title: Tensor Reshaping and Transposing (view, reshape, transpose)
type: concept
tags: [tensors, pytorch, linear-algebra, attention]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-ch03-coding-attention-mechanisms, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Tensor Reshaping and Transposing (view, reshape, transpose)

> **One-liner:** Reshaping regroups the same numbers into a new shape without moving them; transposing swaps axes so the numbers really change places.

## 🟢 Intuition (Dummy)
Take six eggs lined up in a row: 1, 2, 3, 4, 5, 6.

- **Reshape** is putting them in an egg carton. You can use a carton with 2 rows of 3, or 3 rows of 2. The eggs go in **in the same reading order** (left to right, top to bottom). You only changed how they're grouped.
- **Transpose** is turning the carton on its side. What used to be a row is now a column. Eggs that were next to each other in a row are now stacked in a column.

Both can give you a "3 by 2" carton, but the eggs sit in different spots. That's the
single most important thing on this page: **reshape ≠ transpose**.

In LLM code you use both all the time. [[multi-head-attention]] reshapes one wide
vector into several small "head" vectors, then transposes so the heads line up for
[[batched-matrix-multiplication]], and at the end undoes both steps.

## 🟡 Mechanics (Practitioner)

### The book's example (§A.2.3, book pp. 260–261)
$$\mathbf{X} = \texttt{tensor2d} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \qquad \text{shape } [2, 3]$$
```python
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2d.shape              # torch.Size([2, 3])
tensor2d.reshape(3, 2)      # [[1, 2], [3, 4], [5, 6]]
tensor2d.view(3, 2)         # [[1, 2], [3, 4], [5, 6]]   same result
tensor2d.T                  # [[1, 4], [2, 5], [3, 6]]   different!
```
Both `reshape(3, 2)` and `.T` give a $3\times 2$ result, but the entries are in different places. The book describes `.T` as "flipping it across its diagonal" and stresses that this "is not the same" as reshaping (book p. 261).

### Reshape, as a formula
Lay the tensor out flat in **row-major** order (finish row 1, then row 2, …). Entry $(i, j)$ of an $m\times n$ tensor sits at flat position
$$k = i\,n + j \quad (\text{0-indexed}).$$
- $i$ = row number, $j$ = column number, $n$ = how many columns there are.
- $k$ = the position in the flat line. Example: the 5 in $\mathbf{X}$ is at $(1,1)$, so $k = 1\cdot3+1 = 4$ (the fifth slot, counting from 0). ✔

Reshaping to $p\times q$ (you need $pq = mn$, the same number of entries) puts flat slot $k$ at
$$\left(\left\lfloor k/q \right\rfloor,\; k \bmod q\right).$$
- $\lfloor k/q\rfloor$ = "$k$ divided by $q$, rounded down" → the new row.
- $k \bmod q$ = "the remainder after dividing $k$ by $q$" → the new column.
- For the 5: $k=4$, $q=2$ → row $\lfloor 4/2\rfloor = 2$, column $4 \bmod 2 = 0$. And indeed in `[[1,2],[3,4],[5,6]]` the 5 is at row 2, column 0. ✔

The flat order never changes; only the grouping does.

### Transpose, as a formula
$$(\mathbf{X}^\top)_{ij} = X_{ji}$$
- $\mathbf{X}^\top$ = "X transposed" (the little $\top$ means transpose).
- The entry in row $i$, column $j$ of the new matrix is the entry in row $j$, column $i$ of the old one. Rows become columns. Shape $m\times n \to n\times m$. → [[matrix-transpose]]

For higher-rank tensors, `.transpose(d1, d2)` swaps just those two axes and leaves the rest alone.

### `.view()` vs `.reshape()` — the memory catch (book p. 261)
- `.view()` is "the more common command for reshaping tensors in PyTorch" (book p. 260), but it **needs the data to be contiguous in memory** and fails otherwise.
- `.reshape()` always works; if needed it **copies** the data to get the requested shape.
- Why the catch? A transpose doesn't move any bytes — it just changes the "stride" bookkeeping, so the data is no longer laid out in row-major order. That's why multi-head attention calls `.contiguous()` before `.view()` (see below). (This explanation is ours; the book shows the call but doesn't explain it here.)
- PyTorch has two names for many things because it first copied Lua Torch syntax and later added NumPy-style syntax by popular request (book pp. 260–261).

### The big use: splitting and merging attention heads (§3.6.2, book pp. 86–89)
In `MultiHeadAttention`, with $b$ = batch size, $T$ = number of tokens, $h$ = `num_heads`, $d_h$ = `head_dim` $= d_{out}/h$:

| Step | Code | Shape |
|---|---|---|
| project | `keys = self.W_key(x)` | $(b, T, d_{out})$ |
| **split heads** (reshape) | `keys.view(b, num_tokens, self.num_heads, self.head_dim)` | $(b, T, h, d_h)$ |
| **heads first** (transpose) | `keys.transpose(1, 2)` | $(b, h, T, d_h)$ |
| scores | `queries @ keys.transpose(2, 3)` | $(b, h, T, T)$ |
| context | `attn_weights @ values` | $(b, h, T, d_h)$ |
| **tokens first again** | `.transpose(1, 2)` | $(b, T, h, d_h)$ |
| **merge heads** (reshape) | `.contiguous().view(b, num_tokens, self.d_out)` | $(b, T, d_{out})$ |

The `.view` "implicitly split[s] the matrix by adding a num_heads dimension" and "unroll[s] the last dim" (book p. 87). The book calls the transpose "crucial for correctly aligning the queries, keys, and values across the different heads and performing batched matrix multiplications efficiently" (book p. 88). The final `.view` "combines heads, where self.d_out = self.num_heads * self.head_dim" (book p. 87).

**Concrete numbers.** With the book's GPT-2 small sizes (12 heads, $d_{out}=768$, book p. 90): each token's 768-vector is cut into 12 consecutive chunks of $768/12 = 64$ numbers. Chunk 1 is head 1's query, chunk 2 is head 2's, and so on.

### Other places it shows up
- **Flattening for the loss** (§5.1, book pp. 138–139): `logits.flatten(0, 1)` turns $[2, 3, 50257]$ into $[6, 50257]$ and `targets.flatten()` turns $[2, 3]$ into $[6]$, because `cross_entropy` wants one row per prediction. → [[cross-entropy-loss]]
- **Loading GPT-2 weights** (§5.5, book pp. 165–166): most OpenAI weight matrices (e.g. `c_attn`, `c_proj`, `c_fc`) are copied in with `.T`. The book shows the `.T` calls but does not explain them; the reason (ours) is that OpenAI stores them as $(d_{in}, d_{out})$ while PyTorch's `nn.Linear` stores its weight as $(d_{out}, d_{in})$. → [[linear-layer]], [[pretrained-weights]]

## 🔴 Theory (Expert)
- **Strided storage.** A PyTorch tensor is a 1D storage buffer plus a shape $(s_0,\dots,s_{r-1})$, strides $(\sigma_0,\dots,\sigma_{r-1})$ and an offset. Element $(i_0,\dots,i_{r-1})$ lives at `offset + Σ i_ℓ σ_ℓ`. Row-major contiguous means $\sigma_{r-1}=1$ and $\sigma_\ell = \sigma_{\ell+1}s_{\ell+1}$.
- **Transpose = stride permutation.** `transpose(a, b)` swaps $(s_a,\sigma_a)\leftrightarrow(s_b,\sigma_b)$. It is $O(1)$ and copies nothing. The result is generally non-contiguous.
- **View = re-factoring the index space.** A view to a new shape is possible without copying iff the requested shape can be expressed by splitting/merging dimensions whose strides are compatible. Merging axes $\ell,\ell+1$ needs $\sigma_\ell = \sigma_{\ell+1}s_{\ell+1}$. After `transpose(1,2)`, axes $T$ and $h$ violate this, so merging them back into $d_{out}$ needs `.contiguous()` (a real copy) — or `.reshape`, which does that copy for you.
- **Algebraic view.** Reshape is a bijection between index sets $[m]\times[n]\to[p]\times[q]$ that preserves the row-major flat order: on $\mathbb{R}^{m\times n}\cong\mathbb{R}^{mn}\cong\mathbb{R}^{p\times q}$ it is the identity on the underlying [[vector-space]]. Transpose is a genuine permutation of coordinates — a [[linear-mapping]] whose matrix is a commutation (perfect-shuffle) permutation matrix $K_{m,n}$ with $\operatorname{vec}(\mathbf{X}^\top)=K_{m,n}\operatorname{vec}(\mathbf{X})$. Both are orthogonal maps (they preserve the Frobenius [[norm]]).
- **Heads as a direct sum.** Viewing $(b,T,d_{out})$ as $(b,T,h,d_h)$ is the identification $\mathbb{R}^{d_{out}} \cong \mathbb{R}^{d_h}\oplus\cdots\oplus\mathbb{R}^{d_h}$ ($h$ copies). It is consistent with the column-block split of $W_q=[W_q^{(1)}\|\cdots\|W_q^{(h)}]$, because $XW_q = [XW_q^{(1)}\|\cdots\|XW_q^{(h)}]$. → [[weight-splitting-multi-head-attention]]
- **Tensor "rank" warning.** The number of axes (the book calls it order or rank, p. 258) is unrelated to matrix rank in [[basis-and-rank]]; reshaping can change the matrix rank of a 2D unfolding even though it keeps every number.

## Connections
- Builds on: [[tensor]], [[matrix]], [[matrix-transpose]], [[vector-space]]
- Leads to: [[batched-matrix-multiplication]], [[multi-head-attention]], [[weight-splitting-multi-head-attention]]
- Used by: [[self-attention]], [[cross-entropy-loss]], [[pretrained-weights]], [[pytorch]]
- Related: [[linear-mapping]], [[basis-and-rank]], [[norm]], [[linear-layer]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2 and §A.2.3, book pp. 258, 260–261 (tensor rank; `reshape`, `view`, `.T`, contiguity remark; example `[[1,2,3],[4,5,6]]`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.2, book pp. 86–90 (Listing 3.5 `.view`/`.transpose`/`.contiguous`; GPT-2 12 heads × 768).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 138–139 (flattening logits and targets); §5.5, book pp. 165–166 (`.T` when loading weights).
- Strides, commutation matrix and direct-sum remarks are standard background, not from the book.
