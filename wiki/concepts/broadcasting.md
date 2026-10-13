---
title: Tensor Broadcasting and Reductions (dim, keepdim)
type: concept
tags: [pytorch, tensors, linear-algebra, numerics]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Tensor Broadcasting and Reductions (dim, keepdim)

> **One-liner:** A reduction squashes a tensor along one direction (like taking each row's average), and broadcasting lets a small tensor be reused automatically across a big one (like subtracting each row's average from that row).

## 🟢 Intuition (Dummy)
Picture a spreadsheet of test scores: one row per student, one column per test.

- **Reduction.** "Give me each student's average" squashes every row down to one
  number. "Give me each test's average" squashes every column instead. *Which*
  direction you squash is what PyTorch's `dim` argument picks.
- **`keepdim`.** After squashing rows, do you want the answer as a plain list of
  averages, or as a skinny one-column table that still lines up next to the original
  sheet? `keepdim=True` gives you the skinny table.
- **Broadcasting.** Now "subtract each student's average from their own scores".
  You don't copy the average into every cell by hand — PyTorch stretches that skinny
  column across the row for you. Same thing when the LLM adds one set of seat
  vectors to *every* sentence in a batch.

These two ideas show up all over the book: building the
[[input-embeddings]] (book p. 47), [[layer-normalization]] (book pp. 101–103), and
renormalizing [[causal-attention-mask|masked]] attention weights (book p. 76).

## 🟡 Mechanics (Practitioner)

### Reductions: `dim`
For a 2-D [[tensor]] (a [[matrix]]) with shape `[rows, columns]` (book p. 101, Fig. 4.6):
- `dim=0` works *down the rows* (vertically) → one result **per column**.
- `dim=1` or `dim=-1` works *across the columns* (horizontally) → one result **per row**.
- `dim=-1` always means "the last dimension". For 2-D it's the same as `dim=1`; for
  GPT's 3-D tensors `[batch_size, num_tokens, embedding_size]` it means the embedding
  axis, so the code doesn't need to change from `dim=1` to `dim=2` (book p. 102).

In math, the row mean is

$$\mu_r = \frac{1}{n}\sum_{j=1}^{n} X_{r,j}$$

- $\mu_r$ = mean of row $r$ (Greek "mu").
- $n$ = number of columns (here 6).
- $\sum_{j=1}^{n}$ = "add up, for $j$ going from 1 to $n$".
- $X_{r,j}$ = the entry in row $r$, column $j$.

### The book's numbers (book pp. 100–101)
```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
# [[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#  [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]]
mean = out.mean(dim=-1, keepdim=True)   # [[0.1324], [0.2170]]   shape [2, 1]
var  = out.var(dim=-1, keepdim=True)    # [[0.0231], [0.0398]]   shape [2, 1]
```
Check row 1 by hand: $(0.2260+0.3470+0+0.2216+0+0)/6 = 0.7946/6 \approx 0.1324$. ✓

Figure 4.6 also shows the `dim=0` (per-column) means:
`[0.21, 0.29, 0.00, 0.37, 0.16, 0.00]` — e.g. column 2 is $(0.3470+0.2394)/2 \approx 0.29$
(the figure truncates to 2 decimals, book p. 101).

### `keepdim`
`keepdim=True` keeps the same **number of dimensions** as the input, even though one
of them is squashed to size 1 (book p. 101):

| Call | Result | Shape |
|---|---|---|
| `out.mean(dim=-1, keepdim=True)` | `[[0.1324], [0.2170]]` | `[2, 1]` |
| `out.mean(dim=-1)` | `[0.1324, 0.2170]` | `[2]` |

(The book calls the second one a "two-dimensional vector" — it means a vector with 2
entries, i.e. shape `[2]`.) Why care? Because the `[2, 1]` version lines up correctly in
the next step, `out - mean`. That's broadcasting.

### Broadcasting: the rule
Line the two shapes up **from the right**. Each pair of sizes must either be equal, or
one of them must be 1 (or missing). A size-1 or missing axis gets stretched to match.

$$C_{i,j} = A_{i,j} - \mu_{i,1} \quad\text{for every } j$$

- $A$ = the big tensor (shape $2\times6$); $\mu$ = the skinny one ($2\times1$).
- $\mu_{i,1}$ = the only entry in row $i$ of $\mu$ — it's reused for *every* column $j$.

### Every broadcast in the book worth knowing
| Where | Operation | Shapes | Book |
|---|---|---|---|
| Input embeddings | `token_embeddings + pos_embeddings` | `[8,4,256] + [4,256]` → `[8,4,256]` | p. 47 |
| GPT model | `tok_embeds + pos_embeds` | `[B,T,768] + [T,768]` | pp. 97, 119 |
| Masked attention | `masked_simple / row_sums` | `[6,6] / [6,1]` | p. 76 |
| Layer-norm centering | `out - mean`, `/ torch.sqrt(var)` | `[2,6]` with `[2,1]` | p. 102 |
| Layer-norm affine | `self.scale * norm_x + self.shift` | `[emb_dim]` with `[B,T,emb_dim]` | p. 103 |

Line-up for the first row:
```
token_embeddings:  8 × 4 × 256
pos_embeddings:        4 × 256     (missing axis → treated as 1 → stretched to 8)
result:            8 × 4 × 256
```
The book never says the word "broadcasting"; it just describes the effect: PyTorch
adds "the 4 × 256–dimensional pos_embeddings tensor to each 4 × 256–dimensional token
embedding tensor in each of the eight batches" (book p. 47).

What goes wrong without `keepdim`: `out` is `[2, 6]` and a plain mean is `[2]`.
Aligning from the right compares 6 with 2 → error. (If the shapes had happened to
match, e.g. a square `[6,6]` matrix with a `[6]` vector, PyTorch would silently
subtract *column*-wise — a nasty bug. `keepdim=True` prevents both.)

## 🔴 Theory (Expert)
- **Broadcasting = multiplying by a ones vector.** For $A\in\mathbb{R}^{m\times n}$ and a
  column $\boldsymbol\mu\in\mathbb{R}^{m}$, "$A - \boldsymbol\mu$" (with keepdim) means
  $A - \boldsymbol\mu\mathbf{1}_n^\top$, an outer product with the all-ones vector. Adding
  $P\in\mathbb{R}^{T\times d}$ to a batch is $X_b = T_b + P$ for every $b$, i.e.
  $X = T + \mathbf{1}_B \otimes P$. Broadcasting is a memory-free way to write these
  rank-1 "copies" (PyTorch uses stride 0, no data is duplicated).
- **Reduction = multiplying by a ones vector, too.** Row means are
  $\boldsymbol\mu = \tfrac{1}{n}A\mathbf{1}_n$; column means are $\tfrac{1}{m}\mathbf{1}_m^\top A$.
  Centering combines both: $A - \tfrac1n A\mathbf{1}\mathbf{1}^\top = A\,(I - \tfrac1n\mathbf{1}\mathbf{1}^\top)$.
  The matrix $H = I - \tfrac1n\mathbf{1}\mathbf{1}^\top$ is the *centering matrix*, an
  [[orthogonal-projection]] onto the [[orthogonal-complement]] of $\mathbf{1}$ — which is
  why layer-norm outputs have mean 0 (book p. 102).
- **Broadcast and sum are adjoints (the gradient rule).** If $\mathbf{y}=\mathrm{broadcast}(\mathbf{p})$
  copies $\mathbf{p}$ along axis $b$, then by the [[chain-rule]]
  $\partial L/\partial\mathbf{p}=\sum_b \partial L/\partial\mathbf{y}_b$. Forward broadcast ↔
  backward sum-reduction over the stretched axes. That's how the single positional table
  receives the summed gradient of all 8 sequences, and how `scale`/`shift` in
  `LayerNorm` get gradients summed over batch and tokens. Conversely the gradient of a
  mean is a broadcast of $1/n$.
- **Variance and Bessel.** `out.var` defaults to `unbiased=True` (divide by $n-1$,
  [[bessels-correction-biased-variance|Bessel's correction]]); the book's `LayerNorm`
  uses `unbiased=False` (divide by $n$) (book p. 103). With $n=768$ the difference is
  negligible.
- **Floating-point residue.** After centering, the book's means print as
  $-5.9605\times10^{-8}$ and $1.9868\times10^{-8}$, not exactly 0, because of finite
  [[floating-point-precision]] (book p. 102).

## Connections
- Builds on: [[tensor]], [[matrix]], [[vector]], [[expected-value-and-mean]]
- Leads to: [[layer-normalization]], [[input-embeddings]], [[positional-embedding]], [[standardization-z-score]], [[batched-matrix-multiplication]]
- Used by: [[causal-attention-mask]], [[softmax]], [[argmax]] (`argmax(..., dim=-1, keepdim=True)` gives shape `(batch, 1)`, book p. 124)
- Related: [[tensor-reshaping-and-transposing]], [[covariance-and-variance]], [[bessels-correction-biased-variance]], [[chain-rule]], [[backpropagation]]
- People: [[friedrich-wilhelm-bessel]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book p. 47 (adding `[4,256]` positional embeddings to `[8,4,256]` token embeddings).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5, book p. 76 (`sum(dim=-1, keepdim=True)` renormalization).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 100–103 (Linear+ReLU output; `mean`/`var` with `dim=-1, keepdim=True`; Fig. 4.6 `dim=0` vs `dim=1`; `dim=-1` for 3-D tensors; scientific-notation residue; `LayerNorm` class).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 124–125 (`argmax(..., keepdim=True)`).
- Broadcasting rule, ones-vector/centering-matrix view, and broadcast–sum adjoint: standard PyTorch/linear-algebra facts, not stated in the book.
