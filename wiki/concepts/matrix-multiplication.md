---
title: Matrix Multiplication
type: concept
tags: [linear-algebra, matrix, pytorch, deep-learning]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Matrix Multiplication

> **One-liner:** To multiply two matrices, you take every row of the first one and "dot" it with every column of the second one, and write each answer into a new grid.

## 🟢 Intuition (Dummy)
Imagine a shop. One table lists what each customer bought (rows = customers, columns =
products). A second table lists each product's price and weight (rows = products, columns
= price, weight). Multiplying the two tables gives you a third one: for each customer,
their total bill and the total weight of their bag. To fill one cell you walk along the
customer's row and down the "price" column, multiply the pairs, and add them up.

That "walk along a row, walk down a column, multiply pairs, add" is the whole idea. It's
also the single most important operation in deep learning: almost every layer of a
neural network — and nearly all the work inside an LLM — is a matrix multiplication.
That's why we buy GPUs (see [[gpu-computing]]).

One rule to remember: the inner sizes have to match. A table with 3 columns can only be
multiplied by a table with 3 rows — the number of products must agree.

## 🟡 Mechanics (Practitioner)

### The formula
For $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $\mathbf{B}\in\mathbb{R}^{n\times p}$:

$$(\mathbf{A}\mathbf{B})_{ij} = \sum_{k=1}^{n} A_{ik}\,B_{kj}, \qquad \mathbf{A}\mathbf{B}\in\mathbb{R}^{m\times p}$$

Reading it symbol by symbol:
- $\mathbf{A}\in\mathbb{R}^{m\times n}$ — "$\mathbf{A}$ is a [[matrix]] of real numbers with $m$ rows and $n$ columns".
- $(\mathbf{A}\mathbf{B})_{ij}$ — "the entry in row $i$, column $j$ of the product".
- $\sum_{k=1}^{n}$ — "add up, for $k$ going from 1 to $n$, …".
- $A_{ik}$ — the $k$-th entry of row $i$ of $\mathbf{A}$; $B_{kj}$ — the $k$-th entry of column $j$ of $\mathbf{B}$.
- So each entry is a [[inner-product|dot product]] of **row $i$ of $\mathbf{A}$** with **column $j$ of $\mathbf{B}$**.
- Shapes: $[m, n] \times [n, p] \to [m, p]$. The shared $n$ is the **inner dimension** and disappears.

### The book's worked example (§A.2.3, book pp. 260–261)
$$\mathbf{X} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}\ (2\times 3),\qquad
\mathbf{X}^\top = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}\ (3\times 2)$$

$\mathbf{X}^\top$ ("X transpose") flips rows into columns: $(\mathbf{X}^\top)_{ij} = X_{ji}$
(see [[matrix-transpose]]). Then

$$\mathbf{X}\mathbf{X}^\top = \begin{bmatrix} 14 & 32 \\ 32 & 77 \end{bmatrix}$$

Check by hand — every entry is "row dotted with row":
- top-left: $1\cdot1 + 2\cdot2 + 3\cdot3 = 14$
- off-diagonal: $1\cdot4 + 2\cdot5 + 3\cdot6 = 32$
- bottom-right: $4\cdot4 + 5\cdot5 + 6\cdot6 = 77$

Shapes: $[2,3]\times[3,2]\to[2,2]$. The answer is symmetric — this is a [[gram-matrix]].
In PyTorch there are two spellings for the same thing:

```python
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2d.matmul(tensor2d.T)   # tensor([[14, 32], [32, 77]])
tensor2d @ tensor2d.T         # same, more compact
```

Careful: `.T` (transpose) is **not** `.reshape(3, 2)`. Both give a $3\times2$ shape, but
`reshape` gives `[[1,2],[3,4],[5,6]]` — the entries land in different places (book p. 261).

### Matrix multiplication inside a neural network
- **A Linear layer is a matmul plus a bias.** The book: a Linear layer "multiplies the
  inputs by a weight matrix and adds a bias vector" (book p. 267). In PyTorch's convention,
  with a batch of inputs as rows,
  $$\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{b}$$
  $\mathbf{X}$ = inputs $[n, d_\text{in}]$ ($n$ examples), $\mathbf{W}$ = weight matrix stored as
  $[d_\text{out}, d_\text{in}]$, $\mathbf{W}^\top$ its transpose, $\mathbf{b}$ = [[bias-vector]]
  added to every row ([[broadcasting]]). That's why `model.layers[0].weight.shape` prints
  `torch.Size([30, 50])` for `Linear(50, 30)` (book p. 267), and why the output tensor shows
  `grad_fn=<AddmmBackward0>` — "Addmm" = matrix multiply (`mm`) then add (book p. 269).
- **Parameter count.** A `Linear(d_in, d_out)` has $d_\text{in}d_\text{out} + d_\text{out}$
  numbers. The book's MLP 50→30→20→3 has $1530 + 620 + 63 = 2213$ trainable parameters
  (book p. 267); the 2→30→20→2 version has **752** (Exercise A.3, book p. 311).
- **Attention.** In [[self-attention]] the score matrix is $\mathbf{Q}\mathbf{K}^\top$ —
  exactly the "rows dotted with rows" pattern of $\mathbf{X}\mathbf{X}^\top$ above. When there
  is a batch and several heads, PyTorch multiplies stacks of matrices at once
  ([[batched-matrix-multiplication]]). Exercise 3.1 shows the transpose bookkeeping: an
  `nn.Linear` weight must be transposed (`.weight.T`) to be used as a raw $\mathbf{x}\mathbf{W}$
  matrix (book p. 301).

### LoRA: multiplying by two thin matrices (§E.4, book pp. 328–332)
[[low-rank-adaptation|LoRA]] replaces a big weight update $\Delta\mathbf{W}$ by a product of
two thin matrices:

$$\Delta\mathbf{W} \approx \mathbf{A}\mathbf{B},\qquad \mathbf{A}\in\mathbb{R}^{d_\text{in}\times r},\ \mathbf{B}\in\mathbb{R}^{r\times d_\text{out}}$$

$r$ = the **rank**, the tiny inner dimension (16 in the book). The forward pass is

$$\text{LoRA}(\mathbf{x}) = \frac{\alpha}{r}\,(\mathbf{x}\mathbf{A}\mathbf{B})$$

$\mathbf{x}$ = the layer input, $\alpha$ = a scaling factor, $\alpha/r$ = the effective scale.
The book uses `rank=16, alpha=16` and notes $\alpha$ is usually half, double, or equal to the
rank (book p. 332). $\mathbf{A}$ is initialized like a PyTorch Linear layer
(`kaiming_uniform_`), $\mathbf{B}$ with zeros (Listing E.5, book p. 329).
Shapes: $[\dots, d_\text{in}] \to$ `x @ A` $[\dots, r] \to$ `@ B` $[\dots, d_\text{out}]$.

```python
def forward(self, x):
    x = (self.alpha / self.rank) * (x @ self.A @ self.B)
    return x
```

Counting: $\mathbf{A}$ and $\mathbf{B}$ together have $r(d_\text{in}+d_\text{out})$ numbers instead
of $d_\text{in}d_\text{out}$. For a $768\to768$ attention projection with $r=16$:
$16\cdot1536 = 24{,}576$ vs $589{,}824$. Over GPT-2 small this gives **2,666,528** trainable
LoRA parameters vs **124,441,346** total — "almost 50×" fewer (book p. 332).

Because $\mathbf{B}$ starts as all zeros, $\mathbf{A}\mathbf{B}$ is the zero matrix and
$\mathbf{W} + \mathbf{0} = \mathbf{W}$: the model behaves exactly as before at the start
(book pp. 330, 334).

### CPU vs GPU timing (Exercise A.4, book pp. 282, 311–312)
`a = torch.rand(100, 200)`, `b = torch.rand(200, 300)`: `a @ b` took 63.8 µs on CPU vs
13.8 µs on a V100 GPU — about 4× faster. That product needs
$100\cdot300$ outputs × 200 multiply-adds $= 6{,}000{,}000$ multiply-adds.

## 🔴 Theory (Expert)
- **Matmul = composition of linear maps.** If $\mathbf{B}$ represents a [[linear-mapping]]
  $g$ and $\mathbf{A}$ represents $f$, then $\mathbf{A}\mathbf{B}$ represents $f\circ g$. This is
  *why* the row-times-column rule is the right definition, and why a stack of Linear layers
  without nonlinear [[activation-function|activations]] collapses into a single matrix.
- **Associative, distributive, not commutative.** $(\mathbf{A}\mathbf{B})\mathbf{C} =
  \mathbf{A}(\mathbf{B}\mathbf{C})$ and $\mathbf{A}(\mathbf{B}+\mathbf{C}) = \mathbf{A}\mathbf{B}+\mathbf{A}\mathbf{C}$,
  but in general $\mathbf{A}\mathbf{B}\ne\mathbf{B}\mathbf{A}$ (often one side isn't even
  defined). LoRA uses both laws: $\mathbf{x}\mathbf{W} + \tfrac{\alpha}{r}\mathbf{x}\mathbf{A}\mathbf{B} =
  \mathbf{x}(\mathbf{W} + \tfrac{\alpha}{r}\mathbf{A}\mathbf{B})$ (the book credits "the distributive law of matrix multiplication", §E.1, book p. 323), and
  computing `x @ A @ B` left-to-right as $(\mathbf{x}\mathbf{A})\mathbf{B}$ never forms the big
  $d_\text{in}\times d_\text{out}$ matrix (associativity). For $n$ input rows, $(\mathbf{x}\mathbf{A})\mathbf{B}$
  costs $nr(d_\text{in}+d_\text{out})$ multiply-adds, while forming $\mathbf{A}\mathbf{B}$ first costs
  $d_\text{in}rd_\text{out}$ on top. Choosing the cheapest bracketing is the
  matrix-chain-ordering problem — and the same associativity is what makes reverse-mode
  [[automatic-differentiation]] cheap (see [[backpropagation]]).
- **Transpose of a product.** $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top\mathbf{A}^\top$. Hence
  $(\mathbf{X}\mathbf{X}^\top)^\top = \mathbf{X}\mathbf{X}^\top$ (symmetric), and
  $\mathbf{v}^\top\mathbf{X}\mathbf{X}^\top\mathbf{v} = \lVert\mathbf{X}^\top\mathbf{v}\rVert^2 \ge 0$
  (positive semidefinite) — see [[gram-matrix]], [[symmetric-positive-definite-matrix]].
- **Rank bound.** $\operatorname{rank}(\mathbf{A}\mathbf{B}) \le \min(\operatorname{rank}\mathbf{A},
  \operatorname{rank}\mathbf{B})$, so the LoRA update satisfies
  $\operatorname{rank}(\mathbf{A}\mathbf{B})\le r$. This is the precise sense of "low-rank" in
  [[low-rank-approximation]]; see [[basis-and-rank]].
- **Gradients of a matmul.** For $\mathbf{Y}=\mathbf{X}\mathbf{W}$ and a scalar loss $L$:
  $\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top\frac{\partial L}{\partial \mathbf{Y}}$ and
  $\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}}\mathbf{W}^\top$ — the
  backward pass of a matmul is two more matmuls. With LoRA's $\mathbf{B}=\mathbf{0}$ at start,
  $\partial L/\partial \mathbf{A} \propto \mathbf{x}^\top(\partial L/\partial\mathbf{y})\mathbf{B}^\top = \mathbf{0}$
  on the first step, while $\partial L/\partial\mathbf{B}\neq\mathbf{0}$, so learning still begins.
- **Complexity.** Naive cost is $mnp$ multiply-adds ($O(n^3)$ for square matrices).
  Strassen-type algorithms are asymptotically faster ($O(n^{2.81})$ and below) but rarely
  used in deep learning; GPUs win by parallelising the $mp$ independent dot products and by
  [[floating-point-precision|low-precision]] tensor cores.
- **Tensor rank vs matrix rank.** The book calls the number of axes of a [[tensor]] its
  "rank" (book p. 258). That is *not* the matrix rank above (the number of linearly
  independent columns).

### From llm-ch03-coding-attention-mechanisms
Ch. 3 is where matrix multiplication does the real work of attention. The book first computes attention scores with two nested Python `for` loops, then says loops are slow and replaces them with one product, `attn_scores = inputs @ inputs.T` (book p. 62). That one line gives the full $6 \times 6$ score matrix $\Omega = XX^\top$.
- Read $\Omega = XX^\top$ as: "Omega (the score table) equals X times X-transpose". Entry $\omega_{ij}$ is the dot product of row $i$ and row $j$ of $X$, so it measures how similar token $i$ is to token $j$ (see [[gram-matrix]]).
- The same idea gives all context vectors at once: $Z = AX$, "Z equals A times X", where each row of the weight matrix $A$ mixes the rows of $X$ (book pp. 63–64; see [[context-vector]]).
- With trainable weights, the product is $QK^\top$ (book p. 70; see [[scaled-dot-product-attention]]). With batches, PyTorch multiplies the last two axes for every batch item: `queries @ keys.transpose(1, 2)` (Listing 3.3, book p. 81; see [[batched-matrix-multiplication]]).

> **From [[llm-ch04-implementing-a-gpt-model]]:** the output head is one matrix product. The final hidden states $H \in \mathbb{R}^{T\times 768}$ times $W^\top$, where $W \in \mathbb{R}^{50257\times 768}$, gives logits $HW^\top \in \mathbb{R}^{T\times 50257}$ (book p. 119). Entry $(t, v)$ is the [[inner-product]] of token position $t$'s hidden vector with row $v$ of $W$. Because the token-embedding matrix has the same shape `[50257, 768]` (p. 121), the same $W$ can serve both roles ([[weight-tying]]).

## Connections
- Builds on: [[matrix]], [[vector]], [[inner-product]], [[matrix-transpose]], [[tensor]]
- Leads to: [[batched-matrix-multiplication]], [[linear-layer]], [[gram-matrix]], [[low-rank-approximation]]
- Used by: [[self-attention]], [[scaled-dot-product-attention]], [[multilayer-perceptron]], [[feed-forward-network]], [[low-rank-adaptation]], [[backpropagation]], [[gpu-computing]]
- Related: [[linear-mapping]], [[basis-and-rank]], [[parameter-count-and-memory]], [[broadcasting]], [[tensor-reshaping-and-transposing]]
- People: [[edward-j-hu]] (LoRA)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2 / §A.2.3 (tensor rank; `.reshape`, `.T`, `.matmul`, `@`, $\mathbf{X}\mathbf{X}^\top$), book pp. 258, 260–261
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (Linear layer = weight matrix + bias, `[30, 50]` weight, 2213 params, `AddmmBackward0`), book pp. 266–269
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9.2 Exercise A.4, book p. 282; solution in Appendix C, book pp. 311–312
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 3.1 (weight transpose), book p. 301; Exercise A.3 (752 params), book p. 311
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1 (distributive law, $\mathbf{x}(\mathbf{W}+\Delta\mathbf{W})$ split), book p. 323
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4 (LoRA $\mathbf{x}\mathbf{A}\mathbf{B}$, Listings E.5–E.6, 2,666,528 params), book pp. 328–336
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.2 (`inputs @ inputs.T` replaces for loops; $Z = AX$), book pp. 62–64; §3.4.2 ($QK^\top$), book p. 70; §3.5.3 Listing 3.3 (`queries @ keys.transpose(1, 2)`), book p. 81.
- Same PDF, §4.6, book pp. 119, 121 (`out_head = nn.Linear(768, 50257, bias=False)`; token-embedding and output-head weights both `[50257, 768]`, which enables [[weight-tying]]).
