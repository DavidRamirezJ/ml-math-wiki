---
title: Linear Mapping
type: concept
tags: [linear-algebra, foundations]
difficulty: advanced
sources: [mml-ch02-linear-algebra, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch06-fine-tuning-for-classification, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-appA-introduction-to-pytorch]
created: 2026-06-04
updated: 2026-10-13
---

# Linear Mapping

> **One-liner:** A linear mapping is a function between vector spaces that respects adding and scaling — and every such function is just a matrix in disguise.

## 🟢 Intuition (Dummy)
Some functions are "well-behaved": if you stretch the input twice as much, the output stretches
twice as much; if you add two inputs, the outputs add. A linear mapping is exactly that kind of
function on vectors — it doesn't bend or warp space unevenly, it only rotates, scales, shears, or
projects. The magic fact: *any* such map can be written as multiplication by a [[matrix]]. So
"function on vectors" and "matrix" are two views of the same thing.

## 🟡 Mechanics (Practitioner)
**Definition (Def 2.15):** $\Phi: V\to W$ is a **linear mapping** (a.k.a. *vector space
homomorphism* / *linear transformation*) if for all $\mathbf x,\mathbf y\in V$, $\lambda,\psi\in\mathbb R$:
$$ \Phi(\lambda\mathbf x + \psi\mathbf y) = \lambda\,\Phi(\mathbf x) + \psi\,\Phi(\mathbf y) $$
(Equivalently the two conditions $\Phi(\mathbf x+\mathbf y)=\Phi(\mathbf x)+\Phi(\mathbf y)$ and $\Phi(\lambda\mathbf x)=\lambda\Phi(\mathbf x)$.)

**Special maps (Def 2.16 + 2.7):**
| Term | Meaning |
|---|---|
| Injective | $\Phi(\mathbf x)=\Phi(\mathbf y)\Rightarrow \mathbf x=\mathbf y$ |
| Surjective | $\Phi(V)=W$ (everything in $W$ is reached) |
| Bijective | injective **and** surjective (can be undone by $\Phi^{-1}$) |
| Isomorphism | linear **and** bijective |
| Endomorphism | linear, $V\to V$ |
| Automorphism | linear, bijective, $V\to V$ |

**Coordinates (Def 2.18):** pick an *ordered basis* $B=(\mathbf b_1,\dots,\mathbf b_n)$. Any
$\mathbf x = \alpha_1\mathbf b_1+\dots+\alpha_n\mathbf b_n$ has a **coordinate vector**
$\boldsymbol\alpha=[\alpha_1,\dots,\alpha_n]^\top$. A basis *is* a coordinate system; the same vector
$\mathbf x$ has different coordinates in different bases (Fig. 2.9: $[2,3]^\top$ in $(e_1,e_2)$ becomes
$\tfrac12[-1,5]^\top$ in $(b_1,b_2)$).

**Transformation matrix (Def 2.19):** with ordered bases $B$ of $V$, $C$ of $W$, write each
$\Phi(\mathbf b_j) = \sum_i \alpha_{ij}\mathbf c_i$; then $A_\Phi=(\alpha_{ij})$ is the matrix of $\Phi$.
**The coordinates of $\Phi(\mathbf b_j)$ in $C$ are the $j$-th column of $A_\Phi$.**

### In LLMs (Raschka, *Build a LLM from Scratch*)
Inside a GPT model, the workhorse `torch.nn.Linear` layer **is** a linear mapping (when
`bias=False`) or an [[affine-space|affine mapping]] (when `bias=True`). Most of a GPT's
parameters live in these maps.

> ⚠️ Notation difference (not a contradiction): MML writes $\Phi(\mathbf{x}) = A\mathbf{x}$ with
> column vectors. The LLM book and PyTorch write row vectors, $\mathbf{y} = \mathbf{x}W$, and
> `nn.Linear` stores the weight as $(d_{out}, d_{in})$ and computes $\mathbf{y} = \mathbf{x}W^\top + \mathbf{b}$
> (book p. 73). Same map, transposed bookkeeping.

**The `nn.Linear` formula.**
$$\mathbf{y} = \mathbf{x}W^{\top} + \mathbf{b}$$
Symbol by symbol:
- $\mathbf{x}\in\mathbb{R}^{d_{in}}$ = the input row vector (e.g. one token's 768 numbers).
- $W\in\mathbb{R}^{d_{out}\times d_{in}}$ = the weight matrix as PyTorch stores it; $W^\top$ = its transpose.
- $\mathbf{b}\in\mathbb{R}^{d_{out}}$ = the [[bias-vector]]; with `bias=False` it is absent and the map is truly linear.
- $\mathbf{y}\in\mathbb{R}^{d_{out}}$ = the output.
Parameter count: $d_{in}d_{out}$ (+ $d_{out}$ with bias). Applied to a $(B, T, d_{in})$ tensor it acts on the last axis, row by row.

**Where the linear maps are in GPT-2 124M** ($d = 768$):
| Map | $\mathbb{R}^{d_{in}}\to\mathbb{R}^{d_{out}}$ | Bias | Book |
|---|---|---|---|
| Query / key / value projections $W_q, W_k, W_v$ | $768\to768$ (toy: $3\to2$) | `qkv_bias=False` (True for OpenAI weights, §5.5 p. 164) | §3.4, §4.1 p. 95 |
| Attention output projection `out_proj` | $768\to768$ | yes | §3.6 pp. 86–89 |
| Feed-forward expand `layers[0]` | $768\to3072$ | yes | §4.3 pp. 107–108 |
| Feed-forward contract `layers[2]` | $3072\to768$ | yes | §4.3 pp. 107–108 |
| Language-model head `out_head` | $768\to50{,}257$ | no | §4.1 p. 97, §4.6 |
| Classification head (ch. 6) | $768\to2$ | yes | §6.5 p. 186, §E.3 p. 327 |

**Toy projection (§3.4, book p. 66).** With seed 123 and $W_q\in\mathbb{R}^{3\times2}$, the map
$\mathbf{x}\mapsto\mathbf{x}W_q$ sends "journey" $\mathbf{x}^{(2)} = [0.55, 0.87, 0.66]$ to the query
$\mathbf{q}^{(2)} = [0.4306,\ 1.4551]$: a linear map $\mathbb{R}^3\to\mathbb{R}^2$. For all six tokens at
once, $Q = XW_q$ maps the $6\times3$ input to $6\times2$.

**Linear + nonlinearity (§4.2, book p. 100).**
```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)      # shape [2, 6]
```
The `Linear(5, 6)` part is an affine map $\mathbb{R}^5\to\mathbb{R}^6$; [[relu]] then zeroes negatives, so the
output rows contain zeros, e.g. $[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000]$.

**Expand–contract feed-forward (§4.3, book p. 108).** `Linear(768, 3072)` → [[gelu]] →
`Linear(3072, 768)`: "expands the embedding dimension into a higher-dimensional space", applies a
nonlinearity, then contracts back, so blocks can be stacked without changing dimensions. See
[[feed-forward-network]].

**Embedding as a linear map (§2.7).** An [[embedding-layer]] equals [[one-hot-encoding]] followed by a
bias-free fully connected layer (book p. 43): the linear map $\mathbb{R}^V\to\mathbb{R}^d$ evaluated on the
standard basis vectors. By Def 2.19, "the image of basis vector $i$" is row $i$ of the embedding table.

**Changing the codomain: head swap (§6.5).** To turn GPT into a spam classifier, the book replaces
`out_head` ($768\to50{,}257$) by `torch.nn.Linear(in_features=768, out_features=2)` (book p. 186).
Same domain (the hidden space), much smaller codomain (two class [[logits]]). See [[classification-head]].

**LoRA adds a second linear map (§E.1, §E.4).**
$$\mathbf{y} = \underbrace{\mathbf{x}W + \mathbf{b}}_{\text{frozen Linear}} + \underbrace{\tfrac{\alpha}{r}\,\mathbf{x}AB}_{\text{LoRA}} = \mathbf{x}\Big(W + \tfrac{\alpha}{r}AB\Big) + \mathbf{b}$$
- $W, \mathbf{b}$ = the pretrained, frozen weight and bias (row-vector convention, as in appendix E).
- $A\in\mathbb{R}^{d_{in}\times r}$, $B\in\mathbb{R}^{r\times d_{out}}$ = the trainable LoRA matrices; $r$ = rank; $\alpha$ = scaling factor.
- $\tfrac{\alpha}{r}AB$ plays the role of the update $\Delta W$.
$A$ starts with Kaiming-uniform values and $B$ with **zeros**, so at the start $AB = 0$ and the model is
exactly the pretrained one (book pp. 329–330). With `rank=16, alpha=16` on every `Linear` in GPT-2
124M (+ the 2-class head), the trainable count is **2,666,528** vs **124,441,346** total (book p. 332).
Check per layer: $r(d_{in}+d_{out})$, e.g. $16\cdot(768+768) = 24{,}576$ per attention projection.
See [[low-rank-adaptation]], [[lora-rank-and-alpha]].

## 🔴 Theory (Expert)
- **Theorem 2.17:** finite-dimensional $V,W$ are **isomorphic ⇔ $\dim V = \dim W$.** Same dimension
  ⇒ "the same up to relabeling." This justifies treating any $n$-dim space as $\mathbb R^n$, and
  $\mathbb R^{m\times n}\cong\mathbb R^{mn}$ (the matrix-stacking of [[matrix]]).
- **Closure of the class:** compositions of linear maps are linear; the inverse of an isomorphism is
  an isomorphism; sums $\Phi+\Psi$ and scalings $\lambda\Phi$ are linear (linear maps form a vector
  space themselves).
- **Kernel & image** of $\Phi$ (matrix $A$): $\ker(A)$ has dimension $n-\text{rk}(A)$, $\text{image}$
  has dimension $\text{rk}(A)$ — the [[basis-and-rank|rank–nullity]] theorem governs injectivity
  (trivial kernel) and surjectivity (full-rank image).
- In ML, every fully-connected layer *without* its bias is a linear mapping; with bias it becomes an
  [[affine-space|affine mapping]].
- **Why nonlinearities are mandatory.** Compositions of linear maps are linear, so
  `Linear(3072, 768) ∘ Linear(768, 3072)` *without* GELU would collapse to a single map
  $\mathbb{R}^{768}\to\mathbb{R}^{768}$ whose matrix $W_2W_1$ has rank $\le 768$: the 3,072-wide middle
  would add parameters but no expressive power. The [[gelu]]/[[relu]] between them is what makes
  the expansion useful (wiki-side argument, not stated in the book).
- **Rank–nullity in the heads.** `out_head`: $\mathbb{R}^{768}\to\mathbb{R}^{50257}$ has rank $\le 768$, so it
  can never be surjective: the logits always live in a $\le 768$-dimensional subspace of
  $\mathbb{R}^{50257}$ (the "softmax bottleneck" idea, external). The $768\to2$ classification head
  has kernel dimension $\ge 766$: most directions in hidden space are invisible to the classifier.
- **LoRA's image.** $\operatorname{rk}(AB)\le r$, so the LoRA update changes the map only along an
  (at most) $r$-dimensional subspace, "the most influential directions" of the weight change
  (book p. 322; see [[low-rank-approximation]], [[basis-and-rank]]). Distributivity,
  $\mathbf{x}(W+AB) = \mathbf{x}W + \mathbf{x}AB$, is exactly the statement that the sum of two linear maps
  is the linear map of the summed matrices (book pp. 323–324). Associativity lets the code compute
  $(\mathbf{x}A)B$ through an $r$-wide bottleneck, never forming the $d_{in}\times d_{out}$ product.
- **Attention is linear in $V$, not in $X$.** For fixed attention weights $A$, $Z = AV = AXW_v$ is
  linear in $X$; but $A = \operatorname{softmax}(XW_qW_k^\top X^\top/\sqrt{d_k})$ itself depends on $X$, so
  self-attention as a whole is a nonlinear map built from linear pieces.
- **Multi-head = block-column map.** One $W_q\in\mathbb{R}^{d_{in}\times d_{out}}$ split into $h$ column
  blocks equals $h$ separate linear maps $\mathbb{R}^{d_{in}}\to\mathbb{R}^{d_{out}/h}$ run in parallel
  (§3.6, Fig. 3.26, book p. 88); see [[weight-splitting-multi-head-attention]].

## Connections
- Builds on: [[vector-space]], [[matrix]], [[basis-and-rank]]
- Leads to: [[affine-space]], change of basis (Ch. 2.7.2), matrix decompositions (Ch. 4)
- Used by: neural network layers, PCA projections, rotations (Ch. 3.9), [[linear-layer]], [[query-key-value]], [[output-projection]], [[feed-forward-network]], [[classification-head]], [[embedding-layer]], [[low-rank-adaptation]]
- Related: [[bias-vector]], [[activation-function]], [[matrix-multiplication]], [[one-hot-encoding]]
- People: —

> **From [[llm-appA-introduction-to-pytorch]]:** §A.5 builds a [[multilayer-perceptron]] from three `torch.nn.Linear` layers (50 → 30 → 20 → 3, Listing A.4). The first weight matrix prints as `torch.Size([30, 50])`, that is out × in, so the layer computes $\mathbf y = \mathbf x W^\top + \mathbf b$ (book pp. 266–267). Because of the bias $\mathbf b$, one `nn.Linear` is an affine map, not a strict linear mapping ([[affine-space]], [[bias-vector]]). The network has $(50\cdot30+30)+(30\cdot20+20)+(20\cdot3+3) = 2{,}213$ trainable parameters (book p. 267). The output's `grad_fn=<AddmmBackward0>` records "matrix multiply, then add" (book pp. 268–269). See [[linear-layer]].

## Sources
- `raw/mml-book.pdf` §2.7, Defs 2.15–2.19, Thm 2.17, printed pp. 48–60 (PDF pp. 53–65). Validated 2026-06-04.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book p. 43 (embedding = one-hot + fully connected layer).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4, book pp. 65–73 ($XW_q$ projections, `nn.Linear` stores $W^\top$); §3.6, book pp. 86–90 (weight splits, `out_proj`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 95–97 (`qkv_bias`, `out_head`); §4.2, book p. 100 (`Linear(5, 6)` + ReLU); §4.3, book pp. 105–108 (768→3072→768); §4.6, book pp. 117–121 (`out_head`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 183–186 (classification head 768→2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1, book pp. 322–324 ($W+\Delta W$, $W+AB$, distributive law); §E.3, book p. 327 (head swap); §E.4, book pp. 328–334 (`LoRALayer`, `LinearWithLoRA`, 2,666,528 params).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (Listing A.4, `[30, 50]` weight shape, 2,213 parameters, `AddmmBackward0`), book pp. 265–269 (PDF pp. 287–291).
