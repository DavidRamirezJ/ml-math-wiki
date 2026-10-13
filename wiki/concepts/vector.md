---
title: Vector
type: concept
tags: [linear-algebra, foundations, llm, embeddings]
difficulty: beginner
sources: [mml-ch02-linear-algebra, llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-appA-introduction-to-pytorch]
created: 2026-06-04
updated: 2026-10-13
---

# Vector

> **One-liner:** A vector is anything you can *add to another of its kind* and *stretch by a number* and still get the same kind of thing back.

## 🟢 Intuition (Dummy)
Think of an arrow on paper. You can lay two arrows tip-to-tail to get a new arrow (addition),
and you can make an arrow twice as long (scaling). That's a vector.

The surprising part: arrows aren't the only vectors. A **polynomial**, an **audio clip**, and a
**list of numbers** are *also* vectors, because you can add two of them and scale them and still
get the same kind of object. Math doesn't care what the thing *is* — only that it obeys those two
rules. So "vector" is a club, and the membership test is just: *can I add, and can I scale?*

### In LLMs (Raschka)
A computer can't do math on the word "fox". So an LLM gives every [[token]] its own list of numbers —
a vector — and from then on it only works with those lists. Think of each word getting a GPS
location on a giant map of meaning. Words that mean similar things end up near each other: "duck",
"eagle" and "goose" sit together, "Germany" sits next to "Berlin" (book Fig. 2.3, p. 20). The map
has many more than 2 directions — GPT-2 small uses 768 — but the idea is the same.

## 🟡 Mechanics (Practitioner)
In this book (and almost all of ML) a vector is an element of $\mathbb{R}^n$ — a tuple of $n$ real numbers:

$$
\mathbf{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \in \mathbb{R}^3
$$

The two operations, done **component-wise**:
- **Addition:** $\mathbf{a} + \mathbf{b} = \mathbf{c} \in \mathbb{R}^n$
- **Scalar multiplication:** $\lambda \mathbf{a} \in \mathbb{R}^n$ for $\lambda \in \mathbb{R}$

**Worked example:**
$$
\begin{bmatrix}1\\2\end{bmatrix} + \begin{bmatrix}3\\0\end{bmatrix} = \begin{bmatrix}4\\2\end{bmatrix},
\qquad
3\begin{bmatrix}1\\2\end{bmatrix} = \begin{bmatrix}3\\6\end{bmatrix}
$$

**In ML:** a data point with $n$ features is a vector in $\mathbb{R}^n$; an image is a vector of pixel
values; word embeddings are vectors. The whole pipeline runs on $\mathbb{R}^n$. ⚠️ On a computer,
check that your array operations actually perform *vector* operations (broadcasting can fool you).

### In LLMs: tokens become vectors (Raschka Ch. 2–4)
**Vectors as the model's internal language (§1.4, pp. 7–8; §3.1, p. 53).** The original transformer's
encoder turns input text into "a series of numerical representations or vectors that capture the
contextual information of the input", and the decoder turns these vectors back into text (Fig. 1.4).
The Fig. 1.4 caption (book p. 8) gives the book's first definition of an *embedding representation*:
"a numerical representation that captures many different factors in different dimensions". In
vector language: each coordinate $v_i$ of $\mathbf{v}\in\mathbb{R}^d$ is one "factor", and $d$ (the
number of coordinates) is the number of "dimensions".
Even the older encoder–decoder RNN squeezes the whole input sentence into one final *hidden state*,
which the book says "you can think of … as an embedding vector" (§3.1).

**Why vectors at all (§2.1, p. 18).** Text is *categorical* (a word is a symbol, not a number), so it
won't work with matrix multiplications and gradients. An *embedding* fixes this: it is "a mapping from
discrete objects … to points in a continuous vector space" (p. 19). In symbols (the book says it in
words):

$$E:\mathcal{V}\to\mathbb{R}^d,\qquad w\mapsto \mathbf{e}_w$$

- $\mathcal{V}$ = the [[vocabulary]] (the finite set of tokens). $w$ = one token in it.
- $\mathbb{R}^d$ = the space of lists of $d$ real numbers. $d$ = the [[embedding-dimension]].
- $\mathbf{e}_w$ = the vector assigned to $w$.

Example vectors printed in Fig. 2.2 ($d=3$): video → $[1.23, -0.31, 0.89]$, audio →
$[-0.15, 0.45, 2.11]$, text → $[1.78, 0.18, -2.10]$. Real sizes: smallest GPT-2 (117M/125M params)
uses $d=768$; largest GPT-3 (175B) uses $d=12{,}288$ (p. 20).

**A vector = one row of a matrix (§2.7, pp. 42–43).** PyTorch's `nn.Embedding` stores a
weight [[matrix]] $W\in\mathbb{R}^{V\times d}$; token ID $i$ just picks row $i$:

```python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(6, 3)   # vocab_size=6, output_dim=3
embedding_layer(torch.tensor([3]))
# tensor([[-0.4015,  0.9666, -1.1481]])     <- row index 3 of W
```

The same thing as a [[one-hot-encoding|one-hot vector]] times the matrix:
$\mathbf{e}_i = \mathbf{x}_i^\top W$, where $\mathbf{x}_i=(0,\dots,0,1,0,\dots,0)^\top$ has a single
1 at position $i$ ("$^\top$" = [[matrix-transpose|transpose]], turning the column into a row). The
lookup skips multiplying by all the zeros, which is why it is faster.

**Vector addition carries position (§2.8, pp. 45–47).** A token vector says *what* the token is, not
*where* it is. So GPT adds a second vector, a [[positional-embedding]], of the same length:

$$\mathbf{x}_t = E_{\text{tok}}[\text{id}_t] + E_{\text{pos}}[t]$$

- $\mathbf{x}_t$ = the input vector at position $t$ (0, 1, 2, …).
- $E_{\text{tok}}[\text{id}_t]$ = the row of the token-embedding matrix for the token at position $t$.
- $E_{\text{pos}}[t]$ = row $t$ of the positional-embedding matrix.
- $+$ = ordinary component-wise vector addition (the same "add" from the top of this page).

Book example (Fig. 2.18): $[1,1,1] + [1.1, 1.2, 1.3] = [2.1, 2.2, 2.3]$. At realistic size: 8 text
samples × 4 tokens × 256 dimensions, `token_embeddings + pos_embeddings` → shape `[8, 4, 256]`
([[broadcasting]] adds the `[4, 256]` position block to each of the 8 samples).

**Dot product = similarity (§3.3, pp. 57–60).** In self-attention the book uses six 3-D vectors for
"Your journey starts with one step": Your $=[0.43, 0.15, 0.89]$, journey $=[0.55, 0.87, 0.66]$,
starts $=[0.57, 0.85, 0.64]$, … The [[inner-product|dot product]]

$$\mathbf{x}^{(2)}\cdot\mathbf{x}^{(j)}=\sum_{k=1}^{d}x^{(2)}_k\,x^{(j)}_k$$

("multiply matching entries, then add them up"; $k$ runs over the $d$ coordinates) gives one number
per pair, e.g. $\mathbf{x}^{(1)}\cdot\mathbf{x}^{(2)} = 0.9544$. Higher = more aligned = more
"attention". The resulting [[context-vector]] is a weighted sum of vectors —
$\mathbf{z}^{(2)}=\sum_j \alpha_{2j}\mathbf{x}^{(j)} = [0.4419, 0.6515, 0.5683]$ — which uses
exactly the two vector-space operations (scale, add).

**Vectors flow through the whole model (§4.1, §4.5).** In GPT-2 124M every token stays a
**768-dimensional** vector through every [[transformer-block]]; the block changes the *content* but
keeps the *size* (p. 116). At the end, an output layer maps each vector to a
**50,257-dimensional** vector of [[logits]], one entry per vocabulary token (p. 99).

**Vector as a rank-1 tensor (App. A.2, pp. 258–259).** In PyTorch, a vector is a 1-D [[tensor]]
(`torch.tensor([1, 2, 3])`, shape `[3]`). ⚠️ Name clash: a "3D vector" has 3 entries but is still a
*rank-1* tensor (one axis). This "rank" (number of axes) is **not** the matrix rank of
[[basis-and-rank]].

## 🔴 Theory (Expert)
A vector is, formally, an **element of a [[vector-space]]** — there is no intrinsic "vectorness"
to an object, only membership in a set equipped with addition and scalar multiplication satisfying
the eight vector-space axioms. Hence the book's examples:
- **Geometric vectors** (directed segments) — concrete, drawable.
- **Polynomials** — abstract, infinite-dimensional in general.
- **Audio signals** — sampled function values.
- **$\mathbb{R}^n$** — the canonical finite-dimensional case.

Key fact: for **finite-dimensional** vector spaces there is a **1:1 correspondence** between any
kind of vector and $\mathbb{R}^n$ (via a choice of [[basis-and-rank|basis]]). This is why focusing on
$\mathbb{R}^n$ loses no generality in finite dimensions. The unifying principle is **closure**: the
set of all things reachable by adding and scaling a starting set *is* a vector space (§2.4).

### In LLMs: what the vector-space view buys you
- **Embeddings are learned coordinates.** The rows of $W\in\mathbb{R}^{V\times d}$ start as random
  numbers (book: "small, random values", p. 42) and are trained by [[backpropagation]] and
  [[gradient-descent]]. Only the rows that were looked up get a nonzero gradient, because
  $\partial L/\partial W = X^\top\,\partial L/\partial E$ with one-hot $X$ (a consequence of the
  one-hot view; not printed in the book).
- **The embedding is linear on basis vectors.** $\mathbf{x}_i^\top W$ is the [[linear-mapping]]
  $\mathbb{R}^V\to\mathbb{R}^d$ evaluated at the standard basis vector $\mathbf{x}_i$. So the
  "vectorness" of a token is borrowed: the token itself is just an index, and $W$ gives it
  coordinates.
- **Geometry is meaning, approximately.** "Similar words lie close" (Fig. 2.3) is a claim about
  [[norm|distances]] $\lVert\mathbf{e}_a-\mathbf{e}_b\rVert$ or angles (cosine similarity
  $\frac{\mathbf{e}_a^\top\mathbf{e}_b}{\lVert\mathbf{e}_a\rVert\,\lVert\mathbf{e}_b\rVert}$). The raw
  dot product used in attention also grows with vector length, which is one reason the scores are
  divided by $\sqrt{d_k}$ in [[scaled-dot-product-attention]].
- **Context vectors live in the convex hull.** With softmax weights ($\alpha_{ij}\ge0$,
  $\sum_j\alpha_{ij}=1$), each $\mathbf{z}^{(i)}$ is a [[convex-combination]] of the input vectors —
  an element of the same $\mathbb{R}^d$, inside the region spanned by the inputs.
- **Dimension is a hyperparameter, not a law.** $d$ trades expressiveness against compute (p. 19).
  Words can't be visualized in 768-D, so figures use 2-D projections
  ([[dimensionality-reduction]]); distances in the projection are only a rough guide to distances in
  the full space.

## Connections
- Builds on: *(none — foundational)*
- Leads to: [[vector-space]], [[linear-independence]], [[basis-and-rank]]
- Used by: [[system-of-linear-equations]], essentially all of ML; in LLMs: [[token-embedding]], [[positional-embedding]], [[input-embeddings]], [[context-vector]], [[self-attention]], [[logits]]
- Related (LLM side): [[embedding-layer]], [[embedding-dimension]], [[word-embeddings]], [[one-hot-encoding]], [[inner-product]], [[tensor]], [[broadcasting]]
- People: —

## Sources
- `raw/mml-book.pdf` §2 intro, pp. 17–18 (PDF pp. 22–23)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 7–8 (encoder output as vectors / embeddings).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book pp. 18–20 (embedding definition, Figs. 2.2–2.3, GPT-2 768 / GPT-3 12,288).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.2–2.3, book pp. 21–29 (tokens → IDs → vectors pipeline).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book pp. 41–44 (`nn.Embedding` lookup, one-hot equivalence, Fig. 2.16).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 43–48 (positional vectors added, Fig. 2.18, shape [8, 4, 256]).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book pp. 50–53 (RNN hidden state as an embedding vector).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book pp. 55–64 (dot product as similarity, context vector $[0.4419, 0.6515, 0.5683]$).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 92–99 (768-dim token vectors, 50,257-dim logit vectors).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–116 (transformer block preserves vector size).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` App. A.2, book pp. 258–259 (vector = rank-1 tensor, Fig. A.6; `torch.tensor([1, 2, 3])`).
