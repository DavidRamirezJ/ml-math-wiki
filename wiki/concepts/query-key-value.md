---
title: Query, Key and Value Vectors
type: concept
tags: [attention, transformer, linear-algebra, llm]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Query, Key and Value Vectors

> **One-liner:** Each token makes three versions of itself — a question (query), a label (key) and a content package (value) — so it can look up what it needs from the other tokens.

## 🟢 Intuition (Dummy)
Think of a library. You walk in with a **question** ("books about journeys?"). Every book
has a **label** on its spine. You compare your question with every label, and the
better the match, the more of that book's **content** you take home.

Self-attention does the same thing for every token in a sentence:
- **Query** — "what am I looking for?" (the token currently doing the looking);
- **Key** — "what do I offer? here's my label" (every token has one);
- **Value** — "here's my actual content" (what gets passed along once a key matches).

The book says these names are borrowed from information retrieval and databases (book
p. 70). The twist: in a database, a query matches *exactly one* key. In attention, a query
matches *every* key a little bit, and the result is a weighted blend of all the values.

Why not just use the token embedding for all three jobs (as the simplified version in
§3.3 did)? Because the jobs are different. Three separate, **trainable** matrices let the
model learn *how* to ask, *how* to advertise, and *what* to hand over — and those are
learned during training (Ch. 5).

## 🟡 Mechanics (Practitioner)

### The three projections
$$q^{(i)} = x^{(i)} W_q, \qquad k^{(i)} = x^{(i)} W_k, \qquad v^{(i)} = x^{(i)} W_v$$

Symbol by symbol:
- $x^{(i)}$ = the input embedding of token $i$, a row vector of length $d_{in}$.
- $W_q, W_k, W_v$ = three trainable **weight matrices**, each of shape $d_{in}\times d_{out}$.
- $q^{(i)}, k^{(i)}, v^{(i)}$ = the query, key and value vectors of token $i$ (length $d_{out}$).
- Multiplying a row vector by a matrix is a [[linear-mapping]] — a "projection" into a new space.

All tokens at once, with $X$ the $T\times d_{in}$ input matrix:
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v \quad(\text{each } T\times d_{out})$$

### The book's example (§3.4.1, book pp. 65–70)
- Sentence: "Your journey starts with one step", $T = 6$ tokens, $d_{in} = 3$, $d_{out} = 2$.
  (In GPT-like models usually $d_{in} = d_{out}$; the book picks different sizes to make it
  easier to follow, book p. 66.)
- Query token: $x^{(2)}$ = "journey" = [0.55, 0.87, 0.66].

```python
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query       # tensor([0.4306, 1.4551])
keys    = inputs @ W_key      # shape [6, 2]
values  = inputs @ W_value    # shape [6, 2]
```
(`requires_grad=False` is only to keep the printout tidy; for training it would be `True`,
book p. 66.)

Even to compute one context vector we need the keys and values of **all** tokens
(book p. 67).

### How the three are used
$$\underbrace{\omega_{2i} = q^{(2)}\cdot k^{(i)}}_{\text{score}} \;\to\; \underbrace{\alpha_{2i} = \operatorname{softmax}\!\Big(\tfrac{\omega_2}{\sqrt{d_k}}\Big)_i}_{\text{weight}} \;\to\; \underbrace{z^{(2)} = \sum_i \alpha_{2i}\, v^{(i)}}_{\text{context vector}}$$

- **Query · key** → [[attention-scores]]: $\omega_{22} = q^{(2)}\cdot k^{(2)} = 1.8524$, and the full row is `[1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]` (book p. 68).
- **Scale + [[softmax]]** → [[attention-weights]]: `[0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]` (book p. 69).
- **Weights × values** → [[context-vector]]: $z^{(2)}$ = `[0.3061, 0.8210]` (book p. 70).

In matrix form (Fig. 3.18, book p. 72):
$$Z = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$
- $QK^\top$ = every query dotted with every key ($T\times T$).
- $\sqrt{d_k}$ = square root of the key dimension (`keys.shape[-1]`).
- $V$ = the value matrix; $Z$ = all context vectors ($T\times d_{out}$).

### As a PyTorch class (Listing 3.2, book pp. 72–73)
```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```
`nn.Linear` without bias is just a matrix multiply, but it has a better weight
initialization than `torch.rand`, which makes training more stable (book pp. 71–72).
Note: `nn.Linear` stores its weight **transposed**, shape $(d_{out}, d_{in})$, and computes
$xW^\top$ — the hint for Exercise 3.1 (book p. 73). `qkv_bias` adds a [[bias-vector]]; the
pretrained GPT-2 weights use `qkv_bias=True` (Ch. 5).

### In multi-head attention (§3.6)
Each head has its own $W_q^{(h)}, W_k^{(h)}, W_v^{(h)}$ (book p. 83). The efficient version
uses **one** big $W_q$ of size $d_{in}\times d_{out}$ (one matrix multiplication) and then splits the result into
`num_heads` chunks of size `head_dim = d_out // num_heads`
(`.view(b, num_tokens, num_heads, head_dim)` then `.transpose(1, 2)`) (book pp. 86–88).
For GPT-2 small: $d_{in} = d_{out} = 768$, 12 heads, so `head_dim` = 64. See
[[multi-head-attention]], [[weight-splitting-multi-head-attention]].

## 🔴 Theory (Expert)
- **Factored bilinear score.** $q^{(i)}\cdot k^{(j)} = x^{(i)} W_q W_k^\top {x^{(j)}}^\top$. The
  learned "compatibility" matrix $M = W_qW_k^\top$ has rank $\le d_{out}$ (per head,
  $\le$ `head_dim`), so each head scores tokens through a low-rank bilinear form. Splitting
  $W_q$ and $W_k$ (instead of learning $M$ directly) is cheaper and allows non-symmetric
  relations ("A looks for B" ≠ "B looks for A"). (Our analysis.)
- **Why values are separate.** Keys decide *where* to look; values decide *what* is
  retrieved. Decoupling lets the "address" and the "content" live in different subspaces.
  The output $Z = AV = AXW_v$ is linear in $X$ for fixed weights $A$; the non-linearity of
  attention comes only from the softmax over $QK^\top$.
- **Soft key–value memory.** With a one-hot weight row (hard argmax), $Z$ would return a
  single value — a classical dictionary lookup. Softmax makes the lookup differentiable,
  which is what lets $W_q$, $W_k$, $W_v$ be learned by [[backpropagation]].
- **Dimension bookkeeping.** $Q, K \in \mathbb R^{T\times d_k}$ must share $d_k$ for $QK^\top$
  to exist; $V$ may in principle have a different width $d_v$ — the book uses $d_k = d_v = d_{out}$.
- **Parameter count.** Three bias-free $d\times d$ projections cost $3d^2$ weights; for
  GPT-2 small $3\cdot768^2 = 1{,}769{,}472$ per block, plus $768^2+768 = 590{,}592$ for the
  output projection (derived for Exercise 3.3, book p. 91; arithmetic ours).
- **Self vs. cross attention.** In self-attention $Q$, $K$, $V$ all come from the same $X$
  (book p. 56 sidebar). In encoder–decoder "traditional" attention, queries come from the
  decoder and keys/values from the encoder (cf. [[bahdanau-attention]], [[encoder-decoder-architecture]]).

## Connections
- Builds on: [[vector]], [[matrix-multiplication]], [[linear-mapping]], [[linear-layer]], [[inner-product]], [[token-embedding]]
- Leads to: [[attention-scores]], [[attention-weights]], [[context-vector]], [[scaled-dot-product-attention]], [[multi-head-attention]]
- Used by: [[self-attention]], [[causal-attention]], [[transformer-block]], [[gpt-architecture]]
- Related: [[weight-initialization]], [[bias-vector]], [[low-rank-approximation]], [[pytorch-nn-module]]
- People: [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3 sidebar "The 'self' in self-attention", book p. 56.
- Same, §3.4.1, book pp. 65–70 (Figs. 3.14–3.17; sidebars "Weight parameters vs. attention weights" p. 66 and "Why query, key, and value?" p. 70).
- Same, §3.4.2, book pp. 70–73 (Listings 3.1–3.2, `nn.Linear`, Exercise 3.1, Fig. 3.18).
- Same, §3.6.1–3.6.2, book pp. 82–91 (per-head projections, weight splitting, Exercise 3.3).
