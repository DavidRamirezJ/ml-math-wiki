---
title: Self-Attention
type: concept
tags: [llm, transformer, attention, linear-algebra, deep-learning]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Self-Attention

> **One-liner:** Self-attention lets every word in a sentence look at every other word in the *same* sentence and rebuild itself as a blend of the words that matter to it.

## 🟢 Intuition (Dummy)
Take the sentence "Your journey starts with one step." The word "starts" means more
once you know it is the *journey* that starts. Self-attention gives every word a short
"meeting" with every other word in the sentence. In each meeting the word asks, "How
relevant are you to me?" Then it updates itself by mixing in a little of each other
word — a lot of the relevant ones, a little of the rest. After this step, each word's
vector carries information about its neighbors: it has become a
[[context-vector]].

The "self" means the words attend to their *own* sentence, not to some other sequence
(sidebar, book p. 56). Already in Ch. 1 the book names self-attention "a key
component of transformers and LLMs": it lets the model weigh the importance of tokens
**relative to each other**, which lets it capture
[[long-range-dependencies]] and contextual relationships (§1.4, book p. 7). The book calls self-attention "the cornerstone of every LLM
based on the transformer architecture" (book p. 55), and warns it is one of the
toughest parts of the whole book — so this page goes slowly.

## 🟡 Mechanics (Practitioner)

### Step 0 — the data
A sentence of $T$ tokens becomes $T$ embedding vectors (Ch. 2):
$\mathbf x^{(1)}, \dots, \mathbf x^{(T)}$, each $\mathbf x^{(i)}\in\mathbb R^{d}$.
Read "$\mathbb R^{d}$" as "a list of $d$ real numbers". The book's toy input, $T=6$,
$d=3$ (book p. 57):

| token | $\mathbf x^{(i)}$ |
|---|---|
| Your | [0.43, 0.15, 0.89] |
| journey | [0.55, 0.87, 0.66] |
| starts | [0.57, 0.85, 0.64] |
| with | [0.22, 0.58, 0.33] |
| one | [0.77, 0.25, 0.10] |
| step | [0.05, 0.80, 0.55] |

Stacked as rows, this is the input [[matrix]] $X\in\mathbb R^{6\times 3}$.

### Version 1 — simplified self-attention, no trainable weights (§3.3)

**Scores.** The attention score between query $i$ and input $j$ is a
[[inner-product|dot product]]:
$$\omega_{ij} = \mathbf x^{(i)}\cdot\mathbf x^{(j)} = \sum_{k=1}^{d} x^{(i)}_k\,x^{(j)}_k$$
- $\omega_{ij}$ = score of position $i$ looking at position $j$;
- $x^{(i)}_k$ = the $k$-th number in vector $\mathbf x^{(i)}$;
- $\sum_{k=1}^{d}$ = multiply matching entries and add the $d$ products.

Check by hand for "Your"·"journey": $0.43\cdot0.55 + 0.15\cdot0.87 + 0.89\cdot0.66 = 0.2365+0.1305+0.5874 = 0.9544$ — exactly the book's `tensor(0.9544)` (book p. 58). The book's sidebar: a dot product is a **measure of similarity**; the more two vectors point the same way, the bigger it is (book p. 59).

**Weights.** Turn each row of scores into weights that sum to 1 with the [[softmax]]:
$$\alpha_{ij} = \frac{\exp(\omega_{ij})}{\sum_{t=1}^{T}\exp(\omega_{it})}$$
- $\alpha_{ij}$ = attention weight (always $>0$; each row adds up to 1);
- $\exp$ = $e^{(\cdot)}$; the denominator adds the exponentials of the whole row.

The book first tries plain division by the sum (`[0.1455, 0.2278, …]`) but prefers
softmax: it handles extreme values better, has nicer gradients, and keeps weights
positive so they read like probabilities (book pp. 59–60). Use `torch.softmax`, not a
hand-written `exp/sum`, which can overflow or underflow (book p. 60; see
[[numerical-stability]]).

**Mix.** The context vector is the weighted sum of inputs:
$$\mathbf z^{(i)} = \sum_{j=1}^{T}\alpha_{ij}\,\mathbf x^{(j)}$$

For query "journey" (book pp. 58–60): scores `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]`
→ weights `[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]` → $\mathbf z^{(2)}$ = `[0.4419, 0.6515, 0.5683]`.

**All at once with matrices (§3.3.2, book pp. 62–63).**
$$\Omega = XX^\top,\qquad A = \operatorname{softmax}_{\text{row}}(\Omega),\qquad Z = AX$$
- $X^\top$ = the transpose of $X$ (rows become columns), so $XX^\top$ holds *all* $T\times T$ dot products — a [[gram-matrix]];
- $\Omega$ ("capital omega") = 6×6 score matrix; $A$ = 6×6 weight matrix (each row sums to 1);
- $Z$ = 6×3 matrix whose row $i$ is $\mathbf z^{(i)}$.

```python
attn_scores  = inputs @ inputs.T                    # (6,3)@(3,6) -> (6,6)
attn_weights = torch.softmax(attn_scores, dim=-1)   # each row sums to 1
all_context_vecs = attn_weights @ inputs            # (6,6)@(6,3) -> (6,3)
```
Row 2 of `all_context_vecs` is again `[0.4419, 0.6515, 0.5683]` (book p. 64). Matrix
products replace slow Python `for` loops (book p. 62).

### Version 2 — self-attention with trainable weights (§3.4)
This is the real thing used in the original transformer and in GPT, also called
**[[scaled-dot-product-attention]]** (book p. 64). New ingredient: three trainable
weight matrices $W_q, W_k, W_v$ that **project** each input into a query, key and value
([[query-key-value]], book p. 65):

$$\mathbf q^{(i)} = \mathbf x^{(i)}W_q,\qquad \mathbf k^{(i)} = \mathbf x^{(i)}W_k,\qquad \mathbf v^{(i)} = \mathbf x^{(i)}W_v$$
- $\mathbf x^{(i)}$ is a row vector of size $d_{in}$; each $W\in\mathbb R^{d_{in}\times d_{out}}$, so $\mathbf q,\mathbf k,\mathbf v$ have size $d_{out}$;
- **query** = "what am I looking for?", **key** = "what do I offer to be matched on?", **value** = "what content do I hand over?" — names borrowed from databases (sidebar, book p. 70).

Then score, scale, normalize, mix:
$$\omega_{ij} = \mathbf q^{(i)}\cdot\mathbf k^{(j)},\qquad
\alpha_{ij} = \operatorname{softmax}_j\!\left(\frac{\omega_{ij}}{\sqrt{d_k}}\right),\qquad
\mathbf z^{(i)} = \sum_{j}\alpha_{ij}\,\mathbf v^{(j)}$$
- $d_k$ = length of the key vectors (`keys.shape[-1]`); $\sqrt{d_k} = d_k^{0.5}$ (book p. 68);
- dividing by $\sqrt{d_k}$ keeps scores from getting huge. With big scores, softmax turns into a near **step function** and its gradients go to almost zero, so learning stalls — and GPT embedding sizes are often over 1,000 (sidebar, book p. 69).

**Matrix form (Fig. 3.18, book pp. 71–72):**
$$Q = XW_q,\quad K = XW_k,\quad V = XW_v,\qquad Z = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
$X$: $T\times d_{in}$; $Q,K,V$: $T\times d_{out}$; $QK^\top$ and the weights: $T\times T$; $Z$: $T\times d_{out}$.

**Book numbers** ($d_{in}=3$, $d_{out}=2$, `torch.manual_seed(123)`, book pp. 66–70):
`query_2 = [0.4306, 1.4551]`; $\omega_{22}$ = `1.8524`; scores
`[1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]`; weights (after $/\sqrt2$ and softmax)
`[0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]`; context vector `[0.3061, 0.8210]`.

**The compact class (Listing 3.2, book pp. 72–73):**
```python
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```
`nn.Linear` (bias off) does the same matrix product as `nn.Parameter(torch.rand(...))`
in `SelfAttention_v1`, but with a better [[weight-initialization]] scheme (book
pp. 71–72). Careful: `nn.Linear` stores its weight **transposed**, shape
$(d_{out}, d_{in})$ — the hint for Exercise 3.1 (book p. 73; solution p. 301).

**Two kinds of "weights" (sidebar, book p. 66).** *Weight parameters* ($W_q,W_k,W_v$)
are learned during training and then fixed. *Attention weights* ($\alpha_{ij}$) are
recomputed for every input — dynamic and context-specific.

### Versions 3 and 4 — how GPT actually uses it
- **Causal:** a [[causal-attention-mask]] blocks attention to future tokens (§3.5).
- **Multi-head:** several self-attention heads run in parallel, each with its own
  $W_q,W_k,W_v$, and their outputs are concatenated and projected (§3.6). See
  [[multi-head-attention]]. Smallest GPT-2: **12 heads**, embedding size **768**
  (head size $768/12=64$); largest GPT-2: **25 heads**, size **1,600** (book p. 90).
  Exercise 3.3: a GPT-2-size module (`d_in = d_out = 768`, 12 heads, context 1,024)
  has $3\cdot768^2 + (768^2+768) = 2{,}360{,}064$ parameters (book pp. 91, 301–302).
- **In the transformer block** (Ch. 4), masked multi-head self-attention mixes
  information **across positions**, while the [[feed-forward-network]] works on each
  position separately; input and output shapes match (e.g. `[2, 4, 768]` →
  `[2, 4, 768]`), so blocks stack (book pp. 113–116). See [[transformer-block]].

## 🔴 Theory (Expert)
- **Permutation equivariance → need for positions.** Without positional information,
  shuffling the input rows shuffles the output rows the same way: if $P$ is a
  permutation matrix, $\operatorname{softmax}(PXW_q(PXW_k)^\top/\sqrt{d_k})PXW_v = P\,Z$.
  So self-attention is position-agnostic (book pp. 43, 45), and LLMs add
  [[positional-embedding|positional embeddings]] to the token embeddings.
- **Why $\sqrt{d_k}$ exactly.** If the entries of $\mathbf q$ and $\mathbf k$ are
  independent with mean 0 and variance 1, then
  $\operatorname{Var}(\mathbf q\cdot\mathbf k)=\sum_{m=1}^{d_k}\operatorname{Var}(q_mk_m)=d_k$,
  so the standard deviation is $\sqrt{d_k}$. Dividing by it brings scores back to unit
  scale ([[covariance-and-variance]]). The book gives only the gradient argument
  (softmax saturation, book p. 69); this variance argument is the standard rationale.
- **Softmax saturation.** The softmax Jacobian is
  $\partial\alpha_i/\partial\omega_j=\alpha_i(\delta_{ij}-\alpha_j)$. When one weight
  approaches 1 and the rest 0, every entry goes to 0 — the "small gradients" the book
  warns about ([[vanishing-and-exploding-gradients]]).
- **Gram-matrix structure.** In the simplified version, $\Omega = XX^\top$ is
  symmetric and positive semidefinite (book's printed 6×6 is symmetric, e.g.
  $\Omega_{12}=\Omega_{21}=0.9544$, book pp. 62–63). With learned $W_q\ne W_k$,
  $QK^\top = XW_qW_k^\top X^\top$ is a *bilinear form* $\mathbf x_i^\top M\mathbf x_j$
  with $M=W_qW_k^\top$ of rank $\le d_k$; it is generally **not** symmetric, so
  "$i$ attends to $j$" need not equal "$j$ attends to $i$". See
  [[symmetric-positive-definite-matrix]], [[basis-and-rank]].
- **Kernel-smoother view.** $\mathbf z^{(i)}=\sum_j\frac{\kappa(\mathbf q_i,\mathbf k_j)}{\sum_t\kappa(\mathbf q_i,\mathbf k_t)}\mathbf v_j$
  with $\kappa(\mathbf q,\mathbf k)=\exp(\mathbf q^\top\mathbf k/\sqrt{d_k})$ is a
  Nadaraya–Watson estimator with an exponential kernel (background; not in the book;
  cf. [[kernel-trick]]). Every output is a [[convex-combination]] of value vectors.
- **Complexity.** $QK^\top$ costs $O(T^2d_k)$ time and $O(T^2)$ memory per head. Exact
  speedups such as [[flashattention]] reorder memory access without changing the math
  (book p. 291). Appendix B also notes PyTorch's `scaled_dot_product_attention` and
  `MultiHeadAttention` (book p. 292), and He & Hofmann (2023) showing good results even
  without the value matrix and output projection (book p. 292). For contexts under
  32,000 tokens, most LLM compute is in the feed-forward layers, not attention
  (de Vries, book p. 293).
- **Contrast with Bahdanau attention.** Traditional attention relates two different
  sequences (decoder state → encoder states); self-attention relates positions in one
  sequence (sidebar, book p. 56). The book notes the self-attention idea also applies
  beyond text, e.g. pixels in an image (book p. 56).

## Connections
- Builds on: [[vector]], [[inner-product]], [[matrix]], [[matrix-multiplication]], [[matrix-transpose]], [[softmax]], [[linear-mapping]], [[token-embedding]], [[attention-mechanism]]
- Leads to: [[scaled-dot-product-attention]], [[causal-attention]], [[causal-attention-mask]], [[multi-head-attention]], [[transformer-block]], [[gpt-architecture]]
- Related: [[context-vector]], [[query-key-value]], [[attention-scores]], [[attention-weights]], [[gram-matrix]], [[convex-combination]], [[positional-embedding]], [[numerical-stability]], [[flashattention]], [[pytorch-nn-module]], [[linear-layer]]
- Used by: [[large-language-model]], [[transformer-architecture]], [[bert]], [[decoder-only-transformer]]
- People: [[ashish-vaswani]], [[dzmitry-bahdanau]], [[tri-dao]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book p. 7 (self-attention weighs tokens relative to each other; long-range dependencies)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 43–45 (self-attention is position-agnostic)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1–3.2, book pp. 50–55 (four variants; definition, Fig. 3.6)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book pp. 55–64 (simplified self-attention; tensors pp. 57–64; Figs. 3.7–3.12)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4, book pp. 64–74 (trainable $W_q,W_k,W_v$; $\sqrt{d_k}$ sidebar p. 69; Listings 3.1–3.2; Fig. 3.18; Exercise 3.1)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5–3.6, book pp. 74–91 (causal and multi-head extensions; GPT-2 sizes p. 90; Exercise 3.3 p. 91)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–116 (attention inside the transformer block)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book pp. 291–293 (Vaswani 2017; FlashAttention; PyTorch SDPA; He & Hofmann 2023; de Vries)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book pp. 301–302 (solutions to Exercises 3.1–3.3)
