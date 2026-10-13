---
title: Scaled Dot-Product Attention
type: concept
tags: [attention, transformer, llm, linear-algebra, softmax]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Scaled Dot-Product Attention

> **One-liner:** Every token asks a question, every token offers a label and some content, and each token's new meaning is a blend of everyone's content, weighted by how well their label answers its question.

## 🟢 Intuition (Dummy)
Picture a library. You walk in with a **question** ("what do I need to understand the word *journey*?").
Every book on the shelf has a **label** on its spine and some **content** inside. You compare your
question with every label, give each book a "match score", and then you don't pick just one book:
you read a little from every book, reading *more* from the books whose labels matched best.
What you walk out with is a blend of everything, tilted toward the most relevant books.

That is exactly what each token does inside an LLM. The question is the **query**, the spine label
is the **key**, and the content is the **value**. The "match score" is a [[inner-product|dot product]]
(a similarity number). The blend is the new, context-aware version of the token, its
[[context-vector]].

The "scaled" part is a small piece of hygiene. When vectors are long (GPT-like models use embeddings with more than 1,000 numbers, p. 69;
GPT-2 small uses 64 per head), dot products get big, and big scores make the blending step pick a single book almost
100% of the time. That makes the model stubborn and slow to learn. So we divide the scores by a
fixed number first, which keeps the blend soft and trainable. The book says this mechanism is just
another name for the self-attention used in the original transformer, the GPT models and most
other popular LLMs (Raschka §3.4, p. 64).

## 🟡 Mechanics (Practitioner)

### The formula
In matrix form (Raschka Fig. 3.18, p. 72):
$$
Q = XW_q,\qquad K = XW_k,\qquad V = XW_v,\qquad
Z \;=\; \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V .
$$

Reading it symbol by symbol:
- $X \in \mathbb{R}^{T\times d_{in}}$ = the input matrix: $T$ rows (one per token), each row a
  $d_{in}$-dimensional [[token-embedding|embedding]]. "$\in \mathbb{R}^{T\times d_{in}}$" just means "is a table of real numbers with $T$ rows and $d_{in}$ columns".
- $W_q, W_k, W_v \in \mathbb{R}^{d_{in}\times d_{out}}$ = three **trainable weight matrices** (learned during training).
- $Q, K, V \in \mathbb{R}^{T\times d_{out}}$ = the **query**, **key** and **value** matrices. Row $i$ of $Q$ is $\mathbf q^{(i)} = \mathbf x^{(i)}W_q$, and so on.
- $K^\top$ = the [[matrix-transpose|transpose]] of $K$ (rows become columns), so that $QK^\top$ is a $T\times T$ table.
- $QK^\top$ = the **attention scores** $\omega$; entry $(i,j)$ is the dot product $\mathbf q^{(i)}\cdot\mathbf k^{(j)}$, i.e. "how well token $j$'s key answers token $i$'s query".
- $d_k$ = the length of each key vector (`keys.shape[-1]`); $\sqrt{d_k} = d_k^{0.5}$ is the scaling divisor.
- $\operatorname{softmax}(\cdot)$ is applied **row by row** (`dim=-1`): each row becomes positive numbers that sum to 1 — the **attention weights** $A$.
- $Z = AV \in \mathbb{R}^{T\times d_{out}}$ = the **context vectors**; row $i$ is $\mathbf z^{(i)} = \sum_j \alpha_{ij}\mathbf v^{(j)}$, a [[convex-combination|weighted average]] of the value vectors.

For a single query (token 2, "journey"), the same thing written with sums (pp. 67–70):
$$
\omega_{2j} = \mathbf q^{(2)}\cdot\mathbf k^{(j)},\qquad
\alpha_{2j} = \frac{\exp\!\big(\omega_{2j}/\sqrt{d_k}\big)}{\sum_{t=1}^{T}\exp\!\big(\omega_{2t}/\sqrt{d_k}\big)},\qquad
\mathbf z^{(2)} = \sum_{j=1}^{T}\alpha_{2j}\,\mathbf v^{(j)} .
$$
- $\omega_{2j}$ ("omega two-j") = the raw, unnormalized score between query 2 and key $j$.
- $\exp(\cdot)$ = $e^{(\cdot)}$, which makes every score positive.
- The fraction divides by the sum over all $t=1\dots T$, so the $\alpha_{2j}$ ("alpha two-j") add up to 1.
- $\mathbf z^{(2)}$ = the new vector for "journey": each value vector $\mathbf v^{(j)}$ multiplied by its weight, then all added up.

### The four steps, with the book's real numbers (§3.4.1, pp. 65–70)
Sentence: "Your journey starts with one step." ($T=6$, $d_{in}=3$, $d_{out}=2$, `torch.manual_seed(123)`).
Inputs (from §3.3): Your [0.43, 0.15, 0.89], journey [0.55, 0.87, 0.66], starts [0.57, 0.85, 0.64],
with [0.22, 0.58, 0.33], one [0.77, 0.25, 0.10], step [0.05, 0.80, 0.55].

| Step | What happens | Book's result |
|---|---|---|
| 1. Project | $\mathbf q^{(2)} = \mathbf x^{(2)}W_q$; all keys/values via $XW_k$, $XW_v$ | `query_2 = [0.4306, 1.4551]`; `keys`, `values` shape (6, 2) |
| 2. Score | $\omega_{2j} = \mathbf q^{(2)}\cdot\mathbf k^{(j)}$ | `[1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]` ($\omega_{22}=1.8524$) |
| 3. Scale + softmax | divide by $\sqrt{2}\approx1.414$, then softmax | `[0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]` |
| 4. Blend values | $\mathbf z^{(2)} = \sum_j\alpha_{2j}\mathbf v^{(j)}$ | `context_vec_2 = [0.3061, 0.8210]` |

Note: even to get one context vector, you need the keys and values of **all** tokens (p. 67).

**What the scaling actually does here** (our own recomputation, not printed in the book):
softmax of the *unscaled* scores is `[0.1401, 0.2507, 0.2406, 0.1157, 0.0687, 0.1842]` — already
a bit sharper than the scaled `[0.1500, 0.2264, …]`. If the scores were 20× bigger (as happens with
long vectors), softmax gives `[0.0, 0.6945, 0.304, 0.0, 0.0, 0.0015]`: nearly a hard pick. Dividing
by $\sqrt{d_k}$ keeps us in the soft, learnable regime.

### The compact PyTorch version (Listing 3.1, pp. 70–71)
```python
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T          # omega, (T, T)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values     # (T, d_out)
        return context_vec
```
With seed 123 the second output row is `[0.3061, 0.8210]`, the same as the step-by-step result (p. 71).
`SelfAttention_v2` (Listing 3.2, pp. 72–73) swaps `nn.Parameter(torch.rand(...))` for
`nn.Linear(d_in, d_out, bias=qkv_bias)`, which has a better [[weight-initialization]] scheme.
`nn.Linear` stores its weight transposed, as $(d_{out}, d_{in})$, and computes $xW^\top$ (p. 73, Exercise 3.1).

### How the formula grows into the real LLM attention
1. **[[causal-attention|Causal mask]]** (§3.5, pp. 74–78): set the scores of future tokens to $-\infty$ before softmax:
   $$A = \operatorname{softmax}\!\left(\frac{QK^\top + M_{-\infty}}{\sqrt{d_k}}\right),\qquad (M_{-\infty})_{ij} = \begin{cases}-\infty & j>i\\ 0 & j\le i\end{cases}$$
   $M_{-\infty}$ = the mask matrix; $j>i$ means "token $j$ comes after token $i$". Because $e^{-\infty}=0$, future tokens get weight exactly 0 (p. 77).
2. **[[dropout|Dropout]]** on the attention weights during training (§3.5.2, p. 78).
3. **[[multi-head-attention|Multiple heads]]** (§3.6): run the formula $h$ times in parallel on slices of width $d_h = d_{out}/h$, and scale by $\sqrt{d_h}$, the **head** dimension (`keys.shape[-1]`, p. 87). See [[weight-splitting-multi-head-attention]].

**GPT-2 numbers** (p. 90): the smallest GPT-2 has 12 heads and embedding size 768, so each head has
$d_h = 768/12 = 64$ and the scores are divided by $\sqrt{64} = 8$. The largest has 25 heads and size
1,600 (again $d_h = 64$; our arithmetic).

**Fast implementations** (Appendix B, pp. 291–292): PyTorch's `scaled_dot_product_attention`
function computes self-attention and causal attention and supports **FlashAttention**
([[flashattention]], Dao et al. 2022; FlashAttention-2, Dao 2023). FlashAttention is "mathematically
the same" as standard self-attention; it just optimizes memory access patterns. The book notes the
function is "beta and subject to change", and that PyTorch also has an efficient `MultiHeadAttention`
class built on it (p. 292).

## 🔴 Theory (Expert)

**Why $\sqrt{d_k}$: the book's argument.** The sidebar on p. 69 says the scaling improves training
by **avoiding small gradients**. GPT-like embedding sizes are typically above 1,000, large dot
products push softmax to behave "more like a step function", and the gradients then approach zero,
which can "drastically slow down learning or cause training to stagnate".

**Why softmax saturation kills gradients** (standard math, not in the book). For $\mathbf p = \operatorname{softmax}(\mathbf s)$,
$$\frac{\partial p_i}{\partial s_j} = p_i(\delta_{ij} - p_j),\qquad J = \operatorname{diag}(\mathbf p) - \mathbf p\mathbf p^\top .$$
- $\delta_{ij}$ = 1 if $i=j$, else 0; $J$ = the [[jacobian]] of softmax.
If $\mathbf p$ is close to one-hot ($p_k\approx1$), every entry $p_i(\delta_{ij}-p_j)$ is close to 0, so almost no
gradient flows back to the scores — exactly the "step function" behaviour the book describes.

**Why $\sqrt{d_k}$ is the right size: the variance argument** (the original transformer paper's
reasoning; the book states only the gradient argument). Suppose the $d_k$ components of $\mathbf q$
and $\mathbf k$ are independent [[random-variable|random variables]] with mean 0 and variance 1. Then
$$\mathbb E[\mathbf q\cdot\mathbf k] = 0,\qquad \operatorname{Var}(\mathbf q\cdot\mathbf k) = \sum_{m=1}^{d_k}\operatorname{Var}(q_m k_m) = d_k,$$
so the typical score size grows like $\sqrt{d_k}$, and $\operatorname{Var}\!\big(\mathbf q\cdot\mathbf k/\sqrt{d_k}\big) = 1$ whatever the dimension.
A quick simulation (ours, 20,000 random pairs) gives variances ≈ 2.0, 63.3 and 759.3 for
$d_k = 2, 64, 768$, and ≈ 0.99 after dividing by $\sqrt{d_k}$. See [[covariance-and-variance]].

**Scaling is a fixed temperature.** $\operatorname{softmax}(\mathbf s/\tau)$ with $\tau = \sqrt{d_k}$ is the
same operation as [[temperature-scaling]] in decoding, only here the temperature is fixed by the
architecture rather than chosen by the user.

**Structure of the score matrix.** $QK^\top = XW_qW_k^\top X^\top$, so $\omega_{ij} = \mathbf x^{(i)}\,(W_qW_k^\top)\,\mathbf x^{(j)\top}$ is a learned
**bilinear form**. With $W_q = W_k = I$ it collapses to the [[gram-matrix]] $XX^\top$ of §3.3, which is
symmetric; with trainable $W_q \ne W_k$ the scores are generally **not** symmetric (token $i$ can attend to $j$
more than $j$ attends to $i$). The rank of $W_qW_k^\top$ is at most $d_k$.

**Soft dictionary lookup.** The book borrows the names from databases (p. 70): the query probes,
keys index, values are retrieved. A hard lookup returns one value; attention returns a
[[convex-combination]] of all values, which is differentiable and so trainable by [[backpropagation]].
(Viewing this as a kernel smoother with kernel $\exp(\mathbf q\cdot\mathbf k/\sqrt{d_k})$ is outside the book.)

**Cost and symmetry facts** (outside the book): computing $QK^\top$ costs $O(T^2 d_k)$ time and the
$T\times T$ weight matrix needs $O(T^2)$ memory — the bottleneck FlashAttention attacks. Without
[[positional-embedding|positional information]] the operation is *permutation-equivariant*: shuffle
the input rows and the output rows shuffle the same way, which is why LLMs must add positional embeddings.

## Connections
- Builds on: [[inner-product]] (dot product), [[matrix-multiplication]], [[matrix-transpose]], [[softmax]], [[linear-mapping]], [[query-key-value]], [[attention-scores]], [[attention-weights]]
- Leads to: [[causal-attention]], [[multi-head-attention]], [[weight-splitting-multi-head-attention]], [[transformer-block]], [[flashattention]]
- Related math: [[gram-matrix]], [[convex-combination]], [[covariance-and-variance]], [[temperature-scaling]], [[jacobian]], [[numerical-stability]]
- Used by: [[self-attention]], [[gpt-architecture]], [[transformer-architecture]], [[large-language-model]]
- People: [[ashish-vaswani]], [[dzmitry-bahdanau]], [[tri-dao]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (intro, 3.4.1, 3.4.2; Figs. 3.13–3.18; Listings 3.1–3.2; sidebars "Weight parameters vs. attention weights", "The rationale behind scaled-dot product attention", "Why query, key, and value?"), book pp. 64–74.
- Same PDF §3.3 (input tensor, dot product as similarity), book pp. 55–64; §3.5.1 (masking with $-\infty$), book pp. 75–78; §3.6.2 ($\sqrt{d_h}$ per head, GPT-2 sizes), book pp. 86–90; chapter 3 summary, book p. 91.
- Same PDF Appendix B, chapter 3 references (Vaswani et al. 2017; Dao et al. 2022; Dao 2023; PyTorch `scaled_dot_product_attention` docs), book pp. 291–292.
