---
title: Causal Attention Mask
type: concept
tags: [llm, attention, transformer, linear-algebra, pytorch]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Causal Attention Mask

> **One-liner:** The causal mask is a triangle-shaped "no peeking" rule that stops each word from looking at the words that come after it.

## 🟢 Intuition (Dummy)
An LLM learns by guessing the next word. If, during training, it could simply *look*
at the next word, the game would be pointless — like a quiz where the answer is
printed under the question. The book's Fig. 2.12 shows the rule already in Ch. 2:
"The LLM can't access words past the target" (book p. 35).

Inside attention, the **causal mask** enforces this. Draw the attention weights as a
table: rows are "the word that is looking", columns are "the word being looked at".
The mask blanks out everything **above the diagonal** — every "future" word. So
"Your" (word 1) sees only itself, "journey" sees "Your" and itself, and so on; only
the last word sees the whole sentence (book pp. 74–75, Fig. 3.19). Attention with
this mask is called **causal attention** or **masked attention** (book p. 74).

## 🟡 Mechanics (Practitioner)

### The mask as a matrix
For $T$ tokens, the "allowed" pattern is lower-triangular:
$$M_{ij} = \mathbb 1[\,j\le i\,] = \begin{cases}1 & j\le i \ \ (\text{past or current})\\ 0 & j>i\ \ (\text{future})\end{cases}$$
Symbol by symbol: $i$ = row = the current (query) token; $j$ = column = the token
being attended to; $\mathbb 1[\cdot]$ = "1 if the condition is true, else 0".
`torch.tril(torch.ones(6, 6))` builds it (book p. 76):
```
[[1,0,0,0,0,0],
 [1,1,0,0,0,0],
 [1,1,1,0,0,0],
 [1,1,1,1,0,0],
 [1,1,1,1,1,0],
 [1,1,1,1,1,1]]
```

### Way A — softmax, zero out, renormalize (Fig. 3.20, book pp. 75–76)
1. Compute the usual weights $A=\operatorname{softmax}(QK^\top/\sqrt{d_k})$. Book output (6×6, rows sum to 1), first rows `[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510]`, `[0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477]`, …
2. Multiply element-wise by the mask: $A\odot M$ (the "$\odot$" means multiply entry by entry). Row 2 becomes `[0.2041, 0.1659, 0, 0, 0, 0]`.
3. Divide each row by its new sum:
$$\tilde\alpha_{ij}=\frac{\alpha_{ij}M_{ij}}{\sum_{k}\alpha_{ik}M_{ik}}$$
Row 2: $0.2041/(0.2041+0.1659)=0.2041/0.3700\approx0.5517$, giving `[0.5517, 0.4483, 0, 0, 0, 0]`.

```python
mask_simple = torch.tril(torch.ones(context_length, context_length))
masked_simple = attn_weights * mask_simple
row_sums = masked_simple.sum(dim=-1, keepdim=True)   # shape (6,1) -> broadcasts
masked_simple_norm = masked_simple / row_sums
```
Result (book p. 76):
```
[[1.0000, 0,      0,      0,      0,      0     ],
 [0.5517, 0.4483, 0,      0,      0,      0     ],
 [0.3800, 0.3097, 0.3103, 0,      0,      0     ],
 [0.2758, 0.2460, 0.2462, 0.2319, 0,      0     ],
 [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0     ],
 [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]]
```
The last row is unchanged — it had nothing to mask.

### Way B — the efficient one: −∞ then softmax (Fig. 3.21, book pp. 77–78)
Put $-\infty$ on every future **score** before softmax:
$$A=\operatorname{softmax}\!\left(\frac{QK^\top + M_{-\infty}}{\sqrt{d_k}}\right),\qquad (M_{-\infty})_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}$$
Why it works: softmax uses $e^{\omega}$, and $e^{-\infty}=0$, so future positions get
exactly zero weight while the row still sums to 1 — no extra renormalization step
(book p. 77). (Dividing $-\infty$ by $\sqrt{d_k}$ is still $-\infty$.)
```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
```
- `torch.triu(..., diagonal=1)` = 1s **strictly above** the diagonal (the future); `.bool()` turns it into True/False; `masked_fill` writes `-inf` where True.
- Masked scores (book p. 77): `[[0.2899, -inf, …], [0.4656, 0.1723, -inf, …], …]`. After softmax: *identical* to Way A's matrix above (book p. 78).

### Inside the attention class (Listing 3.3, book p. 81)
```python
self.register_buffer('mask',
    torch.triu(torch.ones(context_length, context_length), diagonal=1))
...
attn_scores = queries @ keys.transpose(1, 2)              # (b, T, T)
attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
attn_weights = self.dropout(attn_weights)
```
- The mask is built once at the **maximum** `context_length` and sliced to the actual
  `num_tokens`, so shorter inputs work too (book p. 81).
- `masked_fill_` with a trailing underscore works **in place**, avoiding a memory copy (book p. 81).
- **`register_buffer`**: buffers move automatically to the right device (CPU/GPU)
  together with the model, which avoids device-mismatch errors (book p. 81). A buffer
  is saved module state but **not** a trainable parameter, so the optimizer never
  changes the mask (this last point is background, not stated in the book).
- In `MultiHeadAttention` the same $(T\times T)$ mask is broadcast over the batch and
  head dimensions of the $(b, h, T, T)$ scores (book p. 87).
- Usage: on the duplicated batch of shape `[2, 6, 3]` (`torch.stack((inputs, inputs))`),
  `CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)` returns context
  vectors of shape `[2, 6, 2]` (book pp. 80–82).
- Dropout in transformer attention is typically applied either after computing the
  attention weights or after multiplying them with the values; the book uses the first,
  "more common" variant (book p. 78). Dropout outputs may differ across operating
  systems (book p. 80).
- Order inside the module: mask → softmax → [[dropout]] → multiply by values. The
  full forward pass is
  $Z=\operatorname{Dropout}\!\big(\operatorname{softmax}((QK^\top+M_{-\infty})/\sqrt{d_k})\big)V$.

### Where the same trick shows up again
- **Padding** (§2.4): a mask keeps padded tokens from being attended to (book p. 32).
- **[[top-k-sampling]]** (§5.3): non-top-$k$ logits are set to $-\infty$ so softmax gives
  them probability 0 — "the same trick" as the causal mask (book p. 156). Example:
  logits `[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]`, $k=3$ →
  `[4.51, -inf, -inf, 6.75, -inf, -inf, -inf, 6.28, -inf]` → softmax
  `[0.06, 0, 0, 0.57, 0, 0, 0, 0.36, 0]`.
- **Classification fine-tuning** (§6.5): under the causal mask, only the last token
  attends to all tokens, so it carries the most information and is the one used for
  the class prediction (Fig. 6.12, "Do you have time": rows `1.0` / `0.55 0.45` /
  `0.38 0.30 0.32` / `0.27 0.24 0.24 0.25`, book pp. 188–190).

## 🔴 Theory (Expert)
- **No information leakage — proof.** The book's sidebar worries that the future
  scores were inside the original softmax denominator, then shows renormalizing is the
  same as a fresh softmax over the unmasked subset (book pp. 76–77). Write
  $\alpha_{ij}=e^{s_{ij}}/S_i$ with $S_i=\sum_{k=1}^{T}e^{s_{ik}}$. For $j\le i$:
  $$\frac{\alpha_{ij}}{\sum_{k\le i}\alpha_{ik}}=\frac{e^{s_{ij}}/S_i}{\sum_{k\le i}e^{s_{ik}}/S_i}=\frac{e^{s_{ij}}}{\sum_{k\le i}e^{s_{ik}}}.$$
  The full denominator $S_i$ — the only place future scores appear — cancels. So Way A
  and Way B are mathematically identical, and no future token influences the result.
  This is the same fact as "conditioning a distribution on an event = restricting and
  renormalizing" ([[probability-distribution]]).
- **Why "causal".** With the mask, the output at position $i$ is a function of
  $\mathbf x^{(1)},\dots,\mathbf x^{(i)}$ only. Stacking masked layers keeps this
  property (each layer only mixes past into present), so the whole network models
  $p(x_{i+1}\mid x_1,\dots,x_i)$, and the chain rule of probability gives
  $p(x_1,\dots,x_T)=\prod_i p(x_i\mid x_{<i})$ — the autoregressive factorization
  behind [[next-token-prediction]] and [[autoregressive-text-generation]].
- **Training efficiency.** One forward pass over $T$ tokens yields $T$ next-token
  predictions at once (positions 1…$T$), each seeing only its own prefix; the mask is
  what makes this parallel training legal (cf. sliding inputs/targets, book p. 35).
- **Lower-triangular structure.** The masked weight matrix $\tilde A$ is
  lower-triangular with a positive diagonal; the first row is always $[1, 0, \dots, 0]$
  (token 1 can only attend to itself). Exercise 6.3 explores using the first token for
  classification instead of the last (book p. 190).
- **Gradient flow.** At masked positions the weight is exactly 0, so no gradient flows
  from row $i$ to future values $\mathbf v^{(j)}, j>i$, through that path. Using $-\infty$
  (not a large negative number) makes this exact; in reduced precision, implementations
  sometimes use the dtype's minimum instead (background).
- **Dropout breaks the row-sum.** After dropout (rate 0.5 in the demo; 0.1–0.2 for GPT
  training), survivors are scaled by $1/(1-p)=2$ and rows no longer sum to 1 — e.g.
  row 3 becomes `[0.7599, 0.6194, 0.6206, 0, 0, 0]` (book pp. 78–80). The causal zeros
  stay zeros.

## Connections
- Builds on: [[self-attention]], [[softmax]], [[attention-scores]], [[attention-weights]], [[matrix]], [[broadcasting]], [[tensor]]
- Leads to: [[causal-attention]], [[multi-head-attention]], [[decoder-only-transformer]], [[gpt-architecture]], [[autoregressive-text-generation]]
- Related: [[next-token-prediction]], [[dropout]], [[padding-and-truncation]], [[top-k-sampling]], [[last-token-pooling]], [[classification-fine-tuning]], [[pytorch-nn-module]], [[probability-distribution]], [[context-length]]
- Used by: [[large-language-model]], [[gpt-2]], [[transformer-block]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.4, book p. 32 (masking padded tokens)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book p. 35 (Fig. 2.12: the LLM can't access words past the target)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5, book pp. 74–82 (Fig. 3.19–3.22; §3.5.1 two masking methods; information-leakage sidebar pp. 76–77; §3.5.2 dropout; Listing 3.3 and `register_buffer` p. 81)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.2, book p. 87 (mask truncation and broadcast in `MultiHeadAttention`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3, book p. 156 (−∞ masking reused in top-k sampling, Fig. 5.15)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 188–190 (last-token selection, Fig. 6.12; Exercise 6.3)
