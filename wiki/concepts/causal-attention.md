---
title: Causal (Masked) Attention
type: concept
tags: [attention, transformer, llm, masking, deep-learning]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Causal (Masked) Attention

> **One-liner:** Causal attention is self-attention with a blindfold on the future — each token may look only at itself and the tokens before it.

## 🟢 Intuition (Dummy)
Think of a quiz where you guess the next word of a sentence. If the answer sheet is
lying face-up on the desk, the quiz is pointless. A GPT model learns by guessing the
next word ([[next-token-prediction]]), so during training we must make sure it can't
peek at the words that come later.

Plain [[self-attention]] lets every word look at *every* other word, including future
ones. **Causal attention** covers up the future: the word "journey" in
"Your journey starts with one step" may look at "Your" and at itself, but not at
"starts", "with", "one", or "step". The word "step", being last, may look at everything.

If you draw the "who-looks-at-whom" table, only the lower-left triangle survives.
That triangle is the whole trick.

## 🟡 Mechanics (Practitioner)
**Definition (book p. 74).** Causal attention, also called **masked attention**, is a
specialized form of self-attention that restricts the model to the **previous and
current inputs** when computing attention scores; standard self-attention sees the
entire sequence at once. For GPT-like LLMs, we mask out the attention weights
**above the diagonal** and renormalize the rest so **each row sums to 1** (book pp. 74–75).

**The rule in symbols:**
$$\alpha_{ij}=0\ \text{ for } j>i,\qquad \sum_{j=1}^{i}\alpha_{ij}=1$$
- $\alpha_{ij}$ = the [[attention-weights|attention weight]] that token $i$ (the query, a row) gives to token $j$ (the key, a column).
- $j>i$ = "token $j$ comes after token $i$" = the future → weight forced to 0.
- $\sum_{j=1}^{i}$ = add up the weights over the allowed tokens $1,\dots,i$; they must total 1.

**Book's worked example (Fig. 3.19, book p. 74).** Six tokens, causal weights after
masking:

| | Your | journey | starts | with | one | step |
|---|---|---|---|---|---|---|
| Your | 1.0 | | | | | |
| journey | 0.55 | 0.44 | | | | |
| starts | 0.38 | 0.30 | 0.31 | | | |
| with | 0.27 | 0.24 | 0.24 | 0.23 | | |
| one | 0.21 | 0.19 | 0.19 | 0.18 | 0.19 | |
| step | 0.19 | 0.16 | 0.16 | 0.15 | 0.16 | 0.15 |

**Approach A — softmax, zero-mask, renormalize (Fig. 3.20, book pp. 75–76).**
```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
mask_simple = torch.tril(torch.ones(context_length, context_length))  # 1s on/below diagonal
masked_simple = attn_weights * mask_simple
masked_simple_norm = masked_simple / masked_simple.sum(dim=-1, keepdim=True)
```
Formula: $\tilde\alpha_{ij}=\dfrac{\alpha_{ij}M_{ij}}{\sum_k\alpha_{ik}M_{ik}}$, with
$M_{ij}=1$ if $j\le i$ else 0.
- $M$ = the 0/1 mask (`torch.tril` keeps the lower triangle *including* the diagonal).
- $\alpha_{ij}M_{ij}$ = element-wise (Hadamard) product — zeros out the future.
- Denominator = the row's surviving total; dividing makes the row sum to 1 again.

Row 2 of the book's softmax output is $(0.2041, 0.1659, \dots)$; after masking and
renormalizing: $0.2041/(0.2041+0.1659)=0.5517$ and $0.4483$ — matching the printed
`[0.5517, 0.4483, 0, 0, 0, 0]` (book p. 76).

**Is that "leaky"? (sidebar, book pp. 76–77).** The future scores were inside the
first softmax's denominator. But renormalizing is *identical* to running softmax
only over the unmasked entries, so the future has no effect:
$$\frac{e^{z_{ij}}/S_i}{\sum_{k\le i}e^{z_{ik}}/S_i}=\frac{e^{z_{ij}}}{\sum_{k\le i}e^{z_{ik}}},\qquad S_i=\sum_{k=1}^{n}e^{z_{ik}}$$
- $z_{ij}$ = raw score of token $i$ for token $j$; $S_i$ = the full (leaky-looking) softmax denominator.
- $S_i$ appears top and bottom and **cancels** — no information leakage.

**Approach B — mask with $-\infty$, then one softmax (Fig. 3.21, book pp. 77–78).**
This is the efficient way and what the final code uses:
```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)  # 1s strictly above diagonal
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
```
It works because $e^{-\infty}=0$, so masked entries get zero probability. In one line:
$$A=\operatorname{softmax}\!\left(\frac{QK^\top+M_{-\infty}}{\sqrt{d_k}}\right)$$
- $Q$, $K$ = query and key matrices ([[query-key-value]]); $QK^\top$ = all pairwise [[attention-scores]].
- $M_{-\infty}$ = a matrix with $-\infty$ above the diagonal and 0 elsewhere.
- $\sqrt{d_k}$ = square root of the key dimension ([[scaled-dot-product-attention]]); here $d_k=2$.
- softmax is taken row by row ([[softmax]]).

The output is exactly the same as Approach A's (book p. 78). The masked raw scores
look like `[[0.2899, -inf, ...], [0.4656, 0.1723, -inf, ...], ...]` (book p. 77).

**The compact class (Listing 3.3, book p. 81)** — causal mask + [[dropout]]:
```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values
```
- `register_buffer` makes the mask move with the model to CPU/GPU automatically, avoiding device-mismatch errors (book p. 81).
- `[:num_tokens, :num_tokens]` slices the mask for inputs shorter than `context_length`.
- `masked_fill_` (trailing underscore) works in place, saving memory (book p. 81).
- Demo: a batch of shape `(2, 6, 3)` → output `(2, 6, 2)` with `d_out=2`, dropout `0.0` (book pp. 80–82).

**In the full GPT and beyond.** Every [[transformer-block]] uses *masked*
[[multi-head-attention]] (book p. 94, Fig. 4.2). The same $-\infty$ trick reappears in
[[top-k-sampling]] to zero out non-selected logits (book p. 156). In classification
fine-tuning, the causal mask is why the model reads the **last** token: it is the only
one that has attended to all earlier tokens, so it accumulates the most information
(book p. 190). Fig. 6.12 shows this on "Do you have time": rows `Do 1.0`,
`you 0.55 0.45`, `have 0.38 0.30 0.32`, `time 0.27 0.24 0.24 0.25` — only "time"
has scores for every token (book p. 190). Exercise 6.3 confirms it:
using the *first* token instead drops test accuracy from **95.67%** to **75.00%**
(book p. 307).

## 🔴 Theory (Expert)
- **Autoregressive factorisation.** Causal masking makes the network compute
  $p(x_{t+1}\mid x_{\le t})$ at *every* position in a single forward pass, which
  realises the chain-rule factorisation
  $p(x_1,\dots,x_T)=\prod_t p(x_t\mid x_{<t})$. Training on all $T$ positions in
  parallel ("teacher forcing") is what makes transformer pretraining efficient
  compared with RNNs ([[autoregressive-text-generation]], [[maximum-likelihood-estimation]]).
- **Lower-triangular structure.** $A$ is lower-triangular and row-stochastic. Its
  diagonal is strictly positive ($e^{z_{ii}}>0$), so $A$ is invertible; row 1 is always
  $(1,0,\dots,0)$ — token 1 attends only to itself.
- **Consistency for generation.** Because token $i$'s output depends only on tokens
  $\le i$, previously computed keys/values never change when a new token is appended.
  That is what makes **KV caching** valid at inference time.
- **Numerical note.** Using $-\infty$ is safe as long as each row has at least one
  finite entry (the diagonal guarantees it). With padding masks, a fully masked row
  gives $0/0=$ NaN; implementations often use a large negative finite number instead
  ([[numerical-stability]]).
- **Removing the mask.** Appendix B says recent papers show that classification
  performance can be further improved by **removing the causal mask during
  classification fine-tuning**, alongside other modifications — citing Li et al. 2023,
  "Label Supervised LLaMA Finetuning", and BehnamGhader et al. 2024, "LLM2Vec: Large
  Language Models Are Secretly Powerful Text Encoders" (book p. 296).
- **Fused kernels.** Appendix B notes that PyTorch's `scaled_dot_product_attention`
  implements self-attention and causal attention with FlashAttention support (beta at
  the time of writing) (book p. 292; [[flashattention]]). Contrast with [[bert]], which is bidirectional by design
  ([[masked-language-modeling]] masks *inputs*, not attention).

## Connections
- Builds on: [[self-attention]], [[scaled-dot-product-attention]], [[softmax]], [[attention-weights]], [[query-key-value]], [[matrix]]
- Leads to: [[multi-head-attention]], [[transformer-block]], [[gpt-architecture]], [[last-token-pooling]]
- Used by: [[decoder-only-transformer]], [[next-token-prediction]], [[autoregressive-text-generation]], [[classification-fine-tuning]]
- Related: [[causal-attention-mask]], [[dropout]], [[top-k-sampling]], [[numerical-stability]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1 intro, book p. 51 (Fig. 3.2); §3.5, book pp. 74–82 (Figs. 3.19–3.23, Listing 3.3); §3.6 summary, book p. 91.
- §4.1, book p. 94 (Fig. 4.2); §5.3, book p. 156 (−∞ reuse, Fig. 5.15); §6.5–6.6, book p. 190 (Fig. 6.12, last-token reasoning); appendix B Ch. 3 (PyTorch `scaled_dot_product_attention`), book p. 292; appendix B Ch. 6 (Li et al. 2023; BehnamGhader et al. 2024), book p. 296; appendix C Exercise 6.3, book p. 307.
