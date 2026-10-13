---
title: Attention Weights
type: concept
tags: [attention, transformer, probability, llm]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Attention Weights

> **One-liner:** Attention weights are percentages that tell a token how much to "listen" to every other token — they are all positive and add up to 100%.

## 🟢 Intuition (Dummy)
Think of a student writing a summary of a sentence while focused on one word. She has
a fixed budget of attention: 100%. She might give 24% to "journey", 23% to "starts", and
just 11% to "one". Those shares are the **attention weights**.

They come from the raw match strengths ([[attention-scores]]) after a clean-up step
([[softmax]]) that makes them all positive and makes them add up to exactly 1. Then the
model mixes the words together using those shares, like mixing paint: 24% of this color,
23% of that one… The mixed result is the [[context-vector]] — a new, "context-aware"
version of the word.

Back in the RNN days, attention weights were the trick that let a translator look back
at any input word instead of one squeezed memory (Bahdanau attention, 2014; book p. 54).

**Careful with the word "weight".** The book stresses two very different meanings
(sidebar, book p. 66):
- **weight parameters** (like $W_q$) are learned during training and then stay fixed;
- **attention weights** are recomputed for every input — they are dynamic and context-specific.

## 🟡 Mechanics (Practitioner)

### From scores to weights
For query token 2, take its score row $\omega_2$ and apply softmax:

$$\alpha_{2j} = \operatorname{softmax}(\omega_2)_j = \frac{\exp(\omega_{2j})}{\sum_{t=1}^{T}\exp(\omega_{2t})}, \qquad \sum_{j=1}^{T}\alpha_{2j} = 1,\quad \alpha_{2j} > 0$$

Symbol by symbol:
- $\alpha$ (Greek "alpha") = the book's letter for attention **weights**.
- $\alpha_{2j}$ = how much query token 2 attends to token $j$.
- $\omega_{2j}$ = the raw [[attention-scores|attention score]] between them.
- $\exp(\cdot)$ = $e^{(\cdot)}$, the exponential function (always positive).
- $\sum_{t=1}^{T}$ = add over all $T$ tokens; it makes the bottom of the fraction the same for every $j$.
- The conditions on the right say: the weights in one row add to 1 and each is positive — a [[probability-distribution]].

### The book's worked numbers ("Your journey starts with one step")
Scores for "journey": `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]` (book p. 58).

The book first tries **plain division by the sum** (book p. 59):
$$\alpha^{\text{tmp}}_{2j} = \frac{\omega_{2j}}{\sum_t \omega_{2t}} \;\Rightarrow\; [0.1455,\ 0.2278,\ 0.2249,\ 0.1285,\ 0.1077,\ 0.1656]$$
The goal of normalizing is weights that sum to 1 — a convention the book calls useful
for **interpretation** and for keeping **training stable** (book p. 59). Plain division sums
to 1, but softmax is preferred: it handles extreme values better and has nicer gradient
properties for training (book pp. 59–60). It also makes every weight positive, so the weights
read as probabilities or relative importance (book p. 60). Softmax gives (book p. 60):
```
tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])   # sum = 1
```
Check one: the six values $e^{\omega_{2j}}$ are about 2.597, 4.459, 4.373, 2.324, 2.028,
2.964, summing to 18.745; so $\alpha_{22} = 4.459/18.745 \approx 0.2379$. ✓

### Using the weights: the context vector
$$z^{(2)} = \sum_{j=1}^{T} \alpha_{2j}\, x^{(j)}$$
- $z^{(2)}$ = the [[context-vector]] for token 2; each input vector $x^{(j)}$ is scaled by its weight and they are all added.

Book result: $z^{(2)}$ = `[0.4419, 0.6515, 0.5683]` (book p. 60).

```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
```

### The whole weight matrix (§3.3.2)
$$A = \operatorname{softmax}_{\text{row}}(\Omega), \qquad Z = A X$$
- $A$ = the $T\times T$ weight matrix; "row" softmax means each **row** sums to 1.
- `dim=-1` in PyTorch = normalize along the last dimension (across columns), so each row sums to 1 (book p. 63).

```python
attn_weights = torch.softmax(attn_scores, dim=-1)   # (6,6)
all_context_vecs = attn_weights @ inputs            # (6,3)
```
The full weight matrix (book p. 63; Fig. 3.11 on p. 61 shows it truncated to two digits):
```
tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
        [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
        [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
        [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
        [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
        [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])
```
`attn_weights.sum(dim=-1)` returns six 1.0000 values. Row 2 is the vector above, and row 2
of `all_context_vecs` equals `[0.4419, 0.6515, 0.5683]` again (book pp. 63–64).

### With trainable weights (§3.4)
$$\alpha_{2i} = \operatorname{softmax}\!\left(\frac{\omega_2}{\sqrt{d_k}}\right)_i, \qquad z^{(2)} = \sum_i \alpha_{2i}\, v^{(i)}$$
- $\sqrt{d_k}$ = square root of the key dimension (`keys.shape[-1]`, here 2); dividing keeps softmax from saturating (book pp. 68–69).
- $v^{(i)}$ = the **value** vector of token $i$ (see [[query-key-value]]) — now we mix values, not raw inputs.

Book numbers (book pp. 68–70):
```
attn_weights_2 = tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
context_vec_2  = tensor([0.3061, 0.8210])
```
```python
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
context_vec_2 = attn_weights_2 @ values
```

### Causal masking and dropout (§3.5)
For GPT-style models, a token must not look at the future: weights above the diagonal are
forced to 0 and each row is renormalized (book pp. 74–78):
$$\alpha_{ij} = 0 \ \text{ for } j > i, \qquad \sum_{j=1}^{i}\alpha_{ij} = 1.$$
Book example rows after masking: "Your" → `[1.0]`, "journey" → `[0.5517, 0.4483]`,
"starts" → `[0.3800, 0.3097, 0.3103]` (book p. 76). See [[causal-attention-mask]].

During training, [[dropout]] then zeroes random attention weights and scales the survivors
by $1/(1-p)$; with $p = 0.5$, 0.3800 becomes 0.7599 (book pp. 79–80). After dropout a row no
longer sums to 1 — that is expected.

## 🔴 Theory (Expert)
- **Row-stochastic matrix.** $A$ has non-negative entries and rows summing to 1, so each
  context vector $z^{(i)} = \sum_j A_{ij} v^{(j)}$ is a [[convex-combination]] of the value
  vectors: it lies in their convex hull. Attention can only *average*, never extrapolate
  beyond the values; new directions come from $W_v$, the output projection and the
  [[feed-forward-network]]. ("Row-stochastic" is our term, not the book's.)
- **Soft dictionary lookup.** Hard retrieval returns the value of the single best key
  (an [[argmax]], a one-hot weight row). Softmax replaces it with a smooth, differentiable
  weighting, so gradients flow to every key and value (book p. 70 sidebar on
  query/key/value; differentiability remark is ours).
- **Masking = conditioning.** Renormalizing a softmax row over the subset $j\le i$ equals
  computing softmax only on that subset, because the full denominator cancels:
  $\frac{e^{z_{ij}}/S_i}{\sum_{k\le i} e^{z_{ik}}/S_i} = \frac{e^{z_{ij}}}{\sum_{k\le i}e^{z_{ik}}}$.
  So future tokens leak no information (book p. 77; algebra ours).
- **Entropy view.** A row with low entropy (one weight near 1) is "sharp" attention; a
  near-uniform row (e.g. the simplified-attention rows above, all between ≈ 0.10 and 0.24,
  book p. 63) is "diffuse". (Fig. 3.18's matrix, book p. 72, is illustrative only — the book
  says its values "do not depict the real attention weights".)
  The $1/\sqrt{d_k}$ scale acts like a [[temperature-scaling|temperature]] that controls this.
- **Gradient.** $\partial \alpha_i/\partial \omega_j = \alpha_i(\delta_{ij} - \alpha_j)$. When one
  weight approaches 1 all these entries approach 0 — the vanishing-gradient problem the
  book's scaling sidebar describes (book p. 69; Jacobian is our addition).
- **Inverted dropout** keeps the expectation: $\mathbb E[\tilde a_{ij}] = a_{ij}$ when
  $\tilde a_{ij} = m_{ij}a_{ij}/(1-p)$, $m_{ij}\sim\text{Bernoulli}(1-p)$ (book pp. 79–80; the
  expectation identity is ours).

### From llm-ch06-fine-tuning-for-classification
Fig. 6.12 (§6.5, book p. 190) shows causal attention weights for "Do you have time". Row by row: Do = [1.0]; you = [0.55, 0.45]; have = [0.38, 0.30, 0.32]; time = [0.27, 0.24, 0.24, 0.25]. Each row sums to 1, so these are normalized attention weights (the caption says "attention scores", which is the Ch. 3 name for the values before [[softmax]]). Only the last row has a weight for every token. This is the reason Ch. 6 uses the last token's output for classification ([[last-token-pooling]]).

## Connections
- Builds on: [[attention-scores]], [[softmax]], [[probability-distribution]], [[inner-product]]
- Leads to: [[context-vector]], [[causal-attention-mask]], [[multi-head-attention]]
- Used by: [[self-attention]], [[attention-mechanism]], [[bahdanau-attention]], [[scaled-dot-product-attention]], [[transformer-architecture]]
- Related: [[query-key-value]], [[convex-combination]], [[dropout]], [[expected-value-and-mean]]
- People: [[dzmitry-bahdanau]], [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book p. 54 (attention weights decide token importance; Fig. 3.5).
- Same, §3.3.1–3.3.2, book pp. 56–64 (normalization, softmax weights, context vectors, Figs. 3.7–3.12).
- Same, §3.4.1, book pp. 65–70 (sidebar "weight parameters vs. attention weights" p. 66; scaled weights pp. 68–69).
- Same, §3.5.1–3.5.2, book pp. 74–80 (causal masking, renormalization, dropout on weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book p. 190 (Fig. 6.12, causal attention weights; last token sees all tokens) — see [[llm-ch06-fine-tuning-for-classification]].
