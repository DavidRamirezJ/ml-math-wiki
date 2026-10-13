---
title: Softmax Function
type: concept
tags: [probability, activation-function, deep-learning, llm, numerical-methods]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Softmax Function

> **One-liner:** Softmax turns any list of numbers into a list of positive percentages that add up to 100%, giving the biggest numbers the biggest shares.

## 🟢 Intuition (Dummy)
Picture a talent show where the judges give raw scores like 6.75, 6.28, 4.51, and even
−1.90. You want to turn those into "chance of winning" percentages. Softmax does it in two
moves:

1. **Make everything positive** by raising $e$ (≈ 2.718) to the power of each score.
   Negative scores become small positive numbers; big scores become *much* bigger numbers.
2. **Divide by the total**, so everything adds up to 1.

Because of step 1, softmax is a bit dramatic: the leader gets a bigger slice than plain
"divide by the sum" would give. It is a *soft* version of "pick the max" — hence the name.

In an LLM, softmax shows up in two big places:
- inside **attention**, to turn match strengths into [[attention-weights]] (Ch. 3);
- at the **output**, to turn the model's 50,257 raw scores ([[logits]]) into the
  probability of each possible next token (Chs. 4–7).

## 🟡 Mechanics (Practitioner)

### The formula
$$\operatorname{softmax}(\mathbf z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \qquad i = 1,\dots,K$$

Symbol by symbol:
- $\mathbf z = (z_1,\dots,z_K)$ = the input list (scores or [[logits]]); bold means "a vector".
- $\operatorname{softmax}(\mathbf z)_i$ = the $i$-th output number (the share for item $i$).
- $e^{z_i}$ = $e$ raised to the power $z_i$; always $> 0$.
- $\sum_{j=1}^{K} e^{z_j}$ = the total of all exponentials; the same denominator for every $i$.
- $K$ = number of items: 6 tokens in the attention example, 2 classes in spam
  classification, 50,257 vocabulary tokens in GPT-2.

Outputs are all positive and sum to 1, so they form a [[probability-distribution]].

### Worked example 1 — attention (book §3.3.1, pp. 59–60)
Scores of "journey" against the six words of "Your journey starts with one step":
`[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]`.

| | Your | journey | starts | with | one | step |
|---|---|---|---|---|---|---|
| divide by sum | 0.1455 | 0.2278 | 0.2249 | 0.1285 | 0.1077 | 0.1656 |
| softmax | 0.1385 | 0.2379 | 0.2333 | 0.1240 | 0.1082 | 0.1581 |

Softmax pushes more weight to the top scorers ("journey", "starts") and less to the
weakest ("with"). The book's own implementation:

```python
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)   # preferred in practice
```
The naive version can overflow or underflow on large or small inputs, so use
`torch.softmax` (book p. 60; see [[numerical-stability]]).

**The `dim` argument.** For a 2-D score matrix, `torch.softmax(attn_scores, dim=-1)`
normalizes along the last dimension, so **each row** sums to 1 (book p. 63).

### Worked example 2 — scaled softmax in attention (book §3.4.1, pp. 68–69)
$$\alpha = \operatorname{softmax}\!\left(\frac{\boldsymbol\omega}{\sqrt{d_k}}\right)$$
- $\boldsymbol\omega$ = attention scores; $d_k$ = key dimension (here 2); $\sqrt{d_k} = d_k^{0.5}$.

`softmax([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440] / √2)`
= `[0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820]`. Without the $\sqrt 2$ the result would
be sharper (≈ `[0.140, 0.251, 0.241, 0.116, 0.069, 0.184]` — our computation). For big
$d_k$ (GPT-like models use embeddings above 1,000) unscaled scores make softmax act "like a
step function" and gradients nearly vanish (book p. 69).

### Masking with $-\infty$ (book §3.5.1 p. 77; §5.3.2 p. 156)
Because $e^{-\infty} = 0$, any score set to $-\infty$ gets probability exactly 0 and the
rest still sum to 1. The book uses this for the [[causal-attention-mask]]:
```python
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
```
and again for [[top-k-sampling]]: logits `[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]`
with $k=3$ become `[4.51, -inf, -inf, 6.75, -inf, -inf, -inf, 6.28, -inf]`, and softmax gives
`[0.06, 0, 0, 0.57, 0, 0, 0, 0.36, 0]` (book p. 156).

### Worked example 3 — next-token probabilities (Chs. 4–5)
The GPT model outputs a [[logits]] tensor of shape (batch, tokens, 50,257). Take the last
row and softmax it (book pp. 123–125):
```python
logits = logits[:, -1, :]
probas = torch.softmax(logits, dim=-1)
idx_next = torch.argmax(probas, dim=-1, keepdim=True)
```
An untrained model's probabilities hover near uniform: about $1/7 \approx 0.14$ for a
7-token toy vocabulary and about $1/50{,}257 \approx 0.00002$ for GPT-2 (book p. 135).

**Softmax is monotonic.** It keeps the order of its inputs, so the biggest logit is always
the biggest probability. That means `argmax(logits)` = `argmax(softmax(logits))` — for
[[greedy-decoding]] softmax is "redundant" (book p. 125; same point for classification,
book p. 192, and appendix A, book p. 277). See [[monotonic-function]], [[argmax]].

### Worked example 4 — softmax with temperature (book §5.3.1, pp. 153–155)
$$p_i(T) = \frac{e^{z_i/T}}{\sum_{j=1}^{V} e^{z_j/T}}, \qquad T > 0$$
- $T$ = temperature; $V$ = vocabulary size (9 in the toy).
- $T = 1$: plain softmax. $T > 1$: flatter, more uniform. $T < 1$: sharper, more "peaky".

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```
With the toy logits above, multinomial sampling 1,000 times at $T=1$ gives
582 × "forward", 343 × "toward", 73 × "closer", 2 × "inches" (book pp. 153–154). At
$T = 0.1$ "forward" is picked almost 100% of the time; at $T = 5$ the nonsense "every effort
moves you pizza" appears about 4% of the time (book p. 155). See [[temperature-scaling]],
[[multinomial-sampling]].

### Worked example 5 — classification (Ch. 6 and appendix A)
- Spam classifier, $K=2$: logits `[3.5983, -3.9902]` → `[0.99, 0.01]` → label 0 "not spam";
  `[-3.9846, 5.2940]` → `[0.01, 0.99]` → label 1 "spam" (Fig. 6.14, book p. 191). The model
  is not trained yet, so these labels are wrong; the book shows only the mechanics.
- Appendix A toy net (book pp. 276–277): logits row `[2.8569, -4.1618]` →
  `torch.softmax(outputs, dim=1)` → `[0.9991, 0.0009]`. Check: with two classes softmax is a
  [[sigmoid-function|sigmoid]] of the difference, $1/(1+e^{-(2.8569+4.1618)}) = 1/(1+e^{-7.0187})\approx 0.9991$.
- A 3-class net returns `[-0.1262, 0.1080, -0.1792]`; softmax → `[0.3113, 0.3934, 0.2952]` (book p. 269).

### Softmax inside the loss
Models return raw logits, *not* probabilities. PyTorch's loss functions combine softmax
with the negative log-likelihood in one step for efficiency and numerical stability (book
p. 269, p. 276). So **never** apply softmax before `torch.nn.functional.cross_entropy`:
$$\ell(\mathbf z, y) = -\log \operatorname{softmax}(\mathbf z)_y = -z_y + \log\sum_{j} e^{z_j}$$
- $y$ = index of the correct class/token; $-\log$ of its probability is the
  [[cross-entropy-loss]]. Example: logits `[[-1.0, 1.0], [-0.5, 1.5]]` with targets `[0, 1]`
  give loss `1.1269` (book p. 221). In pretraining, the average log-probability of the
  targets in the book's first batch is −10.7940, so the loss is 10.7940 (book p. 137).

## 🔴 Theory (Expert)
- **Shift invariance.** $\operatorname{softmax}(\mathbf z + c\mathbf 1) = \operatorname{softmax}(\mathbf z)$ for any
  constant $c$, because $e^{c}$ cancels top and bottom. This gives the stable
  implementation $\operatorname{softmax}(\mathbf z) = \operatorname{softmax}(\mathbf z - \max_j z_j)$
  (see [[numerical-stability]]). It also means only score *differences* matter.
- **Not scale invariant.** $\operatorname{softmax}(\mathbf z/T)$ tends to a one-hot vector on
  $\arg\max_i z_i$ as $T\to0^+$ and to the uniform vector $\tfrac1K\mathbf 1$ as $T\to\infty$.
  The $1/\sqrt{d_k}$ factor in attention is exactly a temperature (limits are standard math,
  consistent with book pp. 154–155).
- **Gradient (Jacobian).**
  $$\frac{\partial\, \operatorname{softmax}(\mathbf z)_i}{\partial z_j} = s_i(\delta_{ij} - s_j), \qquad J = \operatorname{diag}(\mathbf s) - \mathbf s\mathbf s^\top,$$
  with $\mathbf s = \operatorname{softmax}(\mathbf z)$ and $\delta_{ij}$ = 1 if $i=j$ else 0.
  If one $s_i \to 1$, then $J \to 0$: saturation kills gradients — the book's "step
  function" warning (book p. 69). $J$ is symmetric PSD with $J\mathbf 1 = \mathbf 0$ (the
  shift-invariance direction).
- **Cross-entropy gradient.** $\nabla_{\mathbf z}\,[-\log s_y] = \mathbf s - \mathbf e_y$: predicted
  probabilities minus the one-hot target. This clean form is why softmax + cross-entropy
  is the default pairing (see [[maximum-likelihood-estimation]]).
- **Log-sum-exp and smooth max.** $\log\sum_j e^{z_j}$ is a smooth, convex upper bound on
  $\max_j z_j$: $\max_j z_j \le \operatorname{LSE}(\mathbf z) \le \max_j z_j + \log K$. Softmax is its
  gradient, $\nabla \operatorname{LSE}(\mathbf z) = \operatorname{softmax}(\mathbf z)$ — so "softmax" is really
  "soft argmax".
- **Exponential family.** Softmax maps natural parameters to the mean parameters of the
  categorical distribution; $\operatorname{LSE}$ is its log-partition function
  ([[exponential-family]]).
- **Two classes.** $\operatorname{softmax}(z_1,z_2)_1 = \sigma(z_1 - z_2)$, linking it to the
  [[sigmoid-function]] and [[logistic-regression]].
- **Monotonicity, precisely.** Each output $s_i$ is strictly increasing in $z_i$ and the
  outputs share one denominator, so $z_i > z_j \iff s_i > s_j$; argmax is unchanged.
- **Masking and renormalization.** Softmax restricted to a subset $S$ equals full softmax
  conditioned on $S$: $s_i / \sum_{k\in S} s_k$. This is why $-\infty$ masking (causal
  attention, top-k) produces a proper distribution with no leakage from masked entries
  (book p. 77).

## Connections
- Builds on: [[vector]], [[probability-distribution]], [[logits]], [[exponential-family]]
- Leads to: [[attention-weights]], [[cross-entropy-loss]], [[temperature-scaling]], [[top-k-sampling]], [[multinomial-sampling]]
- Used by: [[self-attention]], [[scaled-dot-product-attention]], [[causal-attention-mask]], [[next-token-prediction]], [[greedy-decoding]], [[classification-fine-tuning]], [[logistic-regression]]
- Related: [[argmax]], [[monotonic-function]], [[numerical-stability]], [[sigmoid-function]], [[log-probability]], [[maximum-likelihood-estimation]], [[activation-function]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3.1, book pp. 59–60 (softmax vs. sum-normalization, `softmax_naive`, overflow/underflow); §3.3.2, book p. 63 (`dim=-1`).
- Same, §3.4.1, book pp. 68–69 (scaled softmax, step-function sidebar); §3.5.1, book pp. 75–78 ($-\infty$ masking, renormalization, no-leakage sidebar); §3.6, book pp. 85–90 (softmax per head).
- Same, §4.7, book pp. 123–125 (logits → probabilities → argmax; monotonicity).
- Same, §5.1, book pp. 133–137 (probabilities, near-uniform start, log-probabilities and loss); §5.3, book pp. 152–157 (temperature scaling, multinomial sampling, top-k masking).
- Same, §6.6, book pp. 190–193 (spam softmax, softmax optional for argmax, cross-entropy on logits).
- Same, §7.3, book pp. 221–222 (cross-entropy demo, `ignore_index=-100`).
- Same, §A.5, book p. 269 (models return logits; explicit softmax); §A.7, book pp. 276–277 (`F.cross_entropy` applies softmax internally; softmax/argmax on toy outputs).
