---
title: Logits
type: concept
tags: [llm, classification, softmax, neural-network, probability]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Logits

> **One-liner:** Logits are the raw scores a model spits out — one number per possible answer — before they are turned into probabilities; the bigger the score, the more the model "likes" that answer.

## 🟢 Intuition (Dummy)
Imagine a talent show where judges shout scores like "+6.75!", "−1.9!", "+4.5!".
The scores aren't percentages — they can be negative, they don't add up to
anything in particular — but you can already tell who's winning: the highest
score. To turn the scores into "chance of winning" you run them through a
converter called [[softmax]], which makes them all positive and makes them add
up to 100%.

In an LLM, the "contestants" are all 50,257 tokens in the [[vocabulary]], and the
show runs once for every position in the text. The scores are the **logits**. The
last layer of the model (the output head) produces them; everything after that —
picking the next word, computing the loss — starts from these numbers.

## 🟡 Mechanics (Practitioner)

### Where they come from
In `GPTModel` the last line of `forward` is `logits = self.out_head(x)` (§4.6, p. 119),
where `out_head = nn.Linear(768, 50257, bias=False)`. For one position:

$$
\mathbf{z}_t = W_{\text{out}}\,\mathbf{h}_t, \qquad W_{\text{out}} \in \mathbb{R}^{V\times d}
$$

- $\mathbf{h}_t$ = the final hidden vector for token position $t$ (after the final [[layer-normalization]]), length $d = 768$.
- $W_{\text{out}}$ = the output-head weight [[matrix]], $V = 50{,}257$ rows by $d = 768$ columns.
- $\mathbf{z}_t$ = the logit vector for position $t$: one score $z_{t,i}$ for every vocabulary token $i$.

For a whole batch the shape is **[batch, num_tokens, vocab_size]**. The book's first
run of the model on two 4-token texts prints `torch.Size([2, 4, 50257])`, starting
with `[[-1.2034, 0.3201, -0.7130, ...` (§4.1, p. 99).

### From logits to probabilities: softmax
$$
p_i = \operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
$$

- $z_i$ = the logit of token $i$; $e^{z_i}$ = $e$ (≈2.718) raised to that score — always positive.
- $\sum_{j=1}^{V} e^{z_j}$ = the sum of those positive numbers over all $V$ tokens.
- $p_i$ = the probability that the next token is $i$. All $p_i$ are between 0 and 1 and add to 1.

**Book example (§5.3, p. 153)**, a 9-token toy vocabulary
(closer, every, effort, forward, inches, moves, pizza, toward, you):
```python
next_token_logits = torch.tensor([4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79])
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()   # 3 -> "forward"
```
Softmax over all nine logits gives about 0.572 for "forward", 0.358 for "toward",
0.061 for "closer" and ≤ 0.004 for the rest (our computation; the book prints only the
argmax), so greedy decoding picks "forward" (p. 153). The book's top-k figure (Figure 5.15,
p. 156) shows 0.57 / 0.36 / 0.06 — the softmax *after* masking all but the top 3, which
agrees to two decimals because the other six tokens carry almost no mass.

### You don't need softmax to pick the winner
Softmax is **monotonic** — it keeps the order of its inputs (§4.7, p. 125). So

$$
\arg\max_i\, p_i = \arg\max_i\, z_i
$$

- $\arg\max_i$ = "the index $i$ where the value is biggest".

The book uses `torch.argmax` on logits directly in [[greedy-decoding]] and in the
spam classifier (§6.6, p. 192) and notes the softmax step is "redundant" for that
purpose; it is kept only to show the full pipeline.

### Logits go straight into the loss
PyTorch's `cross_entropy` takes **raw logits** and applies (log-)softmax inside
"for efficiency and numerical stability reasons" (§A.7, p. 276). That is why
models return logits, not probabilities (§A.5, p. 269). In pretraining, the
logits are flattened from `[2, 3, 50257]` to `[6, 50257]` first (§5.1, p. 139):
```python
logits_flat  = logits.flatten(0, 1)        # [6, 50257]
targets_flat = targets.flatten()           # [6]
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)   # tensor(10.7940)
```
See [[cross-entropy-loss]].

### Changing the head changes the logits
- **Next-token prediction:** 50,257 logits per position.
- **Spam classification (Ch. 6):** `out_head` is swapped for `nn.Linear(768, 2)` →
  2 logits per position, and only the **last** token's logits are used (§6.5,
  pp. 184–186). Example: the last token's logits are `tensor([[-3.5983, 3.9902]])`;
  `torch.argmax` returns **1**, meaning "spam" — with or without softmax first
  (§6.5, p. 188; §6.6, p. 192). Figure 6.14 (p. 191) shows the same conversion for two
  messages, e.g. logits `[-3.9846, 5.2940]` → probabilities `[0.01, 0.99]`. See [[classification-head]], [[last-token-pooling]].
- **Tabular toy net (appendix A):** a 3-layer MLP whose last `Linear` outputs
  logits, with no activation at the end (§A.5).

### Temperature: logits are the knob for sampling
Dividing logits by a temperature $T > 0$ before softmax (§5.3, p. 154):

$$
p_i(T) = \frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}
$$

- $T = 1$: unchanged. $T < 1$: sharper, more confident. $T > 1$: flatter, more random.

[[top-k-sampling]] also works on logits: keep the $k$ largest, set the rest to
$-\infty$ so their probability becomes exactly 0 (p. 156). See
[[temperature-scaling]], [[decoding-strategies]].

## 🔴 Theory (Expert)
- **Why the name.** For two classes, the *logit* function is
  $\operatorname{logit}(p) = \log\frac{p}{1-p}$, the inverse of the
  [[sigmoid-function]]. In the multi-class case, logit *differences* are
  log-odds: $z_i - z_j = \log(p_i / p_j)$. With $K=2$, softmax reduces to
  $p_1 = \sigma(z_1 - z_0)$ (standard result, not stated in the book; relevant to the
  2-logit spam head of §6.5).
- **Logits are only defined up to a constant.** Adding the same $c$ to every
  logit leaves softmax unchanged:
  $\frac{e^{z_i+c}}{\sum_j e^{z_j+c}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$. This is
  what makes the log-sum-exp trick work: compute
  $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ with $m = \max_j z_j$, so no
  exponential overflows ([[numerical-stability]]).
- **Log-probabilities are shifted logits.**
  $\log p_i = z_i - \log\sum_j e^{z_j}$. So cross-entropy for target $y$ is
  $\mathcal{L} = -z_y + \log\sum_j e^{z_j}$, and its gradient is beautifully
  simple: $\frac{\partial\mathcal{L}}{\partial z_i} = p_i - \mathbb{1}[i=y]$ — "predicted
  minus true". This is why fused softmax + cross-entropy is both stable and cheap.
- **Untrained baseline.** If all logits are nearly equal, $p_i \approx 1/V$ and
  the loss is $\approx \ln V = \ln 50{,}257 \approx 10.82$. The book's untrained
  model gives 10.7940 (§5.1, pp. 137–139) — consistent with near-uniform logits.
- **Logits as inner products.** With [[weight-tying]] (GPT-2), $W_{\text{out}} = E_{\text{tok}}$,
  so $z_{t,i} = \langle \mathbf{h}_t, \mathbf{e}_i\rangle$: the score of token $i$
  is the [[inner-product]] of the hidden state with that token's embedding row.
  The model "predicts" the token whose embedding points most in the direction of $\mathbf{h}_t$.
- **Temperature limits.** $T \to 0^+$ gives a one-hot vector at the argmax
  (greedy); $T \to \infty$ gives the uniform distribution $1/V$.
- **Terminology caveat.** The book sometimes calls logits "vectors representing
  the probability distribution" (§5.1, p. 131). Strictly, they are *unnormalized*
  scores; only after softmax are they a [[probability-distribution]]. The book is
  precise elsewhere: "logits = unscaled model outputs before softmax" (p. 139).

## Connections
- Builds on: [[linear-layer]], [[linear-mapping]], [[matrix-multiplication]], [[vector]], [[output-projection]]
- Leads to: [[softmax]], [[cross-entropy-loss]], [[log-probability]], [[argmax]], [[greedy-decoding]], [[temperature-scaling]], [[top-k-sampling]]
- Used by: [[gpt-architecture]], [[gpt-2]], [[autoregressive-text-generation]], [[classification-head]], [[classification-fine-tuning]], [[llm-training-loop]]
- Related: [[sigmoid-function]], [[logistic-regression]], [[probability-distribution]], [[weight-tying]], [[monotonic-function]], [[numerical-stability]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 97–99 (first logits, shape [2, 4, 50257])
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 119–120 (`out_head`, logits from `GPTModel`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 123–125 (Figure 4.17, softmax is monotonic)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 131–139 (logits → probabilities → cross-entropy 10.7940)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3, book pp. 152–156 (toy logits, temperature, top-k)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 183–186; §6.6, book pp. 190–194 (class logits)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book p. 269; §A.7, book pp. 275–278 (logits into `cross_entropy`, argmax on logits)
