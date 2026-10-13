---
title: Next-Token (Next-Word) Prediction
type: concept
tags: [llm, language-modeling, probability, self-supervised-learning, training-objective]
difficulty: intermediate
aliases: [next-word prediction, language modeling, input-target pairs, next-token prediction]
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Next-Token (Next-Word) Prediction

> **One-liner:** Show the model the start of a text and ask "what comes next?" — over and over, on billions of words — and you have the training task behind every GPT-style LLM.

## 🟢 Intuition (Dummy)
It's a fill-in-the-blank game, but the blank is always at the end:

> "The model is simply trained to predict the next ___" → **word**

That's it. That's the whole pretraining task of GPT (book Fig. 1.7). The clever
part is where the answers come from: **the text itself**. Take any sentence,
hide the next word, and the hidden word *is* the answer key. Nobody has to label
anything, so you can train on mountains of text. This is called
[[self-supervised-learning]] — the data labels itself.

And one sentence gives you *many* practice questions at once:

```
LLMs            → learn
LLMs learn      → to
LLMs learn to   → predict
...
```

The one rule: the model may only look at words **before** the blank, never
after it (it "can't access words past the target", Fig. 2.12). Otherwise it
would just be copying the answer.

When the model *uses* this skill, it plays the game in a loop: predict a word,
stick it on the end, predict again. That's how it writes whole paragraphs
([[autoregressive-text-generation]]).

## 🟡 Mechanics (Practitioner)

### 1. The probability view
The model learns one conditional distribution over the [[vocabulary]] (standard
notation; the book describes it in words and implements it in Ch. 2–5):

$$P(x_t \mid x_1, \dots, x_{t-1})$$

- $x_t$ = the token at position $t$ (the one to guess).
- $x_1,\dots,x_{t-1}$ = the context: every token before it.
- $\mid$ = "given".
- $P(\cdot)$ = a [[probability-distribution]] over all 50,257 possible tokens (for GPT-2).

Chain them together with the chain rule of probability and you get the
probability of a whole text:

$$P(x_1,\dots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_1,\dots,x_{t-1})$$

- $T$ = number of tokens; $\prod$ = "multiply all of these together".

### 2. Making input–target pairs: "shift by one" (§2.6)
The targets are just the inputs **shifted one position**. From "The Verdict"
(5,145 BPE tokens), with `context_size = 4` (p. 36):

```python
x = enc_sample[:context_size]        # [290, 4920, 2241, 287]
y = enc_sample[1:context_size+1]     # [4920, 2241, 287, 257]
```

In symbols, for a window that starts at index $s$ with length $L$:

$$x = (t_s,\dots,t_{s+L-1}), \qquad y = (t_{s+1},\dots,t_{s+L})$$

- $t_k$ = token ID at position $k$; $L$ = context length (`max_length`); $s$ = window start.
- So $y_j = x_{j+1}$: each target is "the next one".

One window of length 4 hides **four** prediction tasks (p. 36):

```
 and                            ---->  established
 and established                ---->  himself
 and established himself        ---->  in
 and established himself in     ---->  a
```

A [[sliding-window-sampling|sliding window]] (`GPTDatasetV1`, Listing 2.5) cuts
the whole text into such pairs; `stride` sets how far the window moves. Batches
come out as `(batch_size, max_length)` tensors of token IDs.

### 3. From logits to a loss (§5.1.2)
The model outputs one [[logits|logit]] vector per input position — shape
`[batch, tokens, 50257]`. [[softmax]] turns each into probabilities:

$$\mathrm{softmax}(\mathbf z)_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

- $\mathbf z$ = logit vector at one position; $z_i$ = score for vocabulary token $i$;
  $V$ = 50,257 = vocabulary size; $e$ = Euler's number (so every output is positive and they sum to 1).

Then we look up the probability the model gave to the **correct** next token,
take the log, average, and flip the sign — that's the [[cross-entropy-loss]]:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{n=1}^{N} \log p_\theta(y_n \mid \text{context}_n)$$

- $N$ = number of predicted positions (batch size × tokens).
- $y_n$ = the correct target token at position $n$.
- $p_\theta(y_n\mid\cdot)$ = the softmax probability the model (weights $\theta$) gave to that correct token.
- The minus sign: logs of probabilities are negative, so we flip them to get a positive number to *minimize*.

**The book's worked numbers** (inputs "every effort moves" / "I really like",
targets " effort moves you" / " really like chocolate", untrained GPT-2 124M):

| Step | Value |
|---|---|
| target probabilities | `[7.4541e-05, 3.1061e-05, 1.1563e-05]`, `[1.0337e-05, 5.6776e-05, 4.7559e-06]` |
| log probabilities | `[-9.5042, -10.3796, -11.3677, -11.4798, -9.7764, -12.2561]` |
| average log prob | $-64.7638/6 = -10.7940$ |
| loss (negated) | **10.7940** |
| [[perplexity]] $=e^{\mathcal L}$ | **48,725.8** (the model is as unsure as if picking among ~48,725 of 50,257 tokens) |

PyTorch does all of it in one call (p. 139):

```python
logits_flat  = logits.flatten(0, 1)   # [6, 50257]
targets_flat = targets.flatten()      # [6]
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)  # tensor(10.7940)
```

Sanity check: an untrained model starts near uniform, about $1/50{,}257\approx0.00002$
per token (p. 135), so the loss starts near $\ln 50{,}257\approx 10.8$. On the
whole "The Verdict" dataset the untrained losses are 10.988 (train) and 10.981
(validation) (p. 145).

### 4. The "no peeking" rule inside the model (§3.5)
Because each position must predict the *next* token, attention may only look at
the current and earlier tokens. [[causal-attention-mask|Causal masking]] sets
scores above the diagonal to $-\infty$ before softmax, so future tokens get weight
$e^{-\infty}=0$. This is what lets one forward pass train all positions of a
window in parallel without leaking answers.

### 5. Same idea at fine-tuning time (§7.3)
In [[instruction-fine-tuning]], targets are still the inputs shifted by one, with
an end-of-text token (ID 50256) appended (Fig. 7.10, p. 217):

```
input  [0, 1, 2, 3, 4]      →  target [1, 2, 3, 4, 50256]
```

## 🔴 Theory (Expert)
- **It is maximum likelihood.** Minimizing $\mathcal L$ is the same as maximizing
  $\sum_n \log p_\theta(y_n\mid\text{context}_n)$, i.e.
  [[maximum-likelihood-estimation]] of an autoregressive model. Equivalently, it
  minimizes the cross-entropy $H(q,p)=-\sum_i q_i\log p_i$ between the one-hot
  target distribution $q$ and the model's distribution $p$; with one-hot $q$ this
  reduces to $-\log p_y$. Averaged over a dataset it is
  [[empirical-risk-minimization]] with log-loss.
- **Why logs.** The book only says that working with log-probabilities "is more
  manageable in mathematical optimization than handling the scores directly"
  and defers details to a lecture listed in App. B (p. 138). The standard reasons:
  products of many tiny probabilities (like $10^{-5}$) underflow, while logs turn
  them into sums and give better-behaved gradients.
- **Exactness of the factorization.** The chain rule $P(x_{1:T})=\prod_t
  P(x_t\mid x_{<t})$ holds for *any* joint distribution — it is not a
  modeling assumption. The modeling choice is that one shared network with fixed
  weights $\theta$ produces every conditional, and that the context is truncated
  to a maximum [[context-length]] (1,024 for GPT-2).
- **Why a simple task gives rich skills.** To predict the next word well across
  diverse text, a model must capture syntax, facts and cross-lingual patterns;
  the book cites [[emergent-behavior]] such as translation as an unexpected
  by-product (§1.6, p. 14).
- **Self-supervision vs. supervision.** Appendix A frames it within the usual
  supervised workflow (training on examples + labels, then inference): during
  pretraining the labels are simply derived from the text itself (A.1.2, p. 254).
  The term "self-supervised" comes from §1.3 and §1.6.
- **Data efficiency of windows.** A window of length $L$ yields $L$ training
  signals in one forward pass. Overlapping windows (stride < $L$) reuse tokens and
  can encourage [[overfitting]]; the book uses stride = $L$ to cover the data
  without overlap (§2.6, p. 41).
- **Contrast: masked LM.** [[bert]] instead predicts hidden words *inside* a
  sentence using both sides ([[masked-language-modeling]]), which suits
  classification better than generation (§1.4).

## Connections
- Builds on: [[probability-distribution]], [[random-variable]], [[softmax]], [[logits]], [[token]], [[token-id]], [[tokenization]], [[vocabulary]]
- Leads to: [[cross-entropy-loss]], [[perplexity]], [[pretraining]], [[autoregressive-text-generation]], [[llm-training-loop]], [[decoding-strategies]], [[greedy-decoding]]
- Used by: [[large-language-model]], [[gpt-architecture]], [[decoder-only-transformer]], [[causal-attention]], [[instruction-fine-tuning]]
- Related math: [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[loss-function]], [[log-probability]], [[self-supervised-learning]], [[sliding-window-sampling]], [[causal-attention-mask]], [[masked-language-modeling]]
- People: [[alec-radford]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1, book p. 2 — next-word objective.
- Same file §1.3, book pp. 6–7 — self-supervised pretraining, Fig. 1.3.
- Same file §1.6, book pp. 12–14 — Fig. 1.7, self-labeling, autoregressive decoding, emergent translation.
- Same file §1.7 + summary, book pp. 14–15.
- Same file §2.1, book p. 17 — next-word task recap.
- Same file §2.6, book pp. 35–41 — input–target pairs, Listing 2.5/2.6, Figs. 2.12–2.14.
- Same file §3.5, book pp. 74–78 — causal masking.
- Same file §4.1, book p. 98 — Fig. 4.4, output shifted by one.
- Same file §5.1, book pp. 128–146 — Listing 5.1, Figs. 5.4–5.7, cross entropy, perplexity, train/val losses.
- Same file §7.3, book pp. 216–218 — Fig. 7.10, shifted targets in instruction batches.
- Same file §A.1.2, book pp. 253–254 — supervised workflow and next-word labels.
