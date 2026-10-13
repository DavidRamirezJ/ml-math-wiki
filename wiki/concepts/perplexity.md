---
title: Perplexity
type: concept
tags: [llm, evaluation, probability, information-theory]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Perplexity

> **One-liner:** Perplexity tells you how many words the model is "torn between" at each step. It is the cross-entropy loss turned back into a count.

## 🟢 Intuition (Dummy)
A loss of 10.79 is hard to picture. Is that bad? How bad? Perplexity turns it
into something you can feel: **"the model is as confused as if it were picking
fairly among *this many* words."**

- Perplexity 1: the model is certain and right every time.
- Perplexity 2: like flipping a coin between two words.
- Perplexity 48,726: like rolling a die with about 48,726 sides. That is almost
  the full GPT-2 vocabulary (50,257 tokens), so the model knows almost nothing.

Lower is better. A model that has really learned English will have a small
perplexity, because at most steps only a handful of next words make sense.

## 🟡 Mechanics (Practitioner)

### The formula
$$\text{PPL}=e^{\mathcal L}=\exp\!\Big(-\frac{1}{N}\sum_{n=1}^{N}\log p_n\Big)$$

Reading it symbol by symbol:
- $\text{PPL}$ = perplexity.
- $e^{\,\cdot}$ or $\exp(\cdot)$ = the exponential function, the inverse of the natural log.
- $\mathcal L$ = the [[cross-entropy-loss]] (average negative log-probability).
- $N$ = number of predicted tokens.
- $p_n$ = probability the model gave to the **correct** token at position $n$.
- $\log p_n$ = its [[log-probability]] (a negative number).

So: average the log-probabilities, flip the sign (that is the loss), then undo
the log with $e$.

### The book's example (book pp. 139–140)
```python
perplexity = torch.exp(loss)   # loss = tensor(10.7940)
print(perplexity)              # tensor(48725.8203)
```
The untrained GPT-2 124M model on two 3-token examples has loss 10.7940, so its
perplexity is about **48,726**. The book reads this as the model being unsure
among about 48,725 tokens of the 50,257-token vocabulary, i.e. the **effective
vocabulary size** it is uncertain about at each step (book p. 140).

### Reading other losses in the book as perplexities
The book only computes the one perplexity above. Converting its other losses
with $e^{\mathcal L}$ (our arithmetic):

| Situation | Loss | Perplexity |
|---|---|---|
| Uniform guess over 50,257 tokens | $\ln 50257\approx10.82$ | 50,257 |
| Untrained model, first training step (§5.2, p. 149) | 9.781 | ≈ 17,700 |
| After 10 epochs, **training** set (p. 149) | 0.391 | ≈ 1.48 |
| After 10 epochs, **validation** set (p. 149) | 6.452 | ≈ 634 |
| Pretrained GPT-2 medium on the instruction data, before fine-tuning (§7.6, p. 230) | 3.826 | ≈ 46 |

The train-vs-validation gap (1.48 vs 634) shows [[overfitting]] very clearly:
on text it memorized the model is almost certain; on new text it is still choosing
among hundreds of options.

### Why use it alongside the loss
Both carry the same information (one is a monotone function of the other), so
ranking two models by loss or by perplexity gives the same order. Perplexity is
"a more interpretable way" to see the model's uncertainty (book p. 139), and
lower perplexity means the model's predicted distribution is closer to the real
word distribution in the data (book p. 139).

## 🔴 Theory (Expert)
- **Geometric mean of inverse probabilities.**
  $\text{PPL}=\big(\prod_{n=1}^{N}p_n\big)^{-1/N}=\dfrac{1}{\sqrt[N]{p_1p_2\cdots p_N}}$.
  So perplexity is the reciprocal of the *geometric-mean* probability of the
  correct tokens. The logs in $\mathcal L$ exist to turn this product into a sum.
- **Uniform-die interpretation.** If a model gives probability $1/k$ to the
  correct token at every step, then $\mathcal L=\log k$ and $\text{PPL}=k$. Any
  model with perplexity $k$ is "as uncertain as" a fair $k$-sided die, which is
  where "effective vocabulary size" comes from.
- **Bounds.** $1\le\text{PPL}$, with 1 only for perfect certainty on every target.
  A model that is never worse than uniform has $\text{PPL}\le V$. The untrained
  model's loss (10.988 on "The Verdict", book p. 145) is slightly above
  $\ln V$, so its perplexity is slightly above $V$.
- **Information-theory link.** With base-2 logs, $\log_2\text{PPL}$ is the
  cross-entropy in **bits per token**: the average number of yes/no questions the
  model needs to pin down the next token. Perplexity is $2^{H}$ (bits) or $e^{H}$
  (nats); the base cancels, the number is the same.
- **Caveats.** Perplexity depends on the tokenizer: a model with a larger
  vocabulary or coarser tokens predicts fewer, harder tokens, so per-token
  perplexities across tokenizers are not directly comparable. It also measures
  only likelihood, not usefulness; the book later uses task metrics and an
  [[llm-as-a-judge]] for instruction models.
- **Exponential amplification.** Small loss changes are big perplexity changes:
  going from loss 3.8 to 3.5 cuts perplexity from ≈45 to ≈33. Report both when
  comparing models.

## Connections
- Builds on: [[cross-entropy-loss]], [[log-probability]], [[probability-distribution]], [[softmax]]
- Leads to: [[llm-evaluation]], [[training-and-validation-loss]]
- Used by: [[pretraining]], [[large-language-model]] evaluation
- Related: [[overfitting]], [[maximum-likelihood-estimation]], [[vocabulary]], [[monotonic-function]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.2, "Perplexity" sidebar, book pp. 139–140 (formula `torch.exp(loss)`, value 48725.8203, effective vocabulary size)
- Same, §5.1.3, book p. 145 (untrained losses 10.988 / 10.981)
- Same, §5.2, book p. 149 (training-run losses used in the conversion table)
- Same, §7.6, book p. 230 (initial instruction-tuning loss 3.826)
