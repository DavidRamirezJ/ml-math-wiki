---
title: Bessel's Correction and Biased Variance
type: concept
tags: [statistics, estimation, normalization, llm, pytorch]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Bessel's Correction and Biased Variance

> **One-liner:** When you measure spread from a sample, dividing by $n$ slightly underestimates the true spread; dividing by $n-1$ (Bessel's correction) fixes that on average.

## 🟢 Intuition (Dummy)
Imagine you want to know how much people's heights vary in a whole country, but you only measure 5 friends. You first work out your friends' average height, then see how far each friend is from *that* average.

Here's the catch: your friends' average was computed from those same 5 people, so it sits right in the middle of them. Their distances to their own average are always a bit *smaller* than their distances to the true national average would be. So the sample makes the spread look a little smaller than it really is.

**Bessel's correction** is a tiny fudge that undoes this: instead of dividing the total squared distance by 5, divide by 4. Smaller denominator, slightly bigger answer, correct on average.

The good news for LLMs: when the "sample" is 768 numbers (one token's embedding), dividing by 768 or by 767 makes almost no difference. So the book's LayerNorm just uses the simpler $n$ version, the one the original GPT-2 used.

## 🟡 Mechanics (Practitioner)
### The two formulas
**Biased** (population-style, divide by $n$):
$$\sigma^2_{\text{biased}} = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2$$
**Unbiased** (Bessel's correction, divide by $n-1$):
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\mu)^2$$

Symbol by symbol:
- $x_i$ = the $i$-th value; $n$ = how many values,
- $\mu = \frac1n\sum_i x_i$ = the sample mean (computed from the same values),
- $(x_i-\mu)^2$ = squared distance of one value from the mean,
- $\sigma^2_{\text{biased}}$ = "biased estimate of the variance" (the book's words),
- $s^2$ = the Bessel-corrected sample variance,
- the two are related by $\sigma^2_{\text{biased}} = \frac{n-1}{n}\,s^2$.

### In PyTorch
```python
x.var(dim=-1, keepdim=True)                  # default: unbiased=True  -> divides by n-1
x.var(dim=-1, keepdim=True, unbiased=False)  # biased -> divides by n
```
(Newer PyTorch spells this `correction=1` / `correction=0`.)

### The book's LayerNorm choice (book p. 103)
Listing 4.2 uses `var = x.var(dim=-1, keepdim=True, unbiased=False)`. The "Biased variance" sidebar explains:
- this does **not** apply Bessel's correction, so it gives a *biased* estimate;
- for LLMs the embedding dimension $n$ is large, so the difference between $n$ and $n-1$ is "practically negligible";
- the choice keeps compatibility with **GPT-2's normalization layers**, reflects **TensorFlow's default behavior** (which the original GPT-2 used), and keeps compatibility with the [[pretrained-weights]].

### Worked numbers (from the book's §4.2 example)
Row 1 of the Linear+ReLU output: $[0.2260, 0.3470, 0, 0.2216, 0, 0]$, $n=6$, mean $0.1324$.
- Sum of squared deviations $\approx 0.1154$.
- Unbiased: $0.1154/5 \approx 0.0231$. That's what the book's `out.var(...)` prints (book p. 101).
- Biased: $0.1154/6 \approx 0.0193$, i.e. $0.0231\times\frac{5}{6}$.

With $n=6$ the gap is about 17%. With GPT-2's $n=768$ the ratio is $\frac{767}{768}\approx 0.9987$, a 0.13% gap. That's why it doesn't matter in practice.

You can even see the biased version at work in the book's output. After `LayerNorm`, the checked variance is 0.9995 and 0.9997, not exactly 1, because $\sigma^2_{\text{biased}}/(\sigma^2_{\text{biased}}+10^{-5}) = 0.0193/0.01931 \approx 0.9995$ (book p. 104).

## 🔴 Theory (Expert)
**Claim.** If $X_1,\dots,X_n$ are i.i.d. with mean $\mu^*$ and variance $\sigma^{*2}$, and $\bar X = \frac1n\sum_i X_i$, then
$$\mathbb{E}\Big[\sum_{i=1}^n (X_i-\bar X)^2\Big] = (n-1)\,\sigma^{*2}.$$
**Proof sketch.** Write $X_i-\bar X = (X_i-\mu^*)-(\bar X-\mu^*)$ and expand:
$$\sum_i (X_i-\bar X)^2 = \sum_i (X_i-\mu^*)^2 - n(\bar X-\mu^*)^2.$$
Take expectations. The first term gives $n\sigma^{*2}$. For the second, $\mathbb{V}[\bar X]=\sigma^{*2}/n$, so it gives $n\cdot\sigma^{*2}/n=\sigma^{*2}$. The difference is $(n-1)\sigma^{*2}$. $\blacksquare$

Consequences:
- $\mathbb{E}[s^2]=\sigma^{*2}$ (unbiased), while $\mathbb{E}[\sigma^2_{\text{biased}}]=\frac{n-1}{n}\sigma^{*2}$ (bias $= -\sigma^{*2}/n \to 0$).
- **Degrees of freedom.** The $n$ deviations $X_i-\bar X$ always sum to zero, so only $n-1$ of them are free. Geometrically, the deviation vector lives in the $(n-1)$-dimensional subspace orthogonal to $\mathbf{1}$.
- **Unbiased ≠ best.** For Gaussian data, dividing by $n+1$ minimizes mean squared error, and dividing by $n$ is the [[maximum-likelihood-estimation|maximum-likelihood]] estimate. Bessel's $n-1$ is only the *unbiased* one. And $s=\sqrt{s^2}$ is still a biased estimator of $\sigma^*$ (by Jensen's inequality).
- **Why it's irrelevant inside LayerNorm.** LayerNorm isn't estimating a population variance. It just rescales one vector by a deterministic function of itself. Any constant factor $c$ in the variance becomes a factor $1/\sqrt{c}$ on the output, and the learned scale $\gamma$ can absorb it. What *does* matter is matching the convention the pretrained weights were trained with.
- MML's *empirical covariance* (Def 6.9) is also the $1/N$ version. The two books agree. PyTorch's `.var()` default ($n-1$) is simply a different library convention.

## Connections
- Builds on: [[covariance-and-variance]], [[expected-value-and-mean]], [[random-variable]], [[statistical-independence]]
- Leads to: [[layer-normalization]], [[standardization-z-score]], [[maximum-likelihood-estimation]]
- Used by: [[layer-normalization]] (`unbiased=False`), [[batch-normalization]], [[pretrained-weights]] compatibility for [[gpt-2]]
- People: [[friedrich-wilhelm-bessel]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 101–104 (unbiased `out.var` output p. 101; Listing 4.2 and "Biased variance" sidebar p. 103; LayerNorm check 0.9995/0.9997 p. 104).
  > ⚠️ Book cross-reference error: the sidebar says the pretrained weights are loaded "in chapter 6". They are actually loaded in §5.5 "Loading pretrained weights from OpenAI" (book pp. 160–167); chapter 6 then reuses them.
- `raw/mml-book.pdf` Def 6.9 (empirical covariance, $1/N$) — see [[covariance-and-variance]].
