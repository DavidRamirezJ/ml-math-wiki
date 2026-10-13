---
title: Log Probability
type: concept
tags: [probability, llm, numerical-stability, loss-function]
difficulty: beginner
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Log Probability

> **One-liner:** A log probability is just the logarithm of a probability. It turns tiny numbers like 0.00007 into friendly ones like −9.5, and turns multiplying into adding.

## 🟢 Intuition (Dummy)
LLM probabilities are often *tiny*. An untrained GPT-2 gives the right next word
a chance of about 0.00007. Multiply a few hundred of those together (one per word
in a paragraph) and the result is so small that a computer rounds it to zero.

The logarithm fixes this. It is like measuring with a "how many zeros?" ruler
instead of a normal ruler:
- probability 1 (certain) → log 0
- probability 0.5 → about −0.69
- probability 0.00007 → about −9.5
- probability close to 0 → a large negative number

All log probabilities are **zero or negative**. Closer to zero means "the model
liked this answer more". And since logs turn products into sums, the
"probability of a whole sentence" becomes a simple sum of per-word scores.

## 🟡 Mechanics (Practitioner)

### Definition
$$\log p, \qquad 0<p\le 1 \;\Longrightarrow\; -\infty<\log p\le 0$$
- $p$ = a probability (here: the model's probability of the correct next token).
- $\log$ = the **natural** logarithm (base $e\approx2.718$), which is what `torch.log` computes.
- $\Longrightarrow$ = "which means".
- The range says: a log probability is never positive, and it falls toward $-\infty$ as $p$ approaches 0.

### The book's worked numbers (book pp. 136–138)
The six target-token probabilities from the untrained GPT-2 model (two texts, three tokens each):
```python
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# target_probas_1 = [7.4541e-05, 3.1061e-05, 1.1563e-05]
# target_probas_2 = [1.0337e-05, 5.6776e-05, 4.7559e-06]
print(log_probas)
# tensor([ -9.5042, -10.3796, -11.3677, -11.4798,  -9.7764, -12.2561])
```
Check one by hand: $\ln(7.4541\times10^{-5})=\ln 7.4541+\ln 10^{-5}\approx 2.0088-11.5129=-9.5042$.

- `torch.cat` glues the two length-3 tensors into one of length 6.
- `torch.log` takes the natural log of every entry.

### Average log probability (book p. 138)
```python
avg_log_probas = torch.mean(log_probas)   # tensor(-10.7940)
```
$$\overline{\log p}=\frac{1}{N}\sum_{n=1}^{N}\log p_n$$
- $\overline{\log p}$ = "the average log probability" (the bar means average).
- $N$ = number of tokens scored (6 here).
- $\sum_{n=1}^{N}$ = add up over $n=1$ to $N$.
- $p_n$ = probability of the correct token at position $n$.

The training goal is to push this average **up toward 0**, which would mean
probability 1 on every correct token (book p. 138). Deep-learning code prefers
to *minimize*, so it flips the sign: $-\overline{\log p}=10.7940$, the
[[cross-entropy-loss]] (book p. 138).

### Why logs instead of raw probabilities
The book says log probabilities are "more manageable in mathematical
optimization" than raw probabilities and leaves the details to a lecture listed
in appendix B (book p. 138). The concrete reasons:
1. **No underflow.** A product of many probabilities like $10^{-5}$ quickly
   drops below the smallest number a float can hold. Sums of logs stay in a
   normal range.
2. **Products become sums.** $\log(p_1p_2\cdots p_N)=\log p_1+\cdots+\log p_N$,
   and sums are easy to average and to differentiate.
3. **Same winner.** $\log$ is a [[monotonic-function|monotonic]] (order-keeping)
   function, so the most likely option is still the most likely after the log.

## 🔴 Theory (Expert)
- **Sequence log-likelihood.** An autoregressive model factorizes a sequence as
  $p_\theta(x_1,\dots,x_T)=\prod_{t}p_\theta(x_{t+1}\mid x_{\le t})$ (the
  probability chain rule). Taking logs gives
  $\log p_\theta(x_{1:T})=\sum_t\log p_\theta(x_{t+1}\mid x_{\le t})$. The
  average of this sum, negated, is the cross-entropy loss, and its exponential
  is the [[perplexity]]. Maximizing it is [[maximum-likelihood-estimation]].
- **Log-softmax and log-sum-exp.** With logits $\mathbf z$,
  $\log\operatorname{softmax}(\mathbf z)_i=z_i-\log\sum_j e^{z_j}$. Computing it
  this way (with the max-shift
  $\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m}$, $m=\max_j z_j$) never forms the
  tiny probability at all. This is why PyTorch's `cross_entropy` takes logits
  directly. See [[numerical-stability]].
- **Information content.** $-\log p$ is the *surprisal* of an outcome: in bits
  (base 2) it is the number of yes/no questions needed to identify it. The
  untrained model's average surprisal of 10.794 nats ≈ 15.6 bits per token is
  close to $\log_2 50257\approx15.6$ bits, the cost of a uniform guess.
- **Gradient friendliness.** $\frac{d}{dp}\log p=\frac1p$: when the model gives
  the right answer a tiny probability, the gradient is large, so learning pushes
  hardest on the worst mistakes. Combined with softmax, the gradient with
  respect to the logits simplifies to $\mathbf p-\mathbf e_y$.
- **Uses beyond the loss.** Summed log probabilities score whole candidate
  sequences in [[beam-search]], and differences of log probabilities between a
  policy and a reference model are the core quantity in
  [[direct-preference-optimization]].

## Connections
- Builds on: [[probability-distribution]], [[softmax]], [[monotonic-function]], [[random-variable]]
- Leads to: [[cross-entropy-loss]], [[perplexity]], [[maximum-likelihood-estimation]]
- Used by: [[pretraining]], [[next-token-prediction]], [[beam-search]], [[direct-preference-optimization]]
- Related: [[numerical-stability]], [[expected-value-and-mean]], [[logits]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.2, book pp. 136–138 (target probabilities, Fig. 5.7 six-step loss, `torch.log`, average −10.7940, sign flip to 10.7940, "more manageable in mathematical optimization")
