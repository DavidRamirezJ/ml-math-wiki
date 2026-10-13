---
title: Monte Carlo Estimation
type: concept
tags: [probability, statistics, sampling, estimation]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Monte Carlo Estimation

> **One-liner:** If you can't (or don't want to) calculate a probability or an average exactly, simulate the random thing many times and count — the count gets closer to the truth the more you simulate.

## 🟢 Intuition (Dummy)
Want to know how often a weighted die lands on 6? You could do the math, or you could
just roll it 1,000 times and count the sixes. If you got 170 sixes, your best guess is
"about 17%". Roll 100,000 times and your guess gets even better.

That's Monte Carlo estimation (named after the casino in Monaco — it's math by gambling).
The book does exactly this in Chapter 5: it "rolls" the language model's next-word die
1,000 times and counts how often each word comes up. "forward" appeared 582 times, so
its estimated probability is 58.2%. The true value, straight from the softmax, is 57.2%.
Close, but not identical — because 1,000 rolls is a finite number.

The book's Exercise 5.1 then asks the key question: "is there a faster and more
accurate way?" Yes — when you already *have* the exact probabilities, just read them
off. Monte Carlo is for when you don't.

## 🟡 Mechanics (Practitioner)

### The formula
To estimate the probability of an event $A$ (e.g. "the sampled token is *forward*"):
$$\hat p_N = \frac{1}{N}\sum_{n=1}^{N}\mathbb 1[x^{(n)}\in A] = \frac{\text{number of hits}}{N}$$
More generally, to estimate an expected value:
$$\mathbb E[f(X)] \approx \frac{1}{N}\sum_{n=1}^{N} f\big(x^{(n)}\big), \qquad x^{(n)}\sim p$$

Reading it symbol by symbol:
- $N$ = how many random samples you draw (the book uses $N=1{,}000$).
- $x^{(n)}$ = the $n$-th sample, drawn from the distribution $p$ (the symbol $\sim$ means
  "drawn from").
- $\mathbb 1[\cdot]$ = the *indicator*: it is 1 if the condition inside is true, else 0.
  Summing indicators just counts hits.
- $\hat p_N$ ("p-hat") = your estimate. The hat means "estimated, not exact".
- $\mathbb E[f(X)]$ = the true average ([[expected-value-and-mean|expected value]]) of
  $f(X)$; a probability is the special case $f=\mathbb 1[\cdot\in A]$.

How wrong can you expect to be? The **standard error** of a frequency estimate is
$$\mathrm{SE}(\hat p_N)=\sqrt{\frac{p(1-p)}{N}}$$
($p$ = the true probability). Error shrinks like $1/\sqrt N$: to get 10× more accurate,
you need 100× more samples.

### The book's worked example (§5.3.1, book pp. 153–154)
```python
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
```
Estimates vs. exact softmax probabilities (counts from the book; exact values and
standard errors computed from the book's logits):

| token | count / 1,000 | estimate $\hat p$ | exact $p$ | SE | off by |
|---|---|---|---|---|---|
| forward | 582 | 0.582 | 0.572 | 0.016 | +0.6 SE |
| toward | 343 | 0.343 | 0.358 | 0.015 | −1.0 SE |
| closer | 73 | 0.073 | 0.061 | 0.008 | +1.6 SE |
| inches | 2 | 0.002 | 0.003 | — | — |

Everything is within about 1–2 standard errors: exactly what random sampling noise
looks like. The book itself summarizes: "forward" is sampled "about 60%" of the time at
$T=1$ (book p. 154).

**Exercise 5.1** (book p. 155; solution appendix C, p. 304): how often is "pizza"
sampled at temperatures 1, 0.1 and 5?
- Monte Carlo answer: 0 times at low temperature, **32 out of 1,000** at $T=5$ →
  estimate 3.2%. (Appendix C writes "0x if the temperature is 0 or 0.1"; the $T=1$ run in
  §5.3.1 also printed "0 x pizza", book p. 154.)
- Exact answer: read the probability off the temperature-scaled softmax,
  `scaled_probas[2][6]` = **4.3%**. Faster *and* exact.
- Gap: 3.2% vs. 4.3% is about 1.7 standard errors (SE $=\sqrt{0.043\cdot0.957/1000}\approx0.0064$) —
  plausible noise, not a bug.

The point of the exercise: Monte Carlo is a tool for when you *can't* compute $p$
directly. Here we can, so sampling is the slow, noisy way.

## 🔴 Theory (Expert)
- **Unbiasedness.** $\mathbb E[\hat p_N]=p$ — on average the estimate is right. Each
  indicator $\mathbb 1[x^{(n)}\in A]$ is Bernoulli($p$), so $N\hat p_N\sim\mathrm{Binomial}(N,p)$
  and $\operatorname{Var}(\hat p_N)=p(1-p)/N$ ([[covariance-and-variance]]).
- **Consistency — law of large numbers.** For i.i.d. samples with
  $\mathbb E|f(X)|<\infty$, the sample mean converges to $\mathbb E[f(X)]$ almost surely
  (strong LLN). So $\hat p_N\to p$ as $N\to\infty$.
- **Error distribution — central limit theorem.** If $\operatorname{Var}f(X)=\sigma^2<\infty$,
  $\sqrt N(\hat\mu_N-\mu)\xrightarrow{d}\mathcal N(0,\sigma^2)$ ([[gaussian-distribution]]),
  giving the approximate 95% interval $\hat\mu_N\pm1.96\,\sigma/\sqrt N$. For "forward":
  $0.582\pm0.031$, which contains the true 0.572.
- **Dimension-free rate.** The $O(N^{-1/2})$ rate does not depend on the dimension of
  $X$ — the reason Monte Carlo beats grid-based numerical integration in high dimensions,
  and why it is everywhere in ML (Bayesian inference, policy-gradient RL, dropout-based
  uncertainty).
- **Rare events are expensive.** Relative error is $\mathrm{SE}/p=\sqrt{(1-p)/(Np)}$; for
  $p=10^{-4}$ ("pizza" at $T=1$) you'd need ~$10^6$ samples for 10% relative error. This is
  why 1,000 samples showed "pizza" 0 times at $T=1$ even though its probability is not zero.
  Remedies: importance sampling, or (best) compute exactly when you can.
- **Connection to training.** The training loss itself is a Monte Carlo estimate: the
  [[empirical-risk-minimization|empirical risk]] (average [[cross-entropy-loss]] over a
  [[mini-batch]]) estimates the expected loss under the data distribution, and its
  gradient is the noisy estimate that [[stochastic-gradient-descent]] follows.
- **Pseudo-randomness.** `torch.manual_seed(123)` makes the "random" samples a fixed,
  reproducible sequence — the estimate is then deterministic for that seed, but a
  different seed gives a different (equally valid) estimate
  ([[random-seed-reproducibility]]).

## Connections
- Builds on: [[random-variable]], [[expected-value-and-mean]], [[probability-distribution]], [[histogram]]
- Uses: [[multinomial-sampling]] (to generate the samples), [[covariance-and-variance]] (for the error)
- Leads to: [[stochastic-gradient-descent]], [[empirical-risk-minimization]], [[bayesian-inference]]
- Used by: sampling-frequency checks of [[temperature-scaling]] (Exercise 5.1)
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.1, book pp. 153–155 (`print_sampled_tokens`, 1,000 samples, counts 73/582/343/2, Exercise 5.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book p. 304 (Exercise 5.1 solution: 32/1,000 = 3.2% vs. exact 4.3%).
- Standard results (LLN, CLT, standard error) are textbook probability, not stated in the LLM book.
