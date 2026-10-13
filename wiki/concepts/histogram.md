---
title: Histogram
type: concept
tags: [density-estimation, nonparametric, statistics]
difficulty: beginner
sources: [mml-ch11-density-estimation-gaussian-mixture-models, llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-06-04
updated: 2026-10-13
---

# Histogram

> **One-liner:** A histogram chops your data's range into bins and draws a bar for each bin as tall as the number of points that fell inside it.

## 🟢 Intuition (Dummy)
Suppose you have a pile of test scores and want to see their "shape." Lay out boxes along the score axis
(0–10, 10–20, …), drop each score into its box, then draw a bar over each box whose height is how many
scores landed there. The result — a row of bars — shows you where scores cluster and where they're sparse.
That's a histogram: the simplest way to turn raw numbers into a picture of their distribution. The one
tricky choice is how wide to make the boxes.

## 🟡 Mechanics (Practitioner)
A histogram is a **nonparametric** density estimate: partition the data space into **bins**, count how
many of the $N$ data points fall into each bin, and draw a bar at the bin center with height proportional
to that count. To read it as a density, normalize so the total area is 1: a bin of width $w$ containing
$n_b$ points gets height
$$ \hat p_{\text{bin}} = \frac{n_b}{N\,w} . $$

- **Bin size (bin width)** is the critical hyperparameter. Too wide → underfitting (oversmoothed, hides
  structure); too narrow → overfitting (spiky, noisy). Choose it by cross-validation.
- It is **unsmoothed**: the estimate is piecewise constant and jumps at bin edges; the result also depends
  on where the edges are placed.

**ML usage.** Fast exploratory data analysis, sanity-checking distributions, and a baseline density
estimator. When you need a *smooth* estimate instead of blocky bars, use [[kernel-density-estimation]].

### In LLMs (Raschka, *Build a LLM from Scratch*)
For text, the "bins" are not intervals on a number line. They are the **discrete categories** of
the vocabulary: one bin per [[token-id]], each bin exactly one ID wide. A histogram is then simply
a table of counts per token.

**1. Counting sampled tokens with `torch.bincount` (§5.3.1).** To see how random
[[multinomial-sampling]] behaves, the book samples the next token 1,000 times from the
probabilities of a 9-token toy vocabulary and counts each ID:

```python
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
```
Output (book pp. 153–154): **73 × closer, 0 × every, 0 × effort, 582 × forward, 2 × inches,
0 × moves, 0 × pizza, 343 × toward**. ("you", ID 8, is not printed: `bincount` returns
max-sampled-ID + 1 = 8 bins.)

`torch.bincount` *is* a histogram with unit-width integer bins:
$$n_k = \sum_{s=1}^{N} \mathbb{1}[\,x_s = k\,], \qquad \hat p_k = \frac{n_k}{N}$$
Symbol by symbol:
- $x_s$ = the token ID drawn in sample $s$; $N$ = number of samples (1,000).
- $\mathbb{1}[x_s = k]$ = "1 if sample $s$ landed in bin $k$, else 0" (the indicator function).
- $n_k$ = the count (bar height) for token $k$; $\hat p_k$ = its relative frequency.

Here the bin width is $w=1$, so the density formula above, $n_b/(N w)$, reduces to $n_k/N$:
$\hat p_{\text{forward}} = 582/1000 = 0.582$, close to the true softmax probability $\approx 0.57$
(book: "about 60%", p. 154). This is [[monte-carlo-estimation]] of a categorical
[[probability-distribution]]. Exercise 5.1 (p. 155) asks for "a faster and more accurate way" to
find how often "pizza" is sampled: read $p_{\text{pizza}}$ straight from the temperature-scaled
[[softmax]] instead of counting ([[temperature-scaling]]).

**2. Pair-frequency counts in BPE (§2.5).** [[byte-pair-encoding]] builds its vocabulary by
"iteratively merging frequent characters into subwords and frequent subwords into words", with
merges decided by "a frequency cutoff" (book pp. 34–35). Each merge step is: build a histogram of
adjacent symbol pairs in the corpus, take the tallest bar ($\arg\max$), merge it, repeat.

> Note: Figure 5.14 (book p. 155) is a **bar chart of probabilities**, not a histogram of data.
> The bars show the model's $p_i$ at temperatures 1, 0.1, 5; the `bincount` output is the
> matching empirical histogram.

## 🔴 Theory (Expert)
- Proposed by **Karl Pearson (1895)**; the histogram is the oldest nonparametric density estimator.
- It is a special case of binned density estimation: equivalent to a piecewise-constant maximum-likelihood
  density over a fixed bin partition. KDE generalizes it by replacing the hard bin indicator with a smooth
  kernel and removing the dependence on edge placement.
- **Bias–variance trade-off** governs bin width exactly as bandwidth does for [[kernel-density-estimation|KDE]];
  the optimal width shrinks as $N$ grows. In $d$ dimensions the number of bins grows like (bins-per-axis)$^d$,
  so histograms suffer badly from the curse of dimensionality and are rarely used beyond 1–2 dimensions.
- **Categorical case (LLMs).** On a finite vocabulary the histogram with unit bins is exactly the
  **maximum-likelihood estimate** of a categorical distribution, $\hat p_k = n_k/N$ (see
  [[maximum-likelihood-estimation]]). There is no bin-width choice and no edge-placement problem.
  By the law of large numbers $\hat p_k\to p_k$, with standard error $\sqrt{p_k(1-p_k)/N}$. For
  "forward" at $N=1000$: $\sqrt{0.57\cdot0.43/1000}\approx0.016$, so 0.582 is well within
  noise of the true $\approx0.572$. Rare tokens ("pizza", $p\approx10^{-4}$ at $T=1$) almost
  always get 0 counts; that's why Exercise 5.1 prefers reading the probability directly.

## Connections
- Builds on: counting, binning, [[probability-distribution]], [[random-variable]]
- Leads to: [[kernel-density-estimation]], [[monte-carlo-estimation]]
- Used by: exploratory data analysis, density-estimation baselines, [[multinomial-sampling]] diagnostics, [[byte-pair-encoding]] merges
- Related: [[probability-distribution]], [[random-variable]], [[decoding-strategies]], [[token-id]]
- People: —

## Sources
- `raw/mml-book.pdf` §11.5, printed pp. 368–369 (PDF pp. 374–375).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.5, book pp. 34–35 (BPE frequency-based merges, frequency cutoff).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.1, book pp. 152–155 (`torch.bincount` over 1,000 multinomial samples, Fig. 5.14, Exercise 5.1).
