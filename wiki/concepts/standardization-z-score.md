---
title: Standardization (z-score)
type: concept
tags: [statistics, preprocessing, deep-learning, normalization, llm]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Standardization (z-score)

> **One-liner:** Standardizing a list of numbers means shifting it so its average is 0 and squashing or stretching it so its spread is 1.

## 🟢 Intuition (Dummy)
Say you scored 80 on a math test and 80 on a history test. Which score is better? You can't tell until you know how everyone else did. If the math class averaged 60 with scores bunched close together, your 80 is amazing. If the history class averaged 85, your 80 is a bit below par.

A **z-score** answers "how many 'typical steps' above or below average am I?" You subtract the average (so "average" becomes 0), then divide by the typical spread (so "one typical step" becomes 1). After that, numbers from very different scales become directly comparable.

Neural networks love this. When the numbers flowing between layers keep drifting to huge or tiny values, training gets wobbly. Standardizing them at each step keeps everything on a calm, predictable scale. That's the core move inside [[layer-normalization]].

## 🟡 Mechanics (Practitioner)
### The formula
For a list $x_1, \dots, x_n$:
$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i,\qquad \sigma^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i-\mu)^2,\qquad z_i = \frac{x_i-\mu}{\sigma}$$

Symbol by symbol:
- $x_i$ = the $i$-th number in the list (in an LLM: the $i$-th feature of one token's vector),
- $n$ = how many numbers there are (in an LLM: the embedding dimension, `emb_dim`),
- $\sum_{i=1}^{n}$ = "add up for every $i$ from 1 to $n$",
- $\mu$ ("mu") = the **mean**, the plain average ([[expected-value-and-mean]]),
- $\sigma^2$ ("sigma squared") = the **variance**, the average squared distance from the mean ([[covariance-and-variance]]),
- $\sigma = \sqrt{\sigma^2}$ = the **standard deviation**, the typical distance from the mean,
- $z_i$ = the **z-score** of $x_i$: how many standard deviations $x_i$ sits above (+) or below (−) the mean.

The book puts it in words: "subtracting the mean and dividing by the square root of the variance (also known as the standard deviation)" (book p. 102).

In deep learning we add a tiny $\varepsilon$ so we never divide by zero ([[numerical-stability]]):
$$\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}},\qquad \varepsilon = 10^{-5}\ \text{in the book's LayerNorm.}$$

### The book's worked example (§4.2, book pp. 100–102)
A `Linear(5, 6)` layer followed by [[relu|ReLU]] produces two rows of activations (seed 123):
```
tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
        [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]])
```
Per-row mean and variance (`dim=-1`, so one value per row):
```python
mean = out.mean(dim=-1, keepdim=True)   # [[0.1324], [0.2170]]
var  = out.var(dim=-1, keepdim=True)    # [[0.0231], [0.0398]]
out_norm = (out - mean) / torch.sqrt(var)
```
Result:
```
tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
        [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]])
Mean:     tensor([[-5.9605e-08], [1.9868e-08]])
Variance: tensor([[1.], [1.]])
```
Check one entry by hand: $(0.3470-0.1324)/\sqrt{0.0231} = 0.2146/0.1520 \approx 1.412$. That matches 1.4126 (up to rounding of the printed inputs). Note that the standardized rows now contain **negative** values, even though the ReLU outputs did not.

The means print as $-5.9605\times10^{-8}$ instead of exactly 0. That's not a bug, just [[floating-point-precision|finite floating-point precision]] (book p. 102). With `torch.set_printoptions(sci_mode=False)` the same means print as `0.0000` (book pp. 102–103).

In the book's `LayerNorm` class (Listing 4.2, book p. 103) the z-score is only the middle step. The class then applies two trainable vectors of length `emb_dim`, `scale` (init ones) and `shift` (init zeros): $y_i = \gamma_i\,\hat{x}_i + \beta_i$. So the model can learn to undo or change the standardization if that helps (see [[layer-normalization]]).

Two practical PyTorch details:
- **`dim=-1`** means "compute across the last axis", so each token's feature vector is standardized on its own. This keeps working for 3-D tensors `[batch, tokens, emb_dim]` (book p. 102).
- **`keepdim=True`** keeps the mean's shape as `[2, 1]` instead of `[2]`, so `out - mean` [[broadcasting|broadcasts]] correctly against `[2, 6]` (book p. 101).

## 🔴 Theory (Expert)
- **Why it works.** Standardization is an affine map $x\mapsto (x-\mu)/\sigma$. For any random variable with finite variance, $\mathbb{E}[(X-\mu)/\sigma]=0$ and $\mathbb{V}[(X-\mu)/\sigma]=\mathbb{V}[X]/\sigma^2=1$, by the affine rules $\mathbb{E}[aX+b]=a\mathbb{E}[X]+b$ and $\mathbb{V}[aX+b]=a^2\mathbb{V}[X]$.
- **Which variance?** If you standardize with one variance formula and then *check* with the same one, you get exactly 1. The book's step-by-step code uses PyTorch's default unbiased $1/(n-1)$ estimator for both, so it prints `1.`. The book's `LayerNorm` class uses the biased $1/n$ version (see [[bessels-correction-biased-variance]]).
- **Effect of $\varepsilon$.** With $\varepsilon>0$ the result's variance is $\sigma^2/(\sigma^2+\varepsilon)$, slightly under 1. In the book's `LayerNorm` test the biased variances are about 0.0193 and 0.0332, which gives 0.9995 and 0.9997, exactly what the book prints (book p. 104).
- **Geometry.** Centering projects the vector $\mathbf{x}\in\mathbb{R}^n$ onto the hyperplane $\sum_i x_i = 0$ (orthogonal to $\mathbf{1}$). Dividing by the (biased) $\sigma$ then rescales it to [[norm|Euclidean norm]] $\sqrt{n}$. So a standardized vector lives on a sphere of radius $\sqrt{n}$ inside that hyperplane: a $(n-2)$-dimensional surface.
- **Degenerate case.** If all $x_i$ are equal, $\sigma=0$ and the z-score is undefined. $\varepsilon$ turns that into $\hat{x}_i = 0$.
- **Standardization vs whitening.** z-scoring fixes each feature's own scale but ignores correlations between features. Whitening multiplies by $\Sigma^{-1/2}$ to also remove correlations ([[data-covariance-matrix]]).
- **Over which axis?** Standardizing each *row* (one example, across features) gives [[layer-normalization]]. Standardizing each *column* (one feature, across the batch) gives [[batch-normalization]]. Classic ML preprocessing (e.g. before [[dimensionality-reduction|PCA]]) standardizes each feature column over the whole dataset.

## Connections
- Builds on: [[expected-value-and-mean]], [[covariance-and-variance]], [[random-variable]], [[broadcasting]]
- Leads to: [[layer-normalization]], [[batch-normalization]], [[rmsnorm]], [[bessels-correction-biased-variance]]
- Used by: [[transformer-block]], [[gpt-architecture]], [[dimensionality-reduction]], [[gaussian-distribution]] (the standard normal $\mathcal{N}(0,1)$ is the z-scored Gaussian)
- Related: [[numerical-stability]], [[floating-point-precision]], [[vanishing-and-exploding-gradients]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 99–104 (motivation p. 99, Fig. 4.5 p. 100, `dim`/`keepdim` pp. 101–102, step-by-step standardization p. 102, `LayerNorm` with $\varepsilon$ pp. 103–104).
  > ⚠️ Book inconsistency: Fig. 4.5 (book p. 100) labels the pre-normalization variance as 0.39. The code for the same row prints 0.0231 (book p. 101), and 0.0231 is the value consistent with the normalized outputs. The 0.39 in the figure looks wrong.
