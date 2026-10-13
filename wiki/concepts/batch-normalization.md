---
title: Batch Normalization
type: concept
tags: [deep-learning, normalization, training, pytorch]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Batch Normalization

> **One-liner:** Batch normalization standardizes each feature using the average and spread of that feature across the examples in the current mini-batch.

## 🟢 Intuition (Dummy)
Picture a teacher grading a stack of essays. One way to keep scores fair is to compare each essay to the *other essays in today's stack*: "this one is above today's average". That's **batch normalization**: each feature of each example gets compared to the same feature in the other examples of the batch.

The other way is to judge each essay on its own, looking only at its internal balance: "this paragraph is long compared to the essay's other paragraphs". That's [[layer-normalization]].

Batch norm worked great for image networks. But it has a weak spot: if today's stack has only 1 or 2 essays, "today's average" is not reliable. LLMs often run with small or uneven batch sizes (because they're so big), so they use layer normalization instead. The book says exactly this in a short sidebar.

## 🟡 Mechanics (Practitioner)
### The formula
For a mini-batch of $B$ examples and one feature $j$:
$$\mu_j = \frac{1}{B}\sum_{b=1}^{B} x_{b,j},\qquad \sigma_j^2 = \frac{1}{B}\sum_{b=1}^{B}(x_{b,j}-\mu_j)^2,\qquad y_{b,j} = \gamma_j\,\frac{x_{b,j}-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}} + \beta_j$$

Symbol by symbol:
- $x_{b,j}$ = feature $j$ of example $b$ in the batch,
- $B$ = the [[mini-batch]] size,
- $\mu_j$, $\sigma_j^2$ = mean and variance of feature $j$ **across the batch** (down a column),
- $\varepsilon$ = a tiny number to avoid division by zero,
- $\gamma_j$, $\beta_j$ = learned scale and shift for feature $j$,
- $y_{b,j}$ = the normalized output.

(This is the standard definition from Ioffe & Szegedy, 2015. The book names batch norm but does not give its formula.)

### The one-line difference from LayerNorm
In PyTorch terms, for a 2-D tensor `[batch, features]`:
- batch norm: `x.mean(dim=0)`, one statistic **per column** (per feature),
- layer norm: `x.mean(dim=-1)`, one statistic **per row** (per example).

The book's Figure 4.6 shows both directions on its 2 × 6 example (book p. 101):
- `dim=-1` (per row, LayerNorm-style): means 0.13 and 0.21,
- `dim=0` (per column, BatchNorm-style): means $[0.21, 0.29, 0.00, 0.37, 0.16, 0.00]$.

So batch norm on that toy tensor would standardize each of the 6 columns using only 2 numbers. That's a very noisy estimate, which shows the small-batch problem in miniature.

### What the book says
- **§4.2 sidebar, "Layer normalization vs. batch normalization"** (book pp. 104–105): batch norm normalizes **across the batch dimension**; layer norm normalizes **across the feature dimension**. LLMs need a lot of compute, and the hardware or use case can dictate the batch size. Because layer norm normalizes each input **independently of the batch size**, it is more flexible and stable, especially for **distributed training** or **resource-constrained** deployment.
- **§A.7** (book pp. 275–276): `model.train()` and `model.eval()` matter for layers that behave differently in training and inference, quote: "such as dropout or batch normalization layers". The book's own `NeuralNetwork` has neither, so there the calls are redundant, but the book calls it best practice to include them anyway.

### Train vs eval mode
Batch norm really does behave differently in the two modes (standard PyTorch behavior):
- **training** (`model.train()`): it uses the current batch's $\mu_j, \sigma_j^2$, and updates running averages of them,
- **evaluation** (`model.eval()`): it uses the stored running averages, so one example's output no longer depends on the other examples.
- variance convention: PyTorch normalizes with the **biased** ($1/B$) batch variance but stores the **unbiased** ($1/(B-1)$) value in the running average (see [[bessels-correction-biased-variance]]).

```python
bn = torch.nn.BatchNorm1d(num_features=6)   # gamma, beta: 6 each
model.train()   # batch statistics
model.eval()    # running statistics
```
Forgetting `model.eval()` at test time is a classic bug. See [[train-vs-eval-mode]].

## 🔴 Theory (Expert)
- **Coupling across examples.** In training mode, $y_{b,j}$ depends on every other example in the batch through $\mu_j$ and $\sigma_j$. The gradient flows across examples too. That acts as a mild [[regularization|regularizer]] (batch noise), but it also breaks the independence of examples. That's awkward for [[autoregressive-text-generation]] (batch of 1 at inference), for variable-length padded sequences ([[padding-and-truncation]]), and for causal models, where statistics would leak information across positions and sequences.
- **Small batches.** The variance of the estimate $\mu_j$ is $\mathbb{V}[x_j]/B$. For $B\to 1$ the batch variance is 0 and the normalized output is undefined (or 0 thanks to $\varepsilon$). In [[distributed-data-parallel]] training, each GPU sees only its local batch unless you add expensive cross-device syncing ("SyncBatchNorm"). This is the book's flexibility argument, made concrete.
- **Why it helps at all.** The original motivation was reducing "internal covariate shift". Later work (Santurkar et al., 2018) argued the real benefit is a smoother loss landscape, which allows larger [[learning-rate|learning rates]]. Either way, it fights [[vanishing-and-exploding-gradients]], the same motivation the book gives for [[layer-normalization]].
- **Folding at inference.** In eval mode batch norm is a fixed per-feature affine map, so it can be merged into the preceding linear or conv layer's weights and [[bias-vector|bias]].
- **Bias redundancy.** A bias added just before batch norm is cancelled by the mean subtraction. That's why layers followed by BN usually set `bias=False`, and $\beta$ takes over the bias's job.
- **Family tree.** Same recipe, different axes: BatchNorm (over the batch), [[layer-normalization]] (over features, per example), InstanceNorm / GroupNorm (over spatial or channel groups), [[rmsnorm]] (LayerNorm without the mean). All are versions of [[standardization-z-score]].

## Connections
- Builds on: [[standardization-z-score]], [[covariance-and-variance]], [[mini-batch]], [[expected-value-and-mean]]
- Leads to: [[layer-normalization]], [[rmsnorm]], [[train-vs-eval-mode]]
- Used by: convolutional nets and classic deep MLPs (not used in GPT-style LLMs)
- Related: [[vanishing-and-exploding-gradients]], [[dropout]], [[distributed-data-parallel]], [[bessels-correction-biased-variance]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book p. 101 (Fig. 4.6, `dim=0` vs `dim=-1`) and sidebar "Layer normalization vs. batch normalization", book pp. 104–105.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7, book pp. 275–276 (`model.train()` / `model.eval()` for dropout and batch-norm layers).
- Formula, running statistics and theory: standard definitions (Ioffe & Szegedy 2015). They are not given in the book and are labeled as such above.
