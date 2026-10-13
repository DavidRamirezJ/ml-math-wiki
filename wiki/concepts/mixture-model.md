---
title: Mixture Model
type: concept
tags: [density-estimation, probability, machine-learning, llm]
difficulty: beginner
sources: [mml-ch11-density-estimation-gaussian-mixture-models, llm-ch01-understanding-large-language-models]
created: 2026-06-04
updated: 2026-10-13
---

# Mixture Model

> **One-liner:** A mixture model builds a complicated bumpy probability curve by blending several simple ones together, each weighted by how much it contributes.

## 🟢 Intuition (Dummy)
Imagine you measure the heights of everyone in a school. If you lump kids and teachers together, the
"one hump" bell curve fits badly — there are really *two* groups with two different typical heights. A
mixture model says: instead of forcing one simple shape onto the data, take a few simple shapes (one per
group) and **blend** them. Each shape gets a weight saying "what fraction of the data comes from this
group." Add the weighted shapes up and you get a curve with several bumps — far more flexible than any
single simple curve, perfect for data that comes in clusters.

### In LLMs: a recipe of data sources
LLM training data is a mixture too. GPT-3 didn't read "the internet" as one big pile. It drew its
training text from five sources — a web crawl, a cleaner web set, two book collections, and
Wikipedia — and each source had its own share, like a smoothie recipe: 60% of this, 22% of that,
3% of the other (book Table 1.1). The recipe is a choice. The cleaner, high-quality sources got a
*bigger* share than their size alone would give them.

## 🟡 Mechanics (Practitioner)
A mixture model describes a density $p(x)$ as a **convex combination** of $K$ simple **base** (component) distributions:
$$ p(x) = \sum_{k=1}^{K} \pi_k\, p_k(x), \qquad 0 \le \pi_k \le 1,\quad \sum_{k=1}^{K}\pi_k = 1 .$$
- The $p_k$ come from a chosen family — Gaussians, Bernoullis, Gammas, etc.
- The $\pi_k$ are the **mixture weights** (a probability vector): how much each component contributes.

Because the weights sum to 1 and are nonnegative, $p(x)$ is itself a valid density. The model is **more
expressive** than its base family: it can represent **multimodal** data (several clusters), which a single
unimodal component cannot.

**Worked example (mini).** Two 1-D Gaussian components, $p(x)=0.7\,\mathcal N(x\mid 0,1)+0.3\,\mathcal N(x\mid 5,1)$,
produces a curve with a tall hump near $0$ and a shorter hump near $5$ — impossible with one Gaussian.

**ML usage.** Density estimation, clustering (soft cluster membership), generative modeling, anomaly
detection. The dominant instance is the [[gaussian-mixture-model]].

### In LLMs: the GPT-3 pretraining data mixture (Raschka §1.5)
The book does not use the words "mixture model" here. But Table 1.1 (book p. 11) is exactly a
mixture with **categorical** components: each training token comes from source $d$ with probability
$p_d$, and then is drawn from that source's text.

| Source | Tokens $N_d$ | Mixture weight $p_d$ | Share of raw tokens $N_d/499$ | Implied passes $e_d$ |
|---|---|---|---|---|
| CommonCrawl (filtered) | 410 B | 60% | ≈ 82% | ≈ 0.44 |
| WebText2 | 19 B | 22% | ≈ 3.8% | ≈ 3.5 |
| Books1 | 12 B | 8% | ≈ 2.4% | 2.0 |
| Books2 | 55 B | 8% | ≈ 11% | ≈ 0.44 |
| Wikipedia | 3 B | 3% | ≈ 0.6% | 3.0 |

(The last two columns are derived from the book's numbers; the book does not print them.)

As a formula:

$$p_{\text{train}}(\text{text}) = \sum_{d\in\mathcal{D}} p_d\; p_d^{\text{src}}(\text{text}),\qquad \sum_{d} p_d = 1$$

Symbol by symbol:
- $\mathcal{D}$ = the set of 5 sources. $d$ = one source.
- $p_d$ = the **mixture weight** = the share of *sampled training tokens* from source $d$ (the book's "proportion in training data").
- $p_d^{\text{src}}$ = "pick a passage uniformly from source $d$" (the component distribution).
- $\sum_d p_d = 1$: the weights form a probability vector. The book's column adds to $60+22+8+8+3 = 101\%$, "adjusted for rounding errors".

**The book's key numbers.** The subsets total $410+19+12+55+3 = 499$ billion tokens, but GPT-3 trained
on only $N_{\text{train}} = 300$ billion. So the training stream is a *sample* from the mixture. The
implied number of passes over source $d$ is

$$e_d = \frac{p_d\, N_{\text{train}}}{N_d}$$

($N_d$ = tokens in source $d$). Wikipedia: $0.03\times300/3 = 3.0$ passes; CommonCrawl:
$0.60\times300/410\approx0.44$ — less than half of it was ever seen. The weights are **not**
proportional to size: high-quality data is up-weighted, the big noisy crawl is down-weighted.

## 🔴 Theory (Expert)
- A mixture is a **finite convex combination** in the space of probability densities; the weight vector
  $\boldsymbol\pi=[\pi_1,\dots,\pi_K]^\top$ lives on the $(K-1)$-simplex.
- Mixtures admit a **[[latent-variable-model|latent-variable]]** representation: introduce a one-hot
  indicator $z$ with $p(z_k=1)=\pi_k$ and $p(x\mid z_k=1)=p_k(x)$; marginalizing $z$ recovers
  $p(x)=\sum_k \pi_k p_k(x)$. This is what makes the [[expectation-maximization|EM algorithm]] applicable.
- **Identifiability** is only up to permutation of the components (label switching), and mixtures of
  finitely many components are not always identifiable for all base families — a subtlety in inference.
- Finite mixtures are **universal approximators** of densities as $K\to\infty$ (e.g., GMMs approximate any
  smooth density arbitrarily well), the theoretical reason they are so widely used.

### In LLMs: why the mixture weights matter
- **Same math, different job.** In Ch. 11 of MML the mixture is a *model* we fit to data (weights
  $\pi_k$ learned by EM). In LLM pretraining the mixture is a *sampling design* chosen by people
  (weights $p_d$ fixed in advance). The formula $\sum_k \pi_k p_k$ is identical; who sets the weights
  differs. Not a contradiction — two uses of one construction.
- **The latent-variable view still applies.** Let $z\in\mathcal{D}$ be "which source this token came
  from", $p(z=d)=p_d$. The LLM never sees $z$; it learns the marginal
  $p_{\text{train}}=\sum_d p_d\,p_d^{\text{src}}$. Pretraining by [[maximum-likelihood-estimation]] /
  [[empirical-risk-minimization]] therefore fits the model to the *mixture*, so changing $p_d$
  changes what "the data distribution" means.
- **Up-weighting is repeated sampling.** $e_d>1$ means the model sees the same documents more than
  once (Wikipedia ≈ 3 times, WebText2 ≈ 3.5). Repetition raises the risk of memorizing those sources
  ([[overfitting]]) — the trade-off behind choosing the weights.
- **Open alternative.** GPT-3's data was not released. Dolma (Soldaini et al. 2024, three trillion
  tokens) is a public corpus with its own documented source mixture (book p. 11). See
  [[pretraining-datasets]].

## Connections
- Builds on: [[gaussian-distribution]], [[probability-distribution]], probability density, [[convex-combination]]
- Leads to: [[gaussian-mixture-model]], [[latent-variable-model]]
- Used by: [[gaussian-mixture-model]], LLM [[pretraining-datasets]] (data-source mixtures), clustering, [[kernel-density-estimation]] (a degenerate "one component per point" mixture)
- People: —

## Sources
- `raw/mml-book.pdf` §11 intro & §11.1 (Eqs. 11.1–11.2), printed pp. 348–349 (PDF pp. 354–355).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book pp. 10–11 (Table 1.1 GPT-3 data mixture; sidebar "GPT-3 dataset details": 499 B vs 300 B tokens, rounding to 100%, Dolma).
