---
title: Probability Distribution
type: concept
tags: [probability, foundations, distributions, llm]
difficulty: beginner
sources: [mml-ch06-probability-and-distributions, llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-06-04
updated: 2026-10-13
---

# Probability Distribution

> **One-liner:** A probability distribution is the complete bookkeeping of how likely each possible value of a random variable is.

## 🟢 Intuition (Dummy)
If a [[random-variable|random variable]] is a machine that outputs numbers, the distribution is the *odds sheet* telling you how often each number shows up. For a die it's "1/6 for each face". For something continuous like a person's height, you can't list every value, so instead you describe a smooth curve where the **area** under a stretch of the curve is the chance of landing in that stretch. Total area (or total of all the odds) is always exactly 1 — something must happen.

### In LLMs: the odds sheet for the next word
Every time an LLM writes a word, it first builds an odds sheet over **every** token it knows — all
50,257 of them for GPT-2. In the book's 9-word toy example, "forward" gets about 57%, "toward" 36%,
"closer" 6%, "pizza" almost 0% (derived from the logits on p. 153). Then
it either picks the top one or rolls a weighted die. Writing text is just doing that over and over.
The same idea shows up inside the model too: attention decides "how much should this word listen to
each other word?" with a little odds sheet that adds up to 1.

## 🟡 Mechanics (Practitioner)
**Discrete — probability mass function (pmf).** $P(X=x)\ge 0$ and $\sum_{x\in T}P(X=x)=1$.

**Continuous — probability density function (pdf).** A function $f:\mathbb R^D\to\mathbb R$ is a pdf if (Def 6.1):
$$ f(\mathbf x)\ge 0\quad\text{and}\quad \int_{\mathbb R^D} f(\mathbf x)\,d\mathbf x = 1, $$
with probabilities recovered by integration: $P(a\le X\le b)=\int_a^b f(x)\,dx$. **A density can exceed 1** (only the integral must be 1).

**Cumulative distribution function (cdf)** (Def 6.2): $F_X(\mathbf x)=P(X_1\le x_1,\dots,X_D\le x_D)=\int_{-\infty}^{x_1}\!\!\cdots\!\int_{-\infty}^{x_D} f(\mathbf z)\,d\mathbf z$.

**Many variables at once:**
- **Joint** $p(x,y)=P(X{=}x, Y{=}y)$ — the full table.
- **Marginal** $p(x)=\sum_y p(x,y)$ (or $\int p(x,y)\,dy$) — sum/integrate out the others.
- **Conditional** $p(y\mid x)=\dfrac{p(x,y)}{p(x)}$ — restrict to a row/column and renormalize.

| | Point probability | Interval probability |
|---|---|---|
| Discrete | $P(X=x)$ (pmf) | n/a |
| Continuous | $p(x)$ (pdf) | $P(X\le x)$ (cdf) |

**Worked example (frequency table).** With counts $n_{ij}$, $N$ total: $P(X{=}x_i)=\frac{c_i}{N}$ (column sum / $N$), $P(Y{=}y_j\mid X{=}x_i)=\frac{n_{ij}}{c_i}$.

**ML usage:** discrete distributions model categorical features/labels (handwriting classes, dice-like choices); continuous densities model real-valued data and parameters; the joint/marginal/conditional triad is the substrate of all probabilistic modelling.

### In LLMs: discrete distributions everywhere (Raschka Ch. 1, 3, 4, 5)
Almost every distribution in an LLM is a **discrete** distribution (a pmf) over a finite set — the
vocabulary or the positions in a sentence — built with [[softmax]].

**1. Softmax turns scores into a distribution (§4.7, pp. 123–124).** The model outputs a
[[logits|logit]] vector $\mathbf{z}\in\mathbb{R}^V$ for the next position. Softmax makes it a pmf:

$$p_i=\operatorname{softmax}(\mathbf{z})_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}$$

- $z_i$ = raw score (logit) of vocabulary token $i$.
- $e^{z_i}$ = the exponential, always positive, so $p_i>0$.
- $\sum_{j=1}^V e^{z_j}$ = sum over all $V$ tokens ($V=50{,}257$); dividing by it makes $\sum_i p_i=1$.
- $p_i$ = probability that the next token is token $i$.

Book example (Fig. 4.17): the last row of logits `[-0.4929, …, 2.4812, …, -0.6093]` becomes
probabilities `[0.0001, …, 0.0200, …, 0.0001]`; the largest is at index 257 → "a". Picking the
largest every time is [[greedy-decoding]] ($\arg\max_i p_i = \arg\max_i z_i$, because softmax is
[[monotonic-function|monotonic]]).

**2. Attention weights are distributions over positions (§3.3, pp. 59–63).** For the query "journey",
the dot-product scores `[0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]` become
$\alpha_2=$ `[0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]` (sum = 1). The book prefers softmax
over dividing by the sum because the weights are always positive and can be read "as probabilities
or relative importance" (p. 60). Each row of the 6×6 [[attention-weights]] matrix is its own pmf.

**3. Masking = zero probability + renormalize (§3.5, pp. 75–78).** In [[causal-attention]], future
tokens must get probability 0. Put $-\infty$ in their scores: $e^{-\infty}=0$, so softmax gives them
exactly 0 and the rest still sums to 1. Zeroing *after* softmax and renormalizing gives the same
answer — this is just the conditional distribution restricted to the allowed positions, exactly
$p(y\mid x)=p(x,y)/p(x)$ from above. Same trick later restricts sampling to the top-$k$ tokens.

**4. Training compares two distributions (§5.1, pp. 135–140).** At the start, the untrained model's
distribution is near uniform: about $1/7\approx0.14$ per token in a 7-token toy vocabulary, about
$1/50{,}257\approx0.00002$ for GPT-2 (p. 135). Training pushes up the probability of the correct next
token. The [[cross-entropy-loss]] "measures the difference between two probability distributions"
(p. 138):

$$\mathcal{L}=-\frac{1}{N}\sum_{n=1}^{N}\log p_\theta(y_n\mid\text{context}_n)$$

- $N$ = number of predicted positions (book: $2\times3=6$).
- $y_n$ = the correct (target) token at position $n$.
- $p_\theta(y_n\mid\text{context}_n)$ = the probability the model (weights $\theta$) gives to that target.
- $\log$ = natural log; the minus sign turns "maximize probability" into "minimize loss".

Book numbers: average log-probability $-10.7940$ → loss $10.7940$; [[perplexity]]
$e^{10.7940}\approx 48{,}726$ — "the model is unsure among about 48,725 tokens" (p. 140).

**5. Sampling from the distribution (§5.3, pp. 152–158).** Instead of argmax, draw the next token
with `torch.multinomial`, i.e. $x_{t+1}\sim\mathrm{Categorical}(p_1,\dots,p_V)$
([[multinomial-sampling]]). Toy vocabulary of 9 words, logits
`[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]`; 1,000 draws gave
**582× forward, 343× toward, 73× closer, 2× inches** (pp. 153–154) — the frequencies estimate the
probabilities ([[monte-carlo-estimation]]). [[temperature-scaling]] reshapes the distribution:

$$p_i(T)=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

- $T$ = temperature, any number $>0$. $T>1$ → flatter, closer to uniform. $T<1$ → sharper ("peaky").
  At $T=0.1$ "forward" is chosen almost 100% of the time; at $T=5$ even "pizza" gets about 4%.

**6. The data itself is a distribution (§1.5, p. 11).** GPT-3's training tokens are sampled from 5
sources with proportions 60/22/8/8/3% — a categorical distribution over sources (see
[[mixture-model]]).

## 🔴 Theory (Expert)
- pdf vs. law: the pdf $f$ is a non-negative function integrating to 1; the **law** of $X$ is the *association* of $X$ with $f$ via $P(a\le X\le b)=\int_a^b f$. Distinct concepts the book keeps separate (remark after Def 6.2).
- There exist cdfs with no corresponding pdf (e.g. mixed or singular distributions) — the cdf is the more fundamental object.
- The "uniform distribution" illustrates the discrete/continuous gap (Fig 6.3, Ex 6.3): discrete states need no ordering ($z_1=\text{red}$), continuous densities can have height $>1$ on a short support.
- Computing marginals = high-dimensional sums/integrals, which is generally **#P-hard** — the central computational bottleneck of probabilistic modelling.

### In LLMs: the model *is* a conditional distribution
- **Autoregressive factorization.** A GPT model defines a distribution over whole token sequences by
  the chain rule of probability (implied by the book, §4.7):
  $p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})$. Each factor is a categorical pmf
  over the [[vocabulary]], produced by softmax. Generation = sampling (or taking the mode of) each
  factor in turn ([[autoregressive-text-generation]]).
- **Cross-entropy, KL, and MLE.** With a one-hot target $q$, $H(q,p)=-\sum_i q_i\log p_i=-\log p_y$.
  Since $H(q,p)=H(q)+D_{\mathrm{KL}}(q\,\|\,p)$, minimizing the average cross-entropy is minimizing the
  KL divergence from the data distribution to the model's, which is [[maximum-likelihood-estimation]].
  A uniform model over $V$ tokens has loss $\log V=\log 50257\approx10.82$ and perplexity $V$; the
  untrained GPT's 10.79 is very close to that.
- **Softmax as the exponential family.** $p_i\propto e^{z_i}$ is the categorical distribution in
  natural parameters (the logits). Temperature $T$ rescales the natural parameters: $T\to0^+$ gives
  a point mass on the argmax, $T\to\infty$ gives the uniform distribution. Entropy rises
  monotonically with $T$. This is the Boltzmann/Gibbs form $p_i\propto e^{-E_i/T}$ with $E_i=-z_i$.
- **Masking is conditioning.** Setting scores to $-\infty$ and re-applying softmax equals restricting
  the pmf to an allowed subset $S$ and renormalizing: $p(i\mid i\in S)=p_i/\sum_{j\in S}p_j$. The book
  shows numerically that this leaks no information from the masked positions (p. 77).
- **Bernoulli noise in dropout.** [[dropout]] keeps each unit with a Bernoulli($1-p$) mask and scales
  survivors by $1/(1-p)$, so the *expected* value is unchanged:
  $\mathbb{E}[\tilde a]=a$ ([[expected-value-and-mean]]). Book demo: $p=0.5$ → survivors become 2
  (pp. 78–80).
- **Numerical side.** $e^{z}$ overflows for large logits; `torch.softmax` is implemented stably
  (subtract $\max_j z_j$ first — the book only warns that the naive version can overflow/underflow,
  p. 60). See [[numerical-stability]], [[log-probability]].

## Connections
- Builds on: [[random-variable]]
- Leads to: [[bayes-theorem]], [[expected-value-and-mean]], [[gaussian-distribution]], [[mixture-model]], [[maximum-likelihood-estimation]]
- Used by: [[exponential-family]], [[conjugate-prior]], density estimation (Ch.11); in LLMs: [[softmax]], [[attention-weights]], [[causal-attention-mask]], [[cross-entropy-loss]], [[perplexity]], [[next-token-prediction]], [[decoding-strategies]], [[temperature-scaling]], [[top-k-sampling]], [[multinomial-sampling]]
- People: —

## Sources
- `raw/mml-book.pdf` §6.2 (Defs 6.1–6.2, Table 6.1), printed pp. 178–183 (PDF pp. 184–189).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book pp. 10–11 (Table 1.1: data-source proportions).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book pp. 59–63 (softmax attention weights as probabilities; naive softmax overflow).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5, book pp. 75–80 ($-\infty$ masking → zero probability; renormalization; dropout scaling).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 122–125 (softmax over 50,257 tokens; argmax / greedy decoding; Fig. 4.17).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 131–140 (uniform start $1/V$; cross-entropy 10.7940; perplexity 48,725.8).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3, book pp. 151–158 (multinomial sampling 582/343/73/2 counts; temperature; top-k).
