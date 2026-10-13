---
title: Covariance and Variance
type: concept
tags: [probability, statistics, summary-statistics, linear-algebra]
difficulty: intermediate
sources: [mml-ch06-probability-and-distributions, llm-ch04-implementing-a-gpt-model, llm-ch07-fine-tuning-to-follow-instructions, llm-ch03-coding-attention-mechanisms]
created: 2026-06-04
updated: 2026-10-13
---

# Covariance and Variance

> **One-liner:** Variance measures how spread out a random quantity is; covariance measures whether two quantities move together.

## 🟢 Intuition (Dummy)
**Variance** is "how jumpy is this number?" A thermostat-controlled room has low temperature variance; the weather outside has high variance. **Covariance** asks about *pairs*: when one thing goes up, does the other tend to go up too? Height and weight covary positively (taller people are usually heavier); hours of TV and exam scores covary negatively. **Correlation** is just covariance rescaled to land between $-1$ and $+1$ so you can compare across different units.

## 🟡 Mechanics (Practitioner)
**Covariance** of two univariate random variables (Def 6.5):
$$ \mathrm{Cov}[x,y]=\mathbb E\big[(x-\mathbb E[x])(y-\mathbb E[y])\big]=\mathbb E[xy]-\mathbb E[x]\,\mathbb E[y]. $$

**Variance** = self-covariance (Def 6.7); **standard deviation** $\sigma(x)=\sqrt{\mathbb V[x]}$. The **raw-score formula** (one-pass, Eq 6.44):
$$ \mathbb V[x]=\mathbb E[x^2]-(\mathbb E[x])^2\quad\text{("mean of the square minus square of the mean").} $$

**Covariance matrix** of a vector random variable $\mathbf x\in\mathbb R^D$ with mean $\boldsymbol\mu$:
$$ \mathbb V[\mathbf x]=\mathbb E[(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^\top]=\mathbb E[\mathbf x\mathbf x^\top]-\boldsymbol\mu\boldsymbol\mu^\top\in\mathbb R^{D\times D}. $$
It is **symmetric and positive semidefinite**; diagonal = per-coordinate variances, off-diagonal = cross-covariances.

**Correlation** (Def 6.8): $\;\mathrm{corr}[x,y]=\dfrac{\mathrm{Cov}[x,y]}{\sqrt{\mathbb V[x]\mathbb V[y]}}\in[-1,1].$

**Affine transformation** (Eqs 6.50–6.51): for $\mathbf y=A\mathbf x+\mathbf b$,
$$ \mathbb E[\mathbf y]=A\boldsymbol\mu+\mathbf b,\qquad \mathbb V[\mathbf y]=A\,\Sigma\,A^\top. $$

**Empirical covariance** (Def 6.9): $\;\Sigma=\frac1N\sum_{n=1}^N(\mathbf x_n-\bar{\mathbf x})(\mathbf x_n-\bar{\mathbf x})^\top$ (biased; unbiased uses $N-1$).

**Worked example.** $X$ with $\mathbb E[x]=0,\ \mathbb E[x^3]=0$, and $Y=x^2$: $\mathrm{Cov}[x,y]=\mathbb E[x^3]-\mathbb E[x]\mathbb E[x^2]=0$ even though $Y$ fully depends on $X$ — covariance only sees *linear* dependence.

**ML usage:** the covariance matrix is the shape parameter of the [[gaussian-distribution|Gaussian]], the object PCA diagonalizes (Ch.10), and the noise model in linear regression (Ch.9); correlation drives feature selection and whitening.

### In LLMs (Raschka, *Build a Large Language Model*)
Variance shows up inside every GPT block, in [[layer-normalization]]. For each token's feature vector $\mathbf{x}=(x_1,\dots,x_n)$, with $n$ = the embedding dimension (768 in GPT-2 124M), the book computes
$$\mu = \frac1n\sum_{i=1}^n x_i,\qquad \sigma^2 = \frac1n\sum_{i=1}^n (x_i-\mu)^2$$
- $\mu$ = the mean of that one token's features,
- $\sigma^2$ = their variance (average squared distance from $\mu$),
- $\frac1n$ = divide by $n$, the **biased** version, the same convention as MML's empirical covariance above (see [[bessels-correction-biased-variance]]).

Then it [[standardization-z-score|standardizes]]: $(x_i-\mu)/\sqrt{\sigma^2+\varepsilon}$ with $\varepsilon=10^{-5}$. Here the "random variable" is just the list of features of one vector, so these are *empirical* statistics, not expectations.

Worked numbers (book §4.2, pp. 101–104): a `Linear(5, 6)` + ReLU layer gives the row $[0.2260, 0.3470, 0, 0.2216, 0, 0]$. Its mean is 0.1324 and its variance is **0.0231** with PyTorch's default `out.var(dim=-1)`, which divides by $n-1=5$. With `unbiased=False` (divide by 6) it is about 0.0193. After LayerNorm the variance checks out as 0.9995, just under 1 because of $\varepsilon$.

```python
mean = x.mean(dim=-1, keepdim=True)
var  = x.var(dim=-1, keepdim=True, unbiased=False)   # biased, 1/n
```

A second appearance (§7.8, book pp. 246–247): the book averages 110 integer LLM-as-a-judge scores (0–100) into one mean, 50.32 (book p. 246). Ollama is "not entirely deterministic across operating systems", so the scores can vary slightly, so the book advises repeating the evaluation several times and averaging. Averaging $R$ independent runs divides the variance of the score by $R$ (see the Theory bullet below).

## 🔴 Theory (Expert)
- **Sums:** $\mathbb V[\mathbf x+\mathbf y]=\mathbb V[\mathbf x]+\mathbb V[\mathbf y]+\mathrm{Cov}[\mathbf x,\mathbf y]+\mathrm{Cov}[\mathbf y,\mathbf x]$; if uncorrelated this collapses to $\mathbb V[\mathbf x]+\mathbb V[\mathbf y]$ (a Pythagorean identity — see [[inner-product]] of random variables).
- **Three expressions for variance** (§6.4.3): standard $\mathbb E[(x-\mu)^2]$, raw-score (numerically unstable when the two large terms nearly cancel), and the pairwise form $\frac{1}{N^2}\sum_{i,j}(x_i-x_j)^2 = 2\big(\frac1N\sum x_i^2-(\frac1N\sum x_i)^2\big)$ — equating "spread about the mean" with "average pairwise distance".
- **Law of total variance:** $\mathbb V[x]=\mathbb E_Y[\mathbb V[x\mid y]]+\mathbb V_Y[\mathbb E[x\mid y]]$.
- Covariance defines an [[inner-product|inner product]] $\langle X,Y\rangle:=\mathrm{Cov}[x,y]$ on zero-mean random variables: symmetric, bilinear, positive definite. Then $\|X\|=\sigma(x)$ and $\cos\theta=\mathrm{corr}[x,y]$ — correlation is literally a cosine, "uncorrelated" means "orthogonal".
- **Variance of an average** (why repeated LLM evaluations help, book §7.8): if $\bar s^{(1)},\dots,\bar s^{(R)}$ are independent runs with variance $\sigma^2$, then $\mathbb V\big[\frac1R\sum_r \bar s^{(r)}\big]=\frac{1}{R^2}\sum_r\mathbb V[\bar s^{(r)}]=\sigma^2/R$, using the sum rule above with zero cross-covariances. The standard deviation shrinks like $1/\sqrt R$.
- **Library conventions** (not a contradiction): MML's empirical covariance and the book's `LayerNorm` both divide by $N$. PyTorch's `.var()` defaults to $N-1$ (Bessel's correction). For large $N$ (e.g. 768 features) the ratio $\frac{N-1}{N}\approx 0.9987$ is negligible; see [[bessels-correction-biased-variance]].
- MML assumes covariance matrices are positive *definite* for intuition; degenerate (low-rank, positive semidefinite) cases are skipped.

### From llm-ch03-coding-attention-mechanisms
In §3.4.1 the book explains why attention scores are divided by $\sqrt{d_k}$ (book p. 69): GPT-like embeddings often have more than 1,000 dimensions, so dot products get large and push [[softmax]] towards a step function with near-zero gradients. The book gives this reason in words only. Variance gives the exact reason (wiki addition, not printed in the book):
$$\operatorname{Var}(q \cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k \quad\Rightarrow\quad \operatorname{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1 .$$
Read it as: "the variance of q dot k is the sum, over the $d_k$ components, of the variance of each product $q_i k_i$; this equals $d_k$". This holds when all components are independent with mean 0 and variance 1. Dividing by $\sqrt{d_k}$ divides the variance by $d_k$, so the scaled score has variance 1 for any width. See [[scaled-dot-product-attention]].

## Connections
- Builds on: [[expected-value-and-mean]], [[matrix]], [[inner-product]]
- Leads to: [[gaussian-distribution]], [[statistical-independence]], [[standardization-z-score]], [[bessels-correction-biased-variance]]
- Used by: PCA / dimensionality reduction (Ch.10), linear regression noise (Ch.9), whitening, [[layer-normalization]], [[batch-normalization]], [[llm-as-a-judge]] (averaging repeated runs)
- People: [[friedrich-wilhelm-bessel]] (Bessel's correction)

## Sources
- `raw/mml-book.pdf` §6.4.1–6.4.4 (Defs 6.5–6.9, Eqs 6.44–6.51), printed pp. 189–194 (PDF pp. 195–200).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2, book pp. 99–104 (per-row mean/variance, `dim`/`keepdim`, biased variance sidebar p. 103, LayerNorm check p. 104).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, book p. 247 (repeat evaluations and average the scores for robustness).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4.1 sidebar "The rationale behind scaled-dot product attention", book p. 69 (the variance derivation is a wiki addition).
