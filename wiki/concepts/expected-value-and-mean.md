---
title: Expected Value and Mean
type: concept
tags: [probability, statistics, summary-statistics, llm]
difficulty: beginner
sources: [mml-ch06-probability-and-distributions, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch]
created: 2026-06-04
updated: 2026-10-13
---

# Expected Value and Mean

> **One-liner:** The expected value is the long-run average outcome you'd get if you repeated a random experiment forever.

## 🟢 Intuition (Dummy)
Roll a fair die many, many times and average the results. You won't land on any single face's value, but the average settles near 3.5. That balance point is the **expected value** — the "center of mass" of a distribution. If you slid a finger under a cardboard cutout of the probability bar chart, the expected value is where it balances. The **mean** is just the expected value of the variable itself; the **median** is the middle value, and the **mode** is the most common value.

**In LLMs** the mean shows up in three friendly places in Raschka's book. (1) **[[dropout|Dropout]]**
switches parts of the network off at random, then boosts the survivors so that *on average* nothing
changes — the expected value is kept the same. (2) **[[layer-normalization|Layer normalization]]** takes the
mean of each token's numbers and subtracts it, so every token is centred at zero. (3) When a judge LLM
grades 110 answers, the **average score** (50.32) is the one number used to compare models.

## 🟡 Mechanics (Practitioner)
**Expected value of a function** $g$ of a random variable $X\sim p(x)$ (Def 6.3):
$$ \mathbb E_X[g(x)]=\int_{\mathcal X} g(x)\,p(x)\,dx\ \text{(continuous)},\qquad \mathbb E_X[g(x)]=\sum_{x\in\mathcal X} g(x)\,p(x)\ \text{(discrete)}. $$

**Mean** = expected value of the identity (Def 6.4); for a vector-valued $X$ it is taken element-wise:
$$ \mathbb E_X[\mathbf x]=\big[\,\mathbb E_{X_1}[x_1],\dots,\mathbb E_{X_D}[x_D]\,\big]^\top\in\mathbb R^D. $$

**Linearity** (the property you use constantly): for $f=ag+bh$,
$$ \mathbb E_X[a\,g(x)+b\,h(x)]=a\,\mathbb E_X[g(x)]+b\,\mathbb E_X[h(x)]. $$

**Other "averages":** **median** = value where $F_X=0.5$ (robust to outliers, hard to generalize to $>1$ dim since there is no ordering of vectors); **mode** = peak(s) of $p(x)$ (a distribution can be multimodal).

**Empirical / sample mean** of data $\mathbf x_1,\dots,\mathbf x_N$ (Def 6.9):
$$ \bar{\mathbf x}=\frac1N\sum_{n=1}^N \mathbf x_n. $$

**Worked example.** Bernoulli with $P(X{=}1)=\mu$: $\mathbb E[x]=1\cdot\mu+0\cdot(1-\mu)=\mu$.

**ML usage:** the expectation operator is everywhere — risk/loss is an expected loss, the marginal likelihood is $\mathbb E_X[p(y\mid x)]$, gradients in training are estimated as sample means (Monte Carlo / SGD), and "the foundational concepts of probability can be derived from the expected value" (p. 187).

### In LLMs (Raschka, *Build a Large Language Model (From Scratch)*)

**1. Dropout keeps the expected value** (§3.5.2, pp. 78–80). With dropout rate $p$, each attention weight $a$ becomes
$$\tilde a = \frac{m}{1-p}\,a,\qquad m\sim\operatorname{Bernoulli}(1-p),\qquad \mathbb E[\tilde a] = \frac{a}{1-p}\,\mathbb E[m] = \frac{a}{1-p}(1-p) = a .$$
- $m$ = a random switch: 1 ("keep") with probability $1-p$, 0 ("drop") with probability $p$; $\mathbb E[m] = 1-p$ by the Bernoulli worked example above.
- $\tilde a$ = the weight after dropout; $\mathbb E[\tilde a]$ = its average over all random switch settings.

Book numbers: with $p = 0.5$ survivors are multiplied by $1/0.5 = 2$ (a 6×6 matrix of ones becomes 0s and 2s, p. 79), and the causal
weight 0.3800 becomes 0.7599 (p. 80). The book's reason: the scaling is "crucial to maintain the overall balance of the attention weights,
ensuring that the average influence of the attention mechanism remains consistent during both the training and inference phases" (pp. 79–80).
The expectation identity is our formalization of that sentence. Note that a single draw is far from the mean (one row became all zeros) —
only the *average* is preserved.

**2. The per-token mean in LayerNorm** (§4.2, pp. 99–104). For a row (one token's activations) $\mathbf x \in \mathbb R^n$:
$$\mu = \frac1n\sum_{i=1}^{n}x_i,\qquad \hat x_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}} .$$
- $\mu$ ("mu") = the empirical mean of the row, `x.mean(dim=-1, keepdim=True)`; $\sigma^2$ = its variance; $\varepsilon = 10^{-5}$ avoids division by zero.

Book numbers (pp. 100–101): the first row of layer outputs `[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000]` has mean
$(0.2260+0.3470+0.2216)/6 = 0.7946/6 \approx 0.1324$, exactly the printed `0.1324`; the second row's mean is `0.2170`.
After normalizing, the means print as `-5.9605e-08` and `1.9868e-08` — zero up to floating-point rounding (p. 102).
`dim=-1` takes the mean across each row's features; `keepdim=True` keeps shape (2, 1) so that `x - mean` broadcasts (p. 101).
See [[standardization-z-score]], [[bessels-correction-biased-variance]].

**3. Average judge score** (§7.8, pp. 246–247). An LLM judge (Llama 3 via Ollama) scores each test response from 0 to 100:
$$\bar s = \frac1N\sum_{i=1}^{N}s_i,\qquad s_i\in\{0,1,\dots,100\}.$$
- $s_i$ = the score for response $i$; $N$ = the number of responses that parsed as integers (110 of 110 here).

Result: **Average score 50.32** for the fine-tuned GPT-2 medium model (p. 246). For reference (same method, p. 247): Llama 3 8B base 58.51,
Llama 3 8B instruct 82.6. Because Ollama is not fully deterministic across operating systems, the book suggests repeating the evaluation
several times and **averaging the scores** (p. 247): $\bar{\bar s} = \frac1R\sum_{r=1}^{R}\bar s^{(r)}$, where $R$ = the number of runs and
$\bar s^{(r)}$ = the average from run $r$. See [[llm-as-a-judge]], [[llm-evaluation]].

Also in the same spirit: the attention [[context-vector]] $\mathbf z = \sum_j\alpha_j\mathbf v^{(j)}$ is a weighted mean, $\mathbb E_{J\sim\alpha}[\mathbf v^{(J)}]$ (see [[convex-combination]]), and training losses such as [[cross-entropy-loss]] are averages over tokens and batches.

## 🔴 Theory (Expert)
- $\mathbb E$ is a **linear operator** on the space of integrable functions of $X$ — this single fact drives the algebra of means and the mean of [[gaussian-distribution|Gaussian]]/exponential-family models.
- $\mathbb E[g(X)]$ via the distribution of $X$ (not of $g(X)$) is the **law of the unconscious statistician**.
- **Iterated expectation / tower rule** (Exercise 6.11): $\mathbb E_X[x]=\mathbb E_Y[\mathbb E_X[x\mid y]]$.
- Mean exists only if the integral converges absolutely — heavy-tailed laws (e.g. Cauchy) have *no* mean, where the median still describes a typical value.
- In high dimensions the mode is computationally hard to find (possibly exponentially many modes), and the median has no canonical definition (no total order on $\mathbb R^D$).
- **Sample mean as an estimator.** For i.i.d. scores with mean $\mu_s$ and variance $\sigma_s^2$, $\mathbb E[\bar s] = \mu_s$ (unbiased, by linearity) and $\operatorname{Var}(\bar s) = \sigma_s^2/N$. Averaging $R$ independent evaluation runs divides the run-to-run noise variance by $R$ — the math behind the book's "repeat and average" advice (p. 247). The judge's 50.32 is itself a [[monte-carlo-estimation|Monte Carlo]] estimate of the expected judge score over the test distribution. (Our framing; the book gives the advice without the formula.)
- **Unbiased ≠ low variance.** Inverted dropout is an unbiased estimator of each weight, yet the output of one dropped-out forward pass can be far from the mean; that noise is exactly what makes dropout a [[regularization|regularizer]]. Also, unbiasedness holds for each weight but not for nonlinear functions of them: $\mathbb E[f(\tilde a)] \ne f(\mathbb E[\tilde a])$ in general (Jensen), so eval-mode outputs only approximate the training-time average.
- **Mean of a row vs. expectation.** LayerNorm's $\mu$ is a deterministic *empirical* mean over the $n$ features of one token, not an expectation over data; it uses the biased $1/n$ variance on purpose (GPT-2/TensorFlow compatibility, p. 103) — the $n$ vs $n-1$ difference is negligible for $n = 768$.

## Connections
- Builds on: [[random-variable]], [[probability-distribution]]
- Leads to: [[covariance-and-variance]], [[bayes-theorem]], [[standardization-z-score]]
- Used by: [[gaussian-distribution]], [[exponential-family]], [[empirical-risk-minimization]], [[monte-carlo-estimation]], [[dropout]], [[layer-normalization]], [[llm-as-a-judge]], [[convex-combination]]
- People: [[sebastian-raschka]], [[friedrich-wilhelm-bessel]]

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** The Ch. 5 loss is an average two times. First, the [[cross-entropy-loss]] of a batch is the mean of $-\log p$ over all its target tokens. In the toy example, six log-probabilities sum to −64.7638, so the mean is −10.7940 and the loss is 10.7940 (book pp. 137–139). Second, `calc_loss_loader` averages the batch losses: `total_loss / num_batches` (book p. 144). Both are sample means $\bar x = \frac{1}{N}\sum_{n=1}^{N} x_n$. Read it as "add the $N$ values $x_n$, then divide by $N$". They estimate the expected loss $\mathbb E[-\log p_\theta(y\mid\text{context})]$ over the text distribution ([[empirical-risk-minimization]]).

> **From [[llm-appA-introduction-to-pytorch]]:** In [[distributed-data-parallel]] (§A.9.3) each GPU gets a different, non-overlapping [[mini-batch]], so each model copy computes a different [[gradient]]. "These gradients are then averaged and synchronized" so that the copies do not diverge (Figure A.13, book p. 283). In symbols (the book says it in words): $g = \frac{1}{W}\sum_{r=0}^{W-1} g_r$. Read it as "add the $W$ per-GPU gradients $g_r$, then divide by the number of GPUs $W$". This is a sample mean. If the per-GPU losses are batch means of equal size, $g$ equals the gradient of the mean loss over the combined batch, because the gradient is linear (wiki annotation).

## Sources
- `raw/mml-book.pdf` §6.4.1 (Defs 6.3–6.4, 6.9), printed pp. 186–189, 191–192 (PDF pp. 192–195, 197–198).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5.2 (inverted-dropout scaling keeps the average influence), book pp. 78–80.
- Same PDF §4.2 (row mean and variance in layer normalization, `keepdim`, biased-variance sidebar), book pp. 99–104.
- Same PDF §7.8 (average LLM-judge score 50.32; Llama 3 reference scores; averaging repeated runs), book pp. 246–247.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.2, book pp. 137–139 (mean of log-probabilities); §5.1.3, book p. 144 (Listing 5.2, average over batches).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.9.3 (gradient averaging across GPUs, Figures A.12–A.13), book pp. 282–284 (PDF pp. 304–306).
