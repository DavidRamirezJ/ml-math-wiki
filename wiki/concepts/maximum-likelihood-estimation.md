---
title: Maximum Likelihood Estimation
type: concept
tags: [machine-learning, estimation, probabilistic-modeling, optimization, llm]
difficulty: intermediate
sources: [mml-ch09-linear-regression, llm-ch01-understanding-large-language-models, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-06-04
updated: 2026-10-13
---

# Maximum Likelihood Estimation

> **One-liner:** Pick the parameter values that make the data you actually observed as unsurprising as possible.

## 🟢 Intuition (Dummy)
Imagine you have a dice-rolling machine with an unknown bias, and you watch it roll. Maximum
likelihood asks: "Of all the possible biases this machine could have, which one would make *exactly
the rolls I just saw* the most likely?" You then declare that bias to be your best guess.

For curve fitting it's the same idea. Each candidate curve assigns a probability to the data points
you measured (because the points are noisy, no curve makes them certain). Maximum likelihood (ML)
picks the curve under which your measured points are the most probable. For the usual bell-curve
(Gaussian) noise, this turns out to be exactly the curve that minimizes the total squared distance
to the points — least squares.

**In LLMs.** A language model like GPT plays the same game with words. For every spot in a
text, it hands out probabilities to all 50,257 possible next tokens. Maximum likelihood says:
tweak the model's knobs so the token that *actually came next* gets as much of that probability as
possible. Do this across billions of words and you have pretraining. The "loss" you watch going down
during training (the cross-entropy) is just maximum likelihood turned upside down, so it becomes
something to *minimize*.

## 🟡 Mechanics (Practitioner)
Given a parametrized likelihood $p(\mathcal D\mid\boldsymbol\theta)$ over a dataset $\mathcal D$, the
**ML estimate** is
$$ \boldsymbol\theta_{\mathrm{ML}} \in \arg\max_{\boldsymbol\theta}\; p(\mathcal D\mid\boldsymbol\theta). $$
Because data points are i.i.d., the likelihood factorizes, and we work with the **negative
log-likelihood** (NLL) — sums beat products numerically and differentiate cleanly:
$$ -\log p(Y\mid X,\boldsymbol\theta) = -\sum_{n=1}^N \log p(y_n\mid \mathbf x_n,\boldsymbol\theta). $$

**Applied to [[linear-regression]] with Gaussian noise.** Each factor is
$\log p(y_n\mid\mathbf x_n,\boldsymbol\theta) = -\tfrac{1}{2\sigma^2}(y_n-\mathbf x_n^\top\boldsymbol\theta)^2+\text{const}$,
so the NLL (dropping constants) is the **least-squares loss**:
$$ \mathcal L(\boldsymbol\theta) = \tfrac{1}{2\sigma^2}\lVert \mathbf y - \boldsymbol\Phi\boldsymbol\theta\rVert^2. $$
Take the gradient, set it to zero:
$$ \frac{d\mathcal L}{d\boldsymbol\theta} = \tfrac{1}{\sigma^2}\big(\boldsymbol\theta^\top\boldsymbol\Phi^\top\boldsymbol\Phi - \mathbf y^\top\boldsymbol\Phi\big) = \mathbf 0^\top \;\Longrightarrow\; \boxed{\;\boldsymbol\theta_{\mathrm{ML}} = (\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}\boldsymbol\Phi^\top\mathbf y\;} $$
These are the **normal equations**. No iterative optimization needed — there is a closed form.

**Geometric view (§9.4): an orthogonal projection.** In the no-feature 1-D case $y=x\theta$,
$$ \boldsymbol\Phi\boldsymbol\theta_{\mathrm{ML}} = \frac{\mathbf X\mathbf X^\top}{\mathbf X^\top\mathbf X}\,\mathbf y, $$
and $\tfrac{\mathbf X\mathbf X^\top}{\mathbf X^\top\mathbf X}$ is exactly the projection matrix onto the line
spanned by $\mathbf X$. Generally, $\boldsymbol\Phi\boldsymbol\theta_{\mathrm{ML}}$ is the
[[orthogonal-projection]] of the target vector $\mathbf y$ onto $\mathrm{col}(\boldsymbol\Phi)$, the
$K$-dimensional subspace of $\mathbb R^N$ spanned by the feature columns — the point in that subspace
*closest* to $\mathbf y$.

**Worked example.** Fit $y=\theta x$ to $(1,2),(2,5),(3,5)$: $\boldsymbol\Phi^\top\boldsymbol\Phi=14$,
$\boldsymbol\Phi^\top\mathbf y=27$, so $\theta_{\mathrm{ML}}=27/14\approx1.93$.

**ML usage.** ML is the default training principle across ML: it underlies cross-entropy loss
(Bernoulli/categorical likelihoods), softmax classifiers, and most neural-net objectives. For
linear-Gaussian models it has a closed form; for nonlinear models you do gradient descent on the NLL.

### In LLMs: next-token MLE = cross-entropy loss
*Build a Large Language Model (From Scratch)*, ch. 5, builds the loss step by step. The book states
the goal in words: "maximize the likelihood of the correct token" (book p. 136). Written out
(standard notation; the book gives code, not LaTeX):

$$\boldsymbol\theta_{\mathrm{ML}} \in \arg\max_{\boldsymbol\theta}\; \prod_{t=1}^{T} p_{\boldsymbol\theta}(x_{t+1}\mid x_1,\dots,x_t)
\quad\Longleftrightarrow\quad
\arg\min_{\boldsymbol\theta}\; \mathcal L(\boldsymbol\theta)= -\frac{1}{N}\sum_{n=1}^{N}\log p_{\boldsymbol\theta}(y_n\mid \text{context}_n)$$

Symbol by symbol:
- $\boldsymbol\theta$ = all model weights (e.g. ~124 M for GPT-2 small).
- $x_1,\dots,x_t$ = the tokens seen so far; $x_{t+1}$ = the true next token. The product $\prod$ is the
  chain-rule factorization of the whole text's probability.
- $p_{\boldsymbol\theta}(y_n\mid\text{context}_n)$ = the [[softmax]] probability the model puts on the
  correct target token $y_n$ at position $n$.
- $N$ = number of predicted positions (batch size × tokens per row).
- $\log$ turns the product into a sum; the minus sign turns "maximize" into "minimize". The $\tfrac1N$
  only rescales, so the maximizer doesn't change.

**The book's six steps (Figure 5.7, book pp. 136–139),** for 2 texts × 3 tokens with an untrained GPT-2:
1. logits `[[[0.1113, -0.1057, -0.3666, ...]]]`
2. softmax probabilities `[[[1.8849e-05, 1.5172e-05, 1.1687e-05, ...]]]`
3. probabilities of the target tokens: `[7.4541e-05, 3.1061e-05, 1.1563e-05]` and `[1.0337e-05, 5.6776e-05, 4.7559e-06]`
4. log-probabilities: `[-9.5042, -10.3796, -11.3677, -11.4798, -9.7764, -12.2561]`
5. average: $-64.7638/6 = -10.7940$
6. negate: **10.7940** — the **cross-entropy loss** (book p. 138).

```python
logits_flat  = logits.flatten(0, 1)        # [6, 50257]
targets_flat = targets.flatten()           # [6]
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)   # tensor(10.7940)
```
PyTorch's `cross_entropy` does softmax → pick target → log → mean → negate in one call (book p. 139).
Since the book says "maximize likelihood" and then minimizes "negative average log probability," the
two are the same optimization problem.

**Sanity numbers.** An untrained model gives each token about $1/50{,}257 \approx 0.00002$ probability
(book p. 135). A perfectly uniform guess would score $\ln 50{,}257 \approx 10.825$ (our arithmetic). The
book's untrained losses on "The Verdict" are 10.988 (train) and 10.981 (validation) (book p. 145).
[[perplexity]] $=e^{\mathcal L}$: $e^{10.7940}\approx 48{,}726$ ("unsure among ~48,725 tokens", book p. 140).

**Same principle, other heads.**
- *Classification fine-tuning* (ch. 6): the categorical likelihood over **2 classes** at the last
  token, $\ell = -\log\operatorname{softmax}(\mathbf z)_y$. Accuracy can't be differentiated, so the book uses
  cross-entropy "as a proxy" (book p. 193). Untrained losses: 2.453 / 2.583 / 2.322 (train/val/test,
  book p. 194).
- *Instruction fine-tuning* (ch. 7): the same next-token likelihood, but positions whose target is
  `-100` are left out, so the mean runs over $\mathcal I=\{i: y_i\neq-100\}$ only (book pp. 221–222).
  Toy check: logits `[[-1.0, 1.0], [-0.5, 1.5]]`, targets `[0, 1]` → loss 1.1269; adding a third target
  `-100` still gives 1.1269.

## 🔴 Theory (Expert)
- **Optimality conditions.** $\boldsymbol\Phi^\top\boldsymbol\Phi$ is the Hessian; it is symmetric
  positive definite when $\mathrm{rk}(\boldsymbol\Phi)=K$, so $\nabla=\mathbf 0$ is necessary *and*
  sufficient for the unique global minimum (strictly convex quadratic). The right-multiplication by
  $(\boldsymbol\Phi^\top\boldsymbol\Phi)^{-1}$ in the derivation is licensed by this positive-definiteness.
- **The likelihood is not a distribution over $\boldsymbol\theta$.** $p(Y\mid X,\boldsymbol\theta)$ is a
  *function* of $\boldsymbol\theta$ that need not integrate to 1 over $\boldsymbol\theta$ (it is a normalized
  density in $y$, not in $\boldsymbol\theta$). This is the conceptual gap that [[maximum-a-posteriori-estimation|MAP]]
  and [[bayesian-linear-regression]] close by adding a prior.
- **ML noise variance.** Differentiating the full log-likelihood w.r.t. $\sigma^2$ gives
  $\sigma^2_{\mathrm{ML}}=\tfrac1N\sum_n(y_n-\boldsymbol\phi^\top(\mathbf x_n)\boldsymbol\theta)^2$ (Eq. 9.22),
  the mean squared residual — a biased estimator.
- **Orthonormal features.** If feature functions are orthonormal then $\boldsymbol\Phi^\top\boldsymbol\Phi=\mathbf I$,
  the projection becomes $\boldsymbol\Phi\boldsymbol\Phi^\top\mathbf y=\sum_k \boldsymbol\phi_k\boldsymbol\phi_k^\top\mathbf y$ —
  a decoupled sum of projections onto each basis vector (the principle behind Fourier/wavelet
  regression). Non-orthogonal bases can be orthogonalized via Gram–Schmidt.
- **Failure mode.** ML maximizes fit to the *training* data only, so it overfits flexible models;
  see [[overfitting]]. It produces a single *point estimate* with no uncertainty.

### In LLMs
- **Self-supervision makes MLE possible at scale.** The labels are the data itself: target = input
  shifted by one token (book p. 133). So the likelihood can be written on trillions of unlabeled
  tokens without any human labeling (book pp. 6, 12).
- **Cross-entropy = NLL = KL up to a constant.** With a one-hot target $q$,
  $H(q,p)=-\sum_i q_i\log p_i=-\log p_y$. Averaged over the data this is the empirical cross-entropy
  $H(\hat p_{\text{data}},p_\theta)=H(\hat p_{\text{data}})+\mathrm{KL}(\hat p_{\text{data}}\Vert p_\theta)$. The
  entropy term doesn't depend on $\theta$, so MLE minimizes the KL divergence from the empirical data
  distribution to the model. The book only says cross-entropy "measures the difference between two
  probability distributions" (book p. 138); the KL identity is standard background.
- **Gaussian ↔ categorical.** In the MML treatment above, a Gaussian likelihood gives least squares and
  a closed form. An LLM's categorical (softmax) likelihood has no closed form, so it's optimized
  with [[gradient-descent]] variants (AdamW) through [[backpropagation]]. It's the same principle with
  a different noise model. (No contradiction between the books; they use different likelihoods.)
- **Why logs.** The book notes log-probabilities are "more manageable in mathematical optimization"
  (book p. 138). Concretely: a product of thousands of numbers around $10^{-5}$ underflows in floating
  point, while the sum of their logs is fine, and $\nabla\log$ splits per token.
- **Masked likelihoods.** `ignore_index=-100` means the model is fitted to a *conditional* likelihood
  over the kept positions only. Masking the instruction tokens as well (Exercise 7.2) changes the
  estimator to $\max_\theta \log p_\theta(\text{response}\mid\text{instruction})$. Shi et al. (2024) report
  that *not* masking the instructions can work better (book pp. 222–223).
- **Same failure mode.** MLE fits the training set as hard as it can. In the book's LoRA spam run (appendix E.4) this
  shows up as 100% training accuracy vs. 96.64% validation accuracy, a small overfitting gap (book p. 336).
  See [[overfitting]].

## Connections
- Builds on: [[linear-regression]], [[gaussian-distribution]], [[gradient]], [[system-of-linear-equations]], [[orthogonal-projection]]
- Leads to: [[maximum-a-posteriori-estimation]], [[regularized-least-squares]], [[bayesian-linear-regression]]
- Used by: least squares, logistic regression, neural-net training (cross-entropy), [[pretraining]] of LLMs, [[classification-fine-tuning]], [[instruction-fine-tuning]]
- LLM links: [[cross-entropy-loss]], [[next-token-prediction]], [[softmax]], [[log-probability]], [[perplexity]], [[ignore-index-loss-masking]], [[surrogate-loss]], [[gpt-architecture]]
- People: [[carl-friedrich-gauss]] (least squares)

## Sources
- `raw/mml-book.pdf` §9.2.1 (Eqs. 9.7–9.22) and §9.4 (Eqs. 9.65–9.71), printed pp. 293–298, 313–315 (PDF pp. 299–304, 319–321).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 5–7 and §1.6, book p. 12 (next-word prediction, self-supervised labels).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.2–5.1.3, book pp. 132–145 (maximize target-token likelihood; six-step cross-entropy = 10.7940; perplexity; train/val losses).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book pp. 190–194 (cross-entropy as a differentiable proxy for accuracy).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 216–223 (shifted targets, `ignore_index=-100`, instruction masking).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book p. 336 (100% train vs. 96.64% validation accuracy).
