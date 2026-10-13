---
title: Empirical Risk Minimization
type: concept
tags: [machine-learning, optimization, supervised-learning]
difficulty: intermediate
sources: [mml-ch08-when-models-meet-data, llm-ch01-understanding-large-language-models, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-06-04
updated: 2026-10-13
---

# Empirical Risk Minimization

> **One-liner:** Pick a family of guess-functions, score how badly each one does on your examples, and keep the one with the smallest average mistake.

## 🟢 Intuition (Dummy)
Imagine you want a rule that predicts someone's salary from their age. You don't know the "true"
rule, so you try out a whole family of candidate rules (say, all straight lines). For each
candidate you look at your real examples and ask: *on average, how far off were you?* The line
that's wrong by the least, on average, across all your examples — that's the one you keep.

That's the whole idea: **average mistake on the data you have, made as small as possible.** The
catch (covered by [[regularization]] and [[cross-validation]]) is that being perfect on the
examples you've *seen* doesn't guarantee being good on examples you *haven't* seen.

**In LLMs.** Raschka's spam filter is ERM in plain words: show the model emails labelled spam / not
spam and let it learn "by minimizing the error in its predictions on a training dataset" (book p. 3).
An LLM does the same thing at a giant scale. Its "examples" are every position in billions of words
of text, and its "mistake" is how surprised it was by the real next word. [[pretraining]] and both
kinds of [[fine-tuning]] in the book are the same recipe: average the mistake over a dataset, then
push it down.

## 🟡 Mechanics (Practitioner)
Supervised setup: examples $\mathbf x_n\in\mathbb R^D$ with labels $y_n\in\mathbb R$, $n=1,\dots,N$.
ERM is built from **four design choices** (MML §8.2):

1. **Hypothesis class** — the set of functions the predictor may be. The book uses
   [[linear-mapping|affine functions]]: prepend a constant feature $x^{(0)}=1$ so
   $\mathbf x_n=[1,x^{(1)}_n,\dots,x^{(D)}_n]^\top$ and write
   $$ f(\mathbf x_n,\boldsymbol\theta)=\boldsymbol\theta^\top\mathbf x_n = \theta_0+\sum_{d=1}^{D}\theta_d x^{(d)}_n .$$
2. **[[loss-function]]** — $\ell(y_n,\hat y_n)\ge 0$ scoring a single prediction $\hat y_n=f(\mathbf x_n,\boldsymbol\theta)$.
3. **[[regularization]]** — how to keep the fit from memorizing the training set.
4. **Optimization / search** — how to actually find the minimizer (gradients, Ch. 7).

The object minimized is the **empirical risk** (average training loss, Eq. 8.6):
$$ R_{\text{emp}}(f,X,\mathbf y)=\frac1N\sum_{n=1}^{N}\ell\big(y_n,\,f(\mathbf x_n,\boldsymbol\theta)\big). $$
Minimizing it is **empirical risk minimization.** Using the i.i.d. assumption, $R_{\text{emp}}$ is
a sample estimate of the real target, the **expected (population) risk**
$$ R_{\text{true}}(f)=\mathbb E_{\mathbf x,y}\big[\ell(y,f(\mathbf x))\big], $$
which we can never compute (it needs infinite data) but care about because it measures performance
on *unseen* data.

**Worked example — ordinary least squares (Examples 8.1–8.2).** Squared loss
$\ell(y_n,\hat y_n)=(y_n-\hat y_n)^2$ with the linear predictor gives
$$ \min_{\boldsymbol\theta\in\mathbb R^D}\ \frac1N\sum_{n=1}^N (y_n-\boldsymbol\theta^\top\mathbf x_n)^2
\;=\; \min_{\boldsymbol\theta}\ \frac1N\,\lVert \mathbf y - X\boldsymbol\theta\rVert^2 , $$
with example matrix $X=[\mathbf x_1,\dots,\mathbf x_N]^\top\in\mathbb R^{N\times D}$ and label vector
$\mathbf y\in\mathbb R^N$. This **least-squares problem** has a closed-form solution via the normal
equations (Ch. 9).

**In LLMs (Raschka, *Build a Large Language Model*).** The book never uses the phrase "empirical
risk", but every training run in it is ERM with the [[cross-entropy-loss]] as $\ell$:

| Stage | "Example" $n$ | Label $y_n$ | Empirical risk the code computes |
|---|---|---|---|
| [[pretraining]] (Ch. 5) | one token position in a 256-token window | the next token (one of 50,257) | mean over all $B\times T$ positions |
| [[classification-fine-tuning]] (Ch. 6) | one SMS message (last-token logits) | spam = 1 / not spam = 0 | mean over the batch |
| [[instruction-fine-tuning]] (Ch. 7) | one non-padding token of an instruction + response | the next token | mean over non-masked positions |

Written out for pretraining:
$$ R_{\text{emp}}(\theta)=-\frac{1}{N}\sum_{n=1}^{N}\log p_\theta\big(y_n\mid \text{context}_n\big). $$
Symbol by symbol: $\theta$ = all [[model-parameters]]; $N$ = number of predicted token positions;
$y_n$ = the true next token at position $n$; $\text{context}_n$ = all tokens before it;
$p_\theta(y_n\mid\cdot)$ = the [[softmax]] probability the model gives that true token;
$-\log$ turns "high probability" into "small loss". This is exactly $\frac1N\sum_n\ell(y_n,\hat y_n)$
from Eq. 8.6 with $\ell=-\log p$.

**The book's numbers.** On the tiny training text *The Verdict* (5,145 tokens, 90 % train / 10 %
validation), the untrained GPT-2-124M-sized model has empirical risks **10.988** (train) and **10.981**
(validation) (book p. 145). Ten epochs of AdamW drive the training risk to **0.391** while the
validation risk stalls at **6.452** (book p. 149), a textbook case of minimizing $R_{\text{emp}}$ while
$R_{\text{true}}$ (estimated by the validation set) stops improving.

**Batches are an estimate of an estimate.** `calc_loss_loader` (Listing 5.2) averages the per-batch
means, $\frac1K\sum_{k=1}^{K}\mathcal L_k$ with $K$ = `num_batches`; using `num_batches=5` is a
cheap Monte Carlo estimate of the full-dataset empirical risk (book pp. 144–145).
```python
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    ...
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
    return total_loss / num_batches
```

**Supervised workflow (Figure A.3).** Training stage: a learning algorithm fits a model on examples +
labels. Inference stage: the trained model predicts labels for new observations. The book adds that
**model evaluation** should sit between the two (book pp. 253–254), which is the practical answer to
"is low $R_{\text{emp}}$ enough?".

**ML usage.** ERM is the backbone of almost all "fit by minimizing a loss" training: linear/logistic
regression, SVMs (where it was popularized), and neural networks (the loss is just non-convex).

## 🔴 Theory (Expert)
- ERM is *not* "probability free": there is an unknown data-generating $p(\mathbf x,y)$, and the
  i.i.d. assumption makes $R_{\text{emp}}\to R_{\text{true}}$ as $N\to\infty$ (law of large numbers).
  Unlike classical statistics, ERM is **agnostic to $p(\mathbf x,y)$** — it never specifies a noise
  model for $y$, and labels may be non-deterministic (§8.2.5).
- The gap $R_{\text{true}}-R_{\text{emp}}$ is the subject of **statistical learning theory**
  (Vapnik): uniform bounds over the hypothesis class control generalization, with capacity measured
  by VC dimension / Rademacher complexity. Richer classes shrink $R_{\text{emp}}$ but loosen the
  bound — the formal face of [[overfitting-and-underfitting]].
- The **performance measure** of a task (accuracy, RMSE, a cost-sensitive metric) and the **loss**
  optimized often differ in practice, for reasons of differentiability and optimization efficiency.
- **ERM in LLMs: self-supervised labels and the MLE view.** In pretraining the labels come from the
  text itself (targets = inputs shifted by one), so ERM needs no human labelling: this is
  [[self-supervised-learning]] (book p. 6). With $\ell=-\log p_\theta$, minimizing the empirical
  risk is *identical* to [[maximum-likelihood-estimation]] of a categorical next-token model, so for
  LLMs the "ERM vs. MLE" distinction above collapses.
- **Surrogate risk.** In Ch. 6 the goal is classification accuracy, $\frac1N\sum_i\mathbb 1[\hat y_i=y_i]$,
  which is not differentiable. The book therefore minimizes cross-entropy "as a proxy" (book p. 193):
  ERM over a [[surrogate-loss]] instead of the 0–1 risk. Likewise in Ch. 7 a low loss does not by
  itself prove good answers, so response quality is scored separately (book p. 233). This is the
  performance-measure vs. loss gap stated above, seen in practice.
- **Data scale as variance reduction.** Pretraining corpora with billions of words (book p. 10) make
  $R_{\text{emp}}$ a much tighter estimate of $R_{\text{true}}$. The book's 5,145-token toy corpus is
  the opposite extreme: the model memorizes passages verbatim and the train/validation gap opens after
  about two epochs (book pp. 150–151). The book notes this overfitting does not occur when training on
  60,000 public-domain Project Gutenberg books (book p. 151). It adds that it is "common to train a model on a much larger dataset for only one epoch" (book p. 151), so each example is seen once and $R_{\text{emp}}$ on the next batch behaves like an unbiased estimate of $R_{\text{true}}$.
- **Tikhonov regularization** (the $\lambda\lVert\boldsymbol\theta\rVert^2$ penalty) recasts ERM as a
  well-posed inverse problem; the constrained variant is **Ivanov regularization** (§8.2.5).

## Connections
- Builds on: [[vector]], [[linear-mapping]], [[loss-function]]
- Leads to: [[regularization]], [[cross-validation]], [[overfitting-and-underfitting]]
- Used by: [[linear-regression]], [[support-vector-machine]], [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[llm-training-loop]]
- Related (LLM book): [[cross-entropy-loss]], [[surrogate-loss]], [[supervised-learning]], [[self-supervised-learning]], [[train-validation-test-split]], [[training-and-validation-loss]]
- Parallels: [[maximum-likelihood-estimation]] (its probabilistic twin)
- People: [[vladimir-vapnik]]

## Sources
- `raw/mml-book.pdf` §8.2 (Eqs. 8.3–8.13, Examples 8.1–8.2), printed pp. 258–264 (PDF pp. 264–270).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1 (spam filter "minimizing the error … on a training dataset"), book p. 3 (PDF p. 25); §1.3 (self-supervised labels), book p. 6 (PDF p. 28); §1.5 (billion-word datasets), book p. 10 (PDF p. 32); §1.7, book pp. 14–15 (PDF pp. 36–37).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3 (train/val losses, Listing 5.2), book pp. 140–145 (PDF pp. 162–167); §5.2 (training run, overfitting), book pp. 146–151 (PDF pp. 168–173).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6 (cross-entropy as a proxy for accuracy), book pp. 191–194 (PDF pp. 213–216).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2 (supervised instruction fine-tuning), book p. 208 (PDF p. 230); §7.7, book p. 233 (PDF p. 255).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.2 (Figure A.3 supervised workflow), book pp. 253–254 (PDF pp. 275–276).
