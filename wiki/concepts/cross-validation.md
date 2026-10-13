---
title: Cross-Validation
type: concept
tags: [machine-learning, generalization, model-selection, llm]
difficulty: beginner
sources: [mml-ch08-when-models-meet-data, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-06-04
updated: 2026-10-13
---

# Cross-Validation

> **One-liner:** Cross-validation estimates how well a model will do on new data by repeatedly hiding part of your data, training on the rest, and testing on the hidden part.

## 🟢 Intuition (Dummy)
You want to know if your study method will work on the *real* exam, but you only have a stack of
practice questions. So you hide a few, study the rest, then test yourself on the hidden ones — and
repeat, hiding a different few each time. Average your scores: that's a fair guess of your real-exam
performance, and you used *all* your questions for both studying and testing along the way.

**In LLMs.** Large language models are so expensive to train that nobody can afford to "hide a
different few and repeat" — you get one study session. So the LLM book uses the simplest version of
this idea: one fixed pile for practice, one fixed pile for check-ins, and (when needed) one sealed
pile for the final exam. See [[train-validation-test-split]].

## 🟡 Mechanics (Practitioner)
We care about the **expected risk** on unseen data, but we only have a finite dataset. Split it into
a **training set** (fit the model) and a held-out **validation / test set** (estimate
generalization). One split wastes data and gives a noisy estimate, so use
**$K$-fold cross-validation** (MML §8.2.4):

1. Partition the data into $K$ equal chunks.
2. For each $k=1,\dots,K$: use chunk $k$ as the validation set $\mathcal V^{(k)}$ and the other
   $K-1$ chunks as the training set $\mathcal R^{(k)}$; train predictor $f^{(k)}$ on $\mathcal R^{(k)}$
   and measure its risk on $\mathcal V^{(k)}$.
3. Average the $K$ risks:
$$ \mathbb E_{\mathcal V}\big[R(f,\mathcal V)\big]\ \approx\ \frac1K\sum_{k=1}^{K} R\big(f^{(k)},\mathcal V^{(k)}\big). $$

The risk $R$ is often RMSE for regression. Every example serves once for validation and $K-1$ times
for training.

**Worked example.** $N=100$ points, $K=5$: five rounds, each training on 80 and validating on 20;
average the five RMSEs. The approximation has two error sources: finite training data (so $f^{(k)}$
isn't optimal) and finite validation data (noisy risk estimate).

**Nested cross-validation (§8.6.1)** for **model selection**: an **inner** CV loop picks the best
model / [[regularization|hyperparameter]] using a validation set; an **outer** CV loop then estimates
the generalization error of that chosen model on a test set. Never tune on data you'll report on.

**ML usage.** The default way to choose hyperparameters (regularization strength, polynomial degree,
kernel) and to report honest generalization performance. It is **embarrassingly parallel** — the $K$
runs are independent.

### In LLMs (Raschka, *Build a Large Language Model (From Scratch)*)
The LLM book never runs $K$-fold CV. Every chapter uses a **single hold-out split** instead
(details and code in [[train-validation-test-split]]):

| Where | Split (train / val / test) | Sizes |
|---|---|---|
| Ch. 5 pretraining, §5.1.3, p. 142 (reused in App. D, p. 314) | 90 / 10 / — (character-level) | 9 train batches, 1 val batch of `[2, 256]` |
| Ch. 6 spam classifier, §6.2, p. 175 (reused in App. E, p. 324) | 70 / 10 / 20 (shuffled, `random_state=123`) | 1,045 / 149 / 300 |
| Ch. 7 instruction tuning, §7.2, p. 210 | 85 / 5 / 10 | 935 / 55 / 110 |

That is the $K=1$ "one split" case that MML calls wasteful — but it's the only affordable option when
one run costs a fortune (the book's Llama 2 7B estimate: 184,320 A100 GPU hours, about \$690,000;
§5.1.3, book p. 141).

The roles mirror **nested CV** exactly. Appendix A (§A.7, book p. 275): the validation set is used
"multiple times to tweak the model settings", while "we only want to use a test set precisely once to
avoid biasing the evaluation." Validation plays the inner loop (choosing [[hyperparameter|hyperparameters]]
such as [[epoch|number of epochs]], [[learning-rate]], `drop_rate`, `weight_decay`); the test set plays
the outer loop.

**Worked example (Ch. 6, §6.7, book p. 200).** Full-split accuracies of the fine-tuned spam
classifier: train **97.21%**, validation **97.32%**, test **95.67%**. The book explains why validation
usually beats test: "the model development often involves tuning hyperparameters to perform well on the
validation set, which might not generalize as effectively to the test set" — the same leakage that nested CV is
designed to prevent.

During training the book also monitors validation *loss* (not just accuracy) every few steps — see
[[training-and-validation-loss]].

## 🔴 Theory (Expert)
- CV estimates $R_{\text{true}}(f)=\mathbb E_{\mathbf x,y}[\ell(y,f(\mathbf x))]$; it is nearly
  unbiased for the risk of a model trained on $\frac{K-1}{K}N$ points, with variance shrinking in $K$
  (the standard error is $\sigma/\sqrt K$).
- **Leakage:** even *observing* test-set performance and re-training leaks information (Blum & Hardt
  2015); nested CV exists precisely to keep model selection from contaminating the generalization
  estimate. Leave-one-out is the $K=N$ extreme (low bias, high variance, $N$ refits).
- Alternatives: bootstrap and jackknife (Efron & Tibshirani 1993). CV is the resampling cousin of the
  probabilistic [[bayesian-model-selection|marginal-likelihood]] approach to the same problem.

**In LLMs — why hold-out wins.**
- *Cost.* $K$-fold multiplies training compute by $K$. For pretraining, that's out of the question; even
  for fine-tuning, practitioners prefer one larger hold-out set.
- *Data size.* CV's main benefit is lower variance when $N$ is small. With web-scale corpora the
  hold-out estimate already has tiny variance. But the book's *fine-tuning* sets are small, so the
  variance is real: the Ch. 6 test accuracy on $n=300$ has standard error
  $\sqrt{0.9567\cdot0.0433/300}\approx 0.012$ (about 1.2 percentage points), and the 1.65-point val–test gap is
  within about one combined standard error (derived). Here, CV (or repeated splits) would give a more
  reliable comparison.
- *Dependence in text.* $K$-fold assumes exchangeable examples. Sliding-window chunks of one text
  (Ch. 5) are strongly dependent, so random folds would leak overlapping context; a contiguous
  hold-out split avoids this (blocked CV is the time-series analogue). *(Interpretation.)*
- *Contamination.* No split helps if the "held-out" data was in the pretraining corpus. Appendix C
  (Exercise 5.5, book pp. 305–306): pretrained GPT-2 scores train loss 3.7547 vs validation 3.5596 on
  "The Verdict", and if the story was in GPT-2's training data, "we can't tell whether the model is
  overfitting the training data because the validation set would have been used for training as well".

> 💡 **Not a contradiction, a different regime.** MML (§8.2.4) recommends $K$-fold because a single
> split wastes data and gives a noisy estimate; the LLM book uses a single split throughout. Both agree on the principle
> (estimate risk on data the model never trained on; never tune on the data you report). They differ
> only because LLM training cost makes $K$ repeated runs impractical.

## Connections
- Builds on: [[empirical-risk-minimization]], [[overfitting-and-underfitting]]
- Leads to: [[bayesian-model-selection]], hyperparameter tuning, [[train-validation-test-split]] (the single-split special case used for LLMs)
- Used by: [[regularization]] (choosing $\lambda$), [[linear-regression]], every supervised method, [[classification-fine-tuning]], [[instruction-fine-tuning]]
- Related: [[training-and-validation-loss]], [[overfitting]], [[hyperparameter]], [[epoch]], [[weight-decay]], [[llm-evaluation]]
- People: [[sebastian-raschka]]

## Sources
- `raw/mml-book.pdf` §8.2.4 (Eq. 8.13, Fig. 8.4) & §8.6.1 (Eq. 8.39, Fig. 8.13), printed pp. 263–264, 284 (PDF pp. 269–270, 290).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3, book pp. 141–145 (90/10 split, Llama 2 cost p. 141); §6.2, book p. 175; §6.7, book p. 200; §7.2, book p. 210; §A.7, book p. 275; §D.1, book p. 314; Appendix C Exercise 5.5, book pp. 305–306.
