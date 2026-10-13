---
title: Weight Decay
type: concept
tags: [regularization, optimization, deep-learning, llm]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Weight Decay

> **One-liner:** Weight decay shrinks every weight a tiny bit at each training step, so weights only stay large if the data really needs them to.

## 🟢 Intuition (Dummy)
Imagine a garden where every plant shrinks a little each night. Only the plants that get watered (the data keeps pushing them) grow big. The rest quietly shrink back toward nothing.

A neural network with huge weights can twist itself into weird shapes to fit every quirk of the training data — that's memorizing, or [[overfitting]]. Weight decay is that nightly shrink: a gentle, constant pull toward "small and simple." It doesn't stop learning; it just makes the model *justify* every big weight.

Raschka's one-sentence version: weight decay "aims to minimize model complexity and prevent overfitting by penalizing larger weights" (§5.2, book p. 148).

## 🟡 Mechanics (Practitioner)

### The rule
Each optimizer step, before (or alongside) the normal gradient update, multiply every weight by a number just under 1:

$$\theta \;\leftarrow\; (1 - \eta\lambda)\,\theta \;-\; \eta\,(\text{gradient step})$$

Symbol by symbol:
- $\theta$ = one weight of the model (the rule is applied to every weight separately).
- $\eta$ = the [[learning-rate]] (`lr`).
- $\lambda$ = the weight-decay strength (`weight_decay`), a [[hyperparameter]].
- $(1-\eta\lambda)$ = the "shrink factor" — slightly less than 1.
- "gradient step" = whatever the optimizer would do anyway: the raw gradient for SGD, or $\hat m_t / (\sqrt{\hat v_t}+\epsilon)$ for [[adamw-optimizer|AdamW]].

### In the book's code
It's just one argument to the optimizer — the book uses the same value everywhere:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
```

| Where | `lr` | `weight_decay` |
|---|---|---|
| §5.2 pretraining, p. 149 | 0.0004 | 0.1 |
| §5.4 checkpoint resume, p. 160 | 5e-4 | 0.1 |
| §6.7 spam classifier, p. 197 | 5e-5 | 0.1 |
| §7.6 instruction tuning, p. 231 | 0.00005 | 0.1 |
| §D.1 / §D.4 schedules, pp. 315, 320 | scheduled | 0.1 |
| §E.4 LoRA, p. 334 | 8e-4 | 0.1 |

### Worked numbers: how strong is it, really?
**Ch. 5 pretraining** ($\eta = 0.0004$, $\lambda = 0.1$):
- Shrink factor per step: $1 - 0.0004 \times 0.1 = 1 - 0.00004 = 0.99996$.
- The run has 9 batches × 10 epochs = 90 steps. If the gradient never pushed back, a weight would end at $0.99996^{90} \approx 0.9964$ of its starting value — a 0.36% shrink.
- "Half-life" (steps to shrink a weight to half on decay alone): $\ln 2 / (\eta\lambda) = 0.693 / 0.00004 \approx 17{,}300$ steps.

**Ch. 6 fine-tuning** ($\eta = 5\times10^{-5}$): shrink factor $0.999995$ per step; over 130 × 5 = 650 steps, $\approx 0.9968$.

So per run, weight decay is a very gentle force. It matters over long pretraining runs (hundreds of thousands of steps) — and it's always there as a brake.

### When to turn it up
At the end of Ch. 6 the classifier scores 97.32% on validation but 95.67% on test. The book's advice for closing such gaps: increase the dropout rate (`drop_rate`) or "the `weight_decay` parameter in the optimizer configuration" (§6.7, book p. 200) — both are [[regularization]] knobs. By the same logic, a widening gap between [[training-and-validation-loss|the loss curves]] is a signal to try a larger value *(interpretation, not stated in the book)*.

### Weight decay vs L2 penalty
The classic textbook way to "penalize large weights" is to add a term to the loss ([[regularized-least-squares]]):

$$\mathcal L_{\text{reg}}(\theta) = \mathcal L(\theta) + \frac{\lambda}{2}\lVert\theta\rVert_2^2$$

- $\mathcal L(\theta)$ = the normal loss (e.g. [[cross-entropy-loss]]).
- $\lVert\theta\rVert_2^2 = \sum_j \theta_j^2$ = the squared Euclidean [[norm]] of all weights.
- $\frac{\lambda}{2}$ = strength; the $\frac12$ just makes the derivative clean.

Its gradient is $\nabla\mathcal L + \lambda\theta$, so a plain SGD step becomes $\theta - \eta(\nabla\mathcal L + \lambda\theta) = (1-\eta\lambda)\theta - \eta\nabla\mathcal L$ — **exactly** weight decay. For SGD, the two are the same thing. For Adam they're **not**, and that difference is the whole reason AdamW exists (see Theory).

## 🔴 Theory (Expert)

**Coupled vs decoupled.** In Adam with an L2 penalty, the term $\lambda\theta$ is added to $g_t$ *before* the adaptive rescaling, so the effective decay on coordinate $j$ is $\eta\lambda\theta_j/(\sqrt{\hat v_{t,j}}+\epsilon)$. Coordinates with large historical gradients (large $\hat v$) are barely decayed; rarely-updated coordinates are decayed hard. The regularization strength becomes entangled with gradient statistics. **Decoupled weight decay** (Loshchilov & Hutter, ICLR 2019) applies $\theta \leftarrow \theta - \eta\lambda\theta$ separately, so every weight shrinks at the same relative rate $\eta\lambda$ regardless of its gradient history. This is what the book means by AdamW "improves the weight decay approach", allowing "more effective regularization and better generalization" (book p. 148).

**Bayesian reading.** An L2 penalty is the negative log of a zero-mean Gaussian prior: if $\theta \sim \mathcal N(\mathbf 0, \sigma^2 I)$, then $-\log p(\theta) = \frac{1}{2\sigma^2}\lVert\theta\rVert_2^2 + \text{const}$, so minimizing loss + penalty is [[maximum-a-posteriori-estimation]] with $\lambda \leftrightarrow 1/\sigma^2$ ([[gaussian-distribution]]). Decoupled decay in an adaptive method no longer corresponds exactly to any fixed prior — it's better seen as a shrinkage operator interleaved with a preconditioned gradient step.

**Equilibrium norm.** With decoupled decay, a weight vector drifts toward a steady state where shrinkage balances the update $\eta u$ ($u$ = the Adam step, coordinates $O(1)$). If successive updates keep pointing the same way, the balance $\eta\lambda\lVert\theta\rVert \approx \eta\lVert u\rVert$ gives $\lVert\theta\rVert \propto 1/\lambda$, independent of $\eta$. If updates are noisy and roughly orthogonal to $\theta$ (common late in training), the norm instead settles at $\lVert\theta\rVert \propto \sqrt{\eta/\lambda}$. Either way, $\lambda$ sets the weight scale. (The book simply keeps $\lambda = 0.1$ for every run and gives no reason.) For layers followed by normalization ([[layer-normalization]]), the function is invariant to the weight scale, and weight decay mainly controls the *effective* learning rate $\eta/\lVert\theta\rVert^2$ rather than model capacity. *(Background, not in the book.)*

**What to decay.** The book passes all of `model.parameters()` to AdamW, so biases, embeddings and LayerNorm scale/shift parameters are decayed too. Many LLM codebases put biases and normalization gains in a separate parameter group with `weight_decay=0`, since shrinking a LayerNorm gain toward 0 (rather than 1) is not a sensible prior. *(Background.)* In PyTorch the decay multiplies by `lr`, so a [[cosine-decay]] schedule also anneals the decay; with LoRA (§E.4) only the small $A$, $B$ matrices are trainable, so decay pulls the *update* $\Delta W = AB$ toward zero, i.e. toward the pretrained weights — a natural prior for fine-tuning ([[low-rank-adaptation]]).

**Relation to other regularizers.** Weight decay restricts the size of $\theta$; [[dropout]] injects noise into activations; early stopping (fewer [[epoch|epochs]]) limits how far $\theta$ travels from initialization. For linear least squares with gradient descent, early stopping after $t$ steps is approximately ridge with $\lambda \approx 1/(\eta t)$ — so the book's three levers (fewer epochs, more dropout, more weight decay) are closely related.

## Connections
- Builds on: [[regularization]], [[norm]], [[gradient-descent]], [[regularized-least-squares]], [[learning-rate]]
- Leads to: [[adamw-optimizer]], [[maximum-a-posteriori-estimation]]
- Used by: [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]], [[llm-training-loop]]
- Related: [[overfitting]], [[overfitting-and-underfitting]], [[dropout]], [[training-and-validation-loss]], [[epoch]], [[hyperparameter]], [[gaussian-distribution]], [[layer-normalization]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 (AdamW sidebar), book pp. 148–149.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.4, book p. 160.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7, book pp. 197, 200 — `weight_decay` as a lever against the val–test gap.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book p. 231; §D.4, book p. 320; §E.4, book p. 334.
- Decoupled formulation: Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization" — not cited by name in the book.
