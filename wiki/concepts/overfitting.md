---
title: Overfitting
type: concept
tags: [machine-learning, model-selection, generalization, llm]
difficulty: beginner
sources: [mml-ch09-linear-regression, llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-06-04
updated: 2026-10-13
---

# Overfitting

> **One-liner:** A model overfits when it memorizes the noise in the training data instead of learning the real pattern, so it predicts badly on new data.

> This page started as the **regression-specific** treatment (MML Ch. 9) and now also covers how overfitting shows up when training LLMs (Raschka). For the general concept, including underfitting and the risk framing, see [[overfitting-and-underfitting]] (Ch. 8).

## 🟢 Intuition (Dummy)
Suppose you're given 10 dots and asked to draw a curve through them. A wiggly enough curve can pass
*through every single dot exactly* — perfect score on those 10. But the dots were noisy, so that
wiggly curve has bent itself to chase the random jitter. Show it a new dot and it guesses wildly,
because it learned the noise, not the trend.

That's overfitting: the model fits the training data "too well." A simpler curve that misses the dots
a little often predicts new points far better. The goal isn't to ace the practice questions — it's to
do well on the real exam (unseen data). That ability is called **generalization**.

### In LLMs: the model that recites the book
In the LLM book, the author trains a small GPT on one short story (*The Verdict*, about 5,000 tokens)
for 10 rounds. By the end, the model writes beautiful sentences — but they are copied **word for
word** from the story. It didn't learn "how English works" so much as "what this one story says".
That's overfitting in its purest form: memorizing instead of understanding. The fix the book points
to is simple: give it much more text (e.g. 60,000 books) and read each piece only once.

## 🟡 Mechanics (Practitioner)
In polynomial [[linear-regression]] the model flexibility is the **degree** $M$. As $M$ grows the model
can fit more wiggles. The diagnostic is to split data into a **training set** and a held-out **test set**
and compare error on each, using **root mean square error (RMSE)**:
$$ \mathrm{RMSE} = \sqrt{\tfrac1N\lVert\mathbf y-\boldsymbol\Phi\boldsymbol\theta\rVert^2} = \sqrt{\tfrac1N\sum_{n=1}^N\big(y_n-\boldsymbol\phi^\top(\mathbf x_n)\boldsymbol\theta\big)^2}. $$
RMSE is preferred over the raw squared error because it (a) is comparable across dataset sizes and
(b) carries the same units as $y$ (fit house prices in EUR → RMSE in EUR).

**The signature of overfitting** (Fig. 9.6): as $M$ increases,
- **training error monotonically decreases** (never goes up — more flexibility always fits training data at least as well),
- **test error first decreases, then increases** sharply.

The sweet spot is the $M$ minimizing *test* error (in the book's example, $M=4$). At $M=N-1$ the
polynomial passes through every training point exactly but oscillates wildly between them.

**Fixes.** Reduce flexibility (lower $M$ — model selection, §8.6), or keep flexibility but
**regularize**: add a penalty on parameter size, giving [[regularized-least-squares|ridge regression]] /
[[maximum-a-posteriori-estimation|MAP]], or go fully [[bayesian-linear-regression|Bayesian]] and report
uncertainty instead of a single overfit curve.

### In LLMs: spotting and fighting overfitting (Raschka)
**The diagnostic: two loss curves.** Split the data into training and validation parts
([[train-validation-test-split]]) and track the [[cross-entropy-loss]] on both
([[training-and-validation-loss]]). The book's working test: in §5.2 (p. 150) the losses "start to diverge past the second
epoch. This divergence and the fact that the validation loss is much larger than the training loss
indicate that the model is overfitting"; in §6.7 (p. 199) "little to no indication of overfitting"
means "there is no noticeable gap between the training and validation set losses." In symbols
(the book says it in words):

$$\text{gap}(e) = \mathcal{L}_{\text{val}}(e) - \mathcal{L}_{\text{train}}(e)$$

- $e$ = the [[epoch]] (one full pass over the training data).
- $\mathcal{L}_{\text{train}}(e)$, $\mathcal{L}_{\text{val}}(e)$ = average cross-entropy on the training / validation data after epoch $e$.
- Overfitting = $\mathcal{L}_{\text{train}}$ keeps falling while $\mathcal{L}_{\text{val}}$ flattens or rises, so the gap grows.

**Example 1 — pretraining on one short story (§5.2, pp. 149–151).** GPT-2 124M config, AdamW
(lr $=4\times10^{-4}$, weight decay $0.1$), 10 epochs:

| | Start (epoch 1) | End (epoch 10) |
|---|---|---|
| Training loss | 9.781 | **0.391** |
| Validation loss | 9.933 | **6.452** |

Past epoch 2 "the validation loss stagnates" around 6 while the training loss keeps dropping (Fig. 5.12,
p. 150). The generated text, e.g. "quite insensible to the irony", can be found **verbatim** in the
training file. Reason given: a "very, very small training dataset" and "multiple epochs" (p. 151). In
practice one trains on a much bigger dataset for **one epoch**; with 60,000 Project Gutenberg books
"this overfitting does not occur". In [[perplexity]] terms (derived): train $e^{0.391}\approx1.5$ vs.
validation $e^{6.452}\approx634$.

**Example 2 — the fancier training loop (App. D.4, p. 321).** With warmup, cosine decay and gradient
clipping, 15 epochs end at train **0.035** vs. val **6.938**: "the model begins to overfit after a
few epochs since it is a very small dataset, and we iterate over it multiple times." Better
optimization makes training loss lower; it does not fix too little data.

**Example 3 — classification fine-tuning (§6.7, pp. 198–200).** Spam classifier, 5 epochs. Losses
and accuracies for train and validation stay close: "little to no indication of overfitting".
Final accuracy: train **97.21%**, validation **97.32%**, test **95.67%** ([[classification-accuracy]]).
Validation is usually a bit above test because hyperparameters were tuned on the validation set.
Sidebar "Choosing the number of epochs" (p. 199): 5 is a good start; if it overfits after a few
epochs, use fewer; if validation loss could still improve, use more.

**Example 4 — LoRA fine-tuning (App. E.4, p. 336).** Train **100.00%**, validation **96.64%**, test
**98.00%** (printed output). Perfect training accuracy with lower held-out accuracy = "a small degree
of overfitting".
> ⚠️ The book is inconsistent here: the printed output says test 98.00%, the paragraph says 97.33%.

**Example 5 — instruction fine-tuning (§7.6, p. 232).** 2 epochs: training loss falls from **0.520** (end of epoch 1, step 115) to
**0.300** (step 230), while validation loss barely moves: **0.665 → 0.657** (it wobbles 0.67–0.685 in
between). A third epoch "is not essential and may even be
counterproductive as it could lead to increased overfitting."

**The book's toolbox against overfitting:**
- **[[dropout]]** — randomly drop units during training only (§3.5, p. 78): demo rate 0.5, GPT uses `drop_rate = 0.1` (§4.1, p. 95), placed after the attention and feed-forward sub-layers of each transformer block (§4.5, p. 115) and also on the input embeddings of the full model (§4.6). Turned off with `model.eval()` at inference (§5.4, p. 160; [[train-vs-eval-mode]]).
- **[[weight-decay]]** in [[adamw-optimizer]] — "penalizing larger weights" to reduce model complexity (§5.2, p. 148). Same idea as the $\ell_2$ penalty below.
- **More data, fewer epochs** — one epoch over a big corpus (§5.2, p. 151; §1.5 on huge [[pretraining-datasets]]).
- **Non-overlapping training windows** — in the [[sliding-window-sampling]] loader, stride = window length (4) so batches don't overlap, because "more overlap could lead to increased overfitting" (§2.6, p. 41).
- **[[instruction-masking]]** — compute the loss only on the response tokens so the model doesn't memorize the instructions; "can help reduce overfitting" (§7.3, p. 222).
- **Tuning regularization** — raise `drop_rate` or `weight_decay` to close a train/test gap (§6.7, p. 200).
- **Sampling-based decoding** — [[temperature-scaling]] and [[top-k-sampling]] reduce *verbatim regurgitation* at generation time (§5.3, p. 151). Note: this hides memorization in the output; it does not undo it in the weights.

```python
# The book's regularization knobs (GPT-2 124M setup)
GPT_CONFIG_124M = {..., "drop_rate": 0.1, ...}
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
model.eval()   # disables dropout for evaluation / generation
```

## 🔴 Theory (Expert)
- **Why magnitudes blow up.** Overfitting in linear regression is signalled by *large parameter
  magnitudes* (Bishop, 2006): to make a polynomial thread every noisy point, coefficients take huge
  positive/negative values that nearly cancel. This is precisely why an $\ell_2$ penalty on
  $\boldsymbol\theta$ (a Gaussian prior) suppresses it — see [[regularized-least-squares]].
- **Degrees of freedom & invertibility.** For $M<N$ the ML estimator is unique. At $M\ge N$ there are
  more parameters than data points: $\boldsymbol\Phi^\top\boldsymbol\Phi$ is singular (its null space is
  nontrivial), the [[system-of-linear-equations|normal equations]] are underdetermined, and there are
  infinitely many zero-training-error solutions. Brute-force model selection only needs to scan
  $0\le M\le N-1$.
- **Generalization gap.** Test error = training error + generalization gap; the gap grows with model
  capacity and shrinks with data. This is the bias–variance trade-off: low-$M$ models are biased
  (underfit), high-$M$ models have high variance (overfit). MAP/Bayesian methods trade a little bias
  for a large variance reduction.
- **Not eliminated by MAP.** A prior *pushes back* the overfitting boundary but does not remove it;
  the principled cure is to marginalize over parameters ([[bayesian-linear-regression]]) so uncertainty
  is reported rather than a single confident-but-wrong fit.

### In LLMs: overfitting over *time*, not just over *capacity*
- **Two axes of the same idea.** MML's polynomial example varies **capacity** (degree $M$) at fixed
  training and shows the U-shaped test error. The LLM book keeps the architecture fixed (124M
  parameters) and varies **training time** (epochs). Both are the same phenomenon — the learner's
  effective complexity outgrows what the data supports — seen along different axes. This is not a
  contradiction, but note that the book's diagnostic (Fig. 5.12 caption: "the training set loss continues to
  decrease past the second epoch, whereas the validation loss stagnates") is a looser test than
  MML's (test error rises). In Fig. 5.12 validation loss mostly *plateaus* around 6.2–6.5 rather
  than climbing.
- **Parameters ≫ tokens.** 124M parameters vs. about 4,600 training tokens (90% of the story's 5,145
  BPE tokens, §5.1 pp. 141–142): far more parameters than data points, the LLM version of MML's $M\ge N$ regime where
  zero training error is reachable. Train loss 0.035 (App. D) means probability
  $e^{-0.035}\approx0.97$ on each correct next token — essentially a lookup table of the story.
- **Memorization as a measurable event.** For language models, overfitting has a direct test: search
  generated text in the training set. Verbatim matches (the book's "quite insensible to the irony")
  show that the model's conditional distribution has collapsed onto the training sequences.
- **Data contamination breaks the test.** Exercise 5.5 (App. C, pp. 305–306): pretrained GPT-2 on
  *The Verdict* gives train 3.75 vs. val 3.56 — nearly equal. Either the story was not in GPT-2's
  pretraining data (so no overfitting to it), **or** it was, in which case *both* splits were seen and
  the gap tells you nothing. A held-out set only measures generalization if it is truly unseen.
- **One epoch as implicit regularization.** With a huge corpus and a single pass, every batch is new
  data, so training loss is itself an unbiased estimate of validation loss; the gap can't open. This
  is why large-scale pretraining rarely needs early stopping, and why the GPT-3 data mixture's
  repeated sources (Wikipedia ≈ 3 passes) are a deliberate risk ([[mixture-model]]).
- **Regularizers in the LLM recipe** map onto MML: weight decay ≈ $\ell_2$ penalty / Gaussian prior
  ([[regularized-least-squares]], with AdamW "decoupling" it from the gradient scaling); dropout ≈ an
  ensemble / noise-injection regularizer ([[regularization]]).

## Connections
- Builds on: [[linear-regression]], [[maximum-likelihood-estimation]]
- Leads to: [[regularized-least-squares]], [[maximum-a-posteriori-estimation]], [[bayesian-linear-regression]], [[regularization]], [[dropout]], [[weight-decay]]
- Used by: model selection, [[cross-validation]], every supervised learner; in LLMs: [[training-and-validation-loss]], [[train-validation-test-split]], [[llm-training-loop]], [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Related: [[overfitting-and-underfitting]], [[epoch]], [[adamw-optimizer]], [[instruction-masking]], [[sliding-window-sampling]], [[model-capacity]], [[perplexity]]
- People: [[nitish-srivastava]], [[geoffrey-hinton]] (dropout)

## Sources
- `raw/mml-book.pdf` §9.2.2 (Eq. 9.23, Figs. 9.5–9.6), printed pp. 298–300 (PDF pp. 304–306).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book pp. 11–12 (large diverse data; fine-tuning on smaller datasets).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book p. 41 (stride = window to avoid overlap and overfitting).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book p. 50; §3.5.2, book pp. 78–80; §3.6, book pp. 82–91 (dropout to reduce overfitting).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book p. 95; §4.5, book p. 115 (`drop_rate = 0.1`, dropout placement).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2, book pp. 148–151 (AdamW weight decay; losses 9.781→0.391 / 9.933→6.452; Fig. 5.12; verbatim memorization).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3, book p. 151 (generation strategies to reduce memorization); §5.4, book p. 160 (`model.eval()` disables dropout).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7, book pp. 198–200 (gap definition; epochs sidebar; 97.21 / 97.32 / 95.67%).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2, book pp. 207–211 (train/val/test partition); §7.3, book p. 222 (instruction masking); §7.6, book p. 232 (third epoch could increase overfitting).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` App. C, book pp. 305–306 (Exercise 5.5: train 3.75 vs val 3.56, contamination).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` App. D.4, book p. 321 (train 0.035 vs val 6.938).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` App. E.4, book p. 336 (100.00 / 96.64 / 98.00% vs text 97.33%).
