---
title: "LLM App. D — Adding Bells and Whistles to the Training Loop"
type: source
tags: [llm, training, optimization, learning-rate, source]
created: 2026-10-13
updated: 2026-10-13
---

# LLM App. D — Adding Bells and Whistles to the Training Loop

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix D, **book pp. 313–321** (PDF pages 335–343)
- **Type:** book appendix — *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025)
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13 (every code listing, number and printed output checked against the page text; Figures D.1–D.2 and book p. 319 checked from rendered PDF pages)
- *Page rule:* PDF page = book page + 22.
- **Related:** Appendix B has no reference group for Appendix D, but one Ch. 5 reference (book p. 294) points here. Appendix C has **no** solutions for Appendix D, because Appendix D has no exercises.

Chapter 5 trained the GPT model with a plain loop (`train_model_simple`): one fixed learning
rate, no safety net. This short appendix adds three small upgrades that real LLM training runs
use to stay stable. None of them changes the model; they only change *how* the weights are
updated. First the learning rate ramps up ([[learning-rate-warmup]]), then it glides down along
a half cosine ([[cosine-decay]]), and the size of every update is capped
([[gradient-clipping]]). Section D.4 puts all three into one new `train_model` function and
runs it.

## Section coverage (all of App. D accounted for)
| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Appendix intro (three techniques; reinitialise `GPT_CONFIG_124M`; load "The Verdict"; 90/10 data loaders) | 313–314 | ✅ [[llm-training-loop]], [[gpt-model-configuration]], [[gpt-2]], [[train-validation-test-split]], [[sliding-window-sampling]], [[pytorch-dataset-and-dataloader]], [[random-seed-reproducibility]], [[train-vs-eval-mode]] |
| D.1 | Learning rate warmup (Fig. D.1) | 314–316 | ✅ [[learning-rate-warmup]], [[learning-rate-schedule]], [[learning-rate]], [[hyperparameter]], [[adamw-optimizer]], [[weight-decay]], [[epoch]] |
| D.2 | Cosine decay (Fig. D.2) | 316–317 | ✅ [[cosine-decay]], [[learning-rate-schedule]], [[gradient-descent]], [[stochastic-gradient-descent]] |
| D.3 | Gradient clipping | 317–319 | ✅ [[gradient-clipping]], [[norm]], [[vanishing-and-exploding-gradients]], [[gradient]], [[backpropagation]], [[automatic-differentiation]], [[matrix]], [[vector]] |
| D.4 | The modified training function | 319–321 | ✅ [[llm-training-loop]], [[training-loop]], [[learning-rate-schedule]], [[gradient-clipping]], [[cross-entropy-loss]], [[training-and-validation-loss]], [[overfitting]] |
| App. B | References (none for App. D; one Ch. 5 entry points here) | 294 | ✅ see "References cited" below |
| App. C | Exercise solutions | — | — (Appendix D has no exercises) |

> ✅ **Every section of Appendix D (intro, D.1–D.4) maps to at least one wiki page.** None of
> the sections has numbered subsections (no D.1.1 etc.), and there is no separate summary section.
>
> **Coverage audit (2026-10-13):** all linked pages exist. Six shared pages owned by other units
> did not cite Appendix D, so they got a short "From llm-appD" note and a citation:
> [[hyperparameter]], [[random-seed-reproducibility]], [[pytorch-dataset-and-dataloader]],
> [[train-vs-eval-mode]], [[pretraining]], [[mini-batch]].

## TL;DR (5 bullets)
- **Same model, better loop:** the appendix keeps the 124M GPT model from Ch. 4–5 and only upgrades the [[training-loop]]. The three "bells and whistles" are [[learning-rate-warmup]], [[cosine-decay]] and [[gradient-clipping]].
- **Warmup (D.1):** start with a tiny [[learning-rate]] and raise it in a straight line to `peak_lr`. Early weights are random and early gradients are noisy, so small first steps avoid big, destabilising updates. Rule of thumb: warmup is **0.1%–20%** of all training steps.
- **Cosine decay (D.2):** after the peak, lower the learning rate along half a cosine wave down to a small `min_lr`. Smaller steps late in training lower the risk of overshooting a minimum of the loss. Warmup + decay together form a [[learning-rate-schedule]].
- **Gradient clipping (D.3):** if the L2 [[norm]] of the gradient is bigger than `max_norm`, scale the whole gradient down so its norm equals `max_norm`. The direction stays the same; only the length is capped. This protects against exploding gradients ([[vanishing-and-exploding-gradients]]).
- **All together (D.4):** the new `train_model` sets the learning rate at every step, clips gradients only after warmup, and returns the learning-rate history. On the tiny "The Verdict" dataset it still [[overfitting|overfits]] (train loss 0.035 vs validation loss 6.938), which is expected for such a small dataset.

## Key concepts introduced
**Setup (appendix intro)**
- [[gpt-model-configuration]] — `GPT_CONFIG_124M`, with context length shortened to 256
- [[train-validation-test-split]] — 90/10 character-level split of the story
- [[sliding-window-sampling]], [[pytorch-dataset-and-dataloader]] — `create_dataloader_v1` with `max_length = stride = 256` (no overlap)
- [[random-seed-reproducibility]] — `torch.manual_seed(123)` before building the model and loaders

**Learning-rate schedule (§D.1–D.2)**
- [[learning-rate]] — the step size $\eta$ in $\theta \leftarrow \theta - \eta \cdot (\text{update direction})$
- [[learning-rate-schedule]] — the learning rate becomes a function $\eta(t)$ of the global step $t$
- [[learning-rate-warmup]] — linear ramp from `initial_lr` to `peak_lr`
- [[cosine-decay]] — half-cosine fall from `peak_lr` to `min_lr` (the code calls it "cosine annealing")
- [[adamw-optimizer]], [[weight-decay]] — the optimiser, `AdamW(..., weight_decay=0.1)`; its learning rate is overwritten by hand through `optimizer.param_groups`
- [[hyperparameter]], [[epoch]] — `n_epochs`, `initial_lr`, `peak_lr`, `warmup_steps`, `min_lr`

**Gradient clipping (§D.3)**
- [[gradient-clipping]] — rescale gradients whose norm is above a threshold
- [[norm]] — L2 (Euclidean) norm of a vector; the same sum-of-squares rule on a matrix is the Frobenius norm
- [[vanishing-and-exploding-gradients]] — the failure mode that clipping prevents
- [[gradient]], [[partial-derivative]], [[backpropagation]], [[automatic-differentiation]] — `loss.backward()` fills each parameter's `.grad`

**Modified training function (§D.4)**
- [[llm-training-loop]], [[training-loop]] — `train_model` replaces Ch. 5's `train_model_simple`
- [[cross-entropy-loss]] — computed by `calc_loss_batch` (from Ch. 5)
- [[training-and-validation-loss]], [[overfitting]] — the train/validation gap at the end of the run
- [[gradient-descent]], [[stochastic-gradient-descent]], [[mini-batch]] — each `optimizer.step()` is one mini-batch update

## Notable results / numbers
- **Setup (book pp. 313–314):** `GPT_CONFIG_124M` = vocab 50,257, context length **256** (GPT-2 original: 1,024), embedding 768, 12 heads, 12 layers, dropout 0.1, `qkv_bias=False`. Batch size 2, split ratio 0.90.
- **Warmup hyperparameters (book p. 315):** `n_epochs = 15`, `initial_lr = 0.0001`, `peak_lr = 0.01`, `warmup_steps = int(0.2 * total_steps)` → prints **27**. This implies 9 batches per epoch and $T = 9 \times 15 = 135$ total steps (wiki-derived; the plots end near step 135).
- **Linear warmup formula** (from the code):
  $$\Delta = \frac{\eta_{\text{peak}} - \eta_{\text{init}}}{T_{\text{warm}}}, \qquad \eta(t) = \eta_{\text{init}} + t\,\Delta \quad (t < T_{\text{warm}})$$
  With the book's values, $\Delta = 0.0099/27 \approx 3.667\times10^{-4}$ (wiki arithmetic).
- **Cosine decay formula** (book pp. 316–317), for $t \ge T_{\text{warm}}$:
  $$p_t = \frac{t - T_{\text{warm}}}{T - T_{\text{warm}}}, \qquad \eta(t) = \eta_{\min} + (\eta_{\text{peak}} - \eta_{\min})\cdot\tfrac12\bigl(1 + \cos(\pi p_t)\bigr)$$
  At $p_t = 0$ this gives $\eta_{\text{peak}}$ (a smooth handover from warmup); as $p_t \to 1$ it goes to $\eta_{\min}$. In D.2, `min_lr = 0.1 * initial_lr` $= 10^{-5}$. Because the last step is $t = T-1$, $p_t$ stays below 1, so $\eta_{\min}$ is approached but never exactly reached (wiki-derived).
- **L2 norm (book p. 317):** $\|\mathbf v\|_2 = \sqrt{v_1^2 + \dots + v_n^2}$. **Clipping rule (book p. 318):** if $\|\mathbf G\|_2 > \text{max\_norm}$, then $\mathbf G' = \dfrac{\text{max\_norm}}{\|\mathbf G\|_2}\,\mathbf G$. In general form: $\mathbf g \leftarrow \mathbf g \cdot \min\bigl(1, \text{max\_norm}/\|\mathbf g\|_2\bigr)$.
- **Clipping demo (book pp. 318–319):** after one `loss.backward()`, the largest single gradient entry is `tensor(0.0411)`; after `clip_grad_norm_(model.parameters(), max_norm=1.0)` it is `tensor(0.0185)`. The ratio $\approx 0.45$ is the common scale factor, so the total gradient norm was about 2.2 (wiki-derived). PyTorch computes **one** norm over all parameters together and scales every gradient by the same factor.
- **`train_model` (book pp. 319–320):** defaults `initial_lr=3e-05`, `min_lr=1e-6`; `peak_lr` is read from `optimizer.param_groups[0]["lr"]`; clipping with `max_norm=1.0` only when `global_step >= warmup_steps`. Order per batch: zero grads → step counter → set lr → forward + loss → `backward()` → clip (after warmup) → `optimizer.step()`. Returns four lists: train losses, validation losses, tokens seen, learning rates.
- **Run (book pp. 320–321):** seed 123, `peak_lr = 0.001`, AdamW `weight_decay=0.1`, `n_epochs = 15`, `eval_freq = 5`, `eval_iter = 1`, `initial_lr = 1e-5`, `min_lr = 1e-5`, `start_context = "Every effort moves you"`. About **5 minutes** on a MacBook Air.
- **Losses (book p. 321):** first log line train 10.934 / val 10.939 (close to $\ln 50{,}257 \approx 10.82$, a near-uniform guess; wiki-derived) → last log line (Ep 15, Iter 130) train **0.035** / val **6.938**. The first samples are only commas; the last sample copies fluent text from the story.

> ⚠️ **Book inconsistency — 20 vs 27 warmup steps (book pp. 315–317):** the code prints
> `warmup_steps = 27`, and the plots in Figs. D.1 and D.2 peak at about step 27. But the prose,
> a code annotation, and both figure captions say "20 steps". Read "20" as a typo for 27.

> ⚠️ **Book erratum — clipping example (book p. 318):** for $\mathbf G = \begin{bmatrix}1&2\\3&4\end{bmatrix}$
> the book writes $\sqrt{1^2+2^2+2^2+4^2} = \sqrt{25} = 5$ and scales by $1/5$. The correct
> norm is $\sqrt{1+4+9+16} = \sqrt{30} \approx 5.477$, so the right factor is $1/\sqrt{30} \approx 0.183$
> and $\mathbf G' \approx \begin{bmatrix}0.183&0.365\\0.548&0.730\end{bmatrix}$. With the book's $1/5$
> the clipped norm is $\approx 1.095$, not 1. The method is correct; only the arithmetic has a typo.

> ⚠️ **Code detail (book p. 320):** the run sets `peak_lr = 0.001` but does not pass `lr=peak_lr`
> to `AdamW`. It still works because PyTorch's AdamW default learning rate is also `1e-3`.

## Exercises
Appendix D has **no numbered exercises**, and Appendix C has no solutions for it.

| # | Book p. | Task | Solution idea |
|---|---|---|---|
| (informal) | 321 | Train the model on a larger text dataset and compare `train_model` (warmup + cosine decay + clipping) with Ch. 5's `train_model_simple`. | No solution in the book. Use the same data and seed for both; compare [[training-and-validation-loss]] curves. On a larger dataset the gap from [[overfitting]] should shrink, and the schedule plus clipping should give a smoother loss curve. |

## References cited
Appendix B has no group for Appendix D. One entry in the Ch. 5 group refers to it:
1. Ibrahim et al. (2024), "Simple and Scalable Strategies to Continually Pre-train Large Language Models", arXiv 2403.08763 — shows that linear warmup and cosine annealing (as in Appendix D) also work for **continued** [[pretraining]] of already pretrained LLMs (book p. 294).

Background, **not** cited in the book: AdamW comes from Loshchilov & Hutter, "Decoupled Weight
Decay Regularization"; cosine annealing is usually credited to Loshchilov & Hutter, "SGDR:
Stochastic Gradient Descent with Warm Restarts" (2017).

## People
- No person is named in the body of Appendix D.
- [[sebastian-raschka]] — the author; his repo `rasbt/LLMs-from-scratch` hosts `the-verdict.txt` (book p. 314).

## What this changes in the wiki
- **New training-practice pages:** [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]] and [[learning-rate-schedule]] are owned by this unit. They turn the constant step size of [[gradient-descent]] into a time-varying step size $\eta_t$.
- **Updates an MML page:** [[norm]] gets a new "Used by: gradient clipping" connection, and a note that the sum-of-squares rule on a matrix is the Frobenius norm (the same as the L2 norm of the flattened matrix). This links the MML Ch. 3 geometry pages to a real training trick.
- **Strengthens existing links:** [[stochastic-gradient-descent]] (decreasing step sizes help SGD settle near a minimum), [[vanishing-and-exploding-gradients]] (clipping is the standard fix for exploding gradients), [[adamw-optimizer]] and [[weight-decay]] (the optimiser whose learning rate is set by hand), and [[llm-training-loop]] (the full upgraded loop).
- **Caveat added to [[gradient-clipping]]:** with plain SGD, clipping caps the step length at $\eta \cdot \text{max\_norm}$. With AdamW, the per-parameter normalisation means the step is not capped in the same direct way. The book does not discuss this.
- **No contradictions** with existing wiki pages. Two book-internal problems (20 vs 27 warmup steps; the $\sqrt{25}$ vs $\sqrt{30}$ clipping example) are flagged above.
