---
title: Training and Validation Loss Curves
type: concept
tags: [machine-learning, evaluation, overfitting, training-dynamics, llm]
difficulty: beginner
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Training and Validation Loss Curves

> **One-liner:** Plot the model's error on the data it learns from and on data it never learns from, side by side — the shape of the two lines tells you whether it's learning, memorizing, or stuck.

*Also known as:* loss curves, learning curves (loss version).

## 🟢 Intuition (Dummy)
Think of two report cards for the same student, printed every week:

- **Training loss** = how badly they do on the homework they practice with every day.
- **Validation loss** = how badly they do on a surprise quiz with questions they've never seen.

Lower is better for both. Now look at the pattern:
- **Both going down together** → the student is really learning. 🎉
- **Homework score keeps improving, quiz score stalls or gets worse** → the student is *memorizing the homework*, not understanding. That's [[overfitting]].
- **Both stuck high** → the student isn't learning much at all (underfitting).

The book shows all three moods of this story, and the "memorizing" one shows up in the very first training run.

## 🟡 Mechanics (Practitioner)

### What's being measured
Both curves are the average [[cross-entropy-loss]] of next-token prediction (or class prediction), computed on a different split of the data (see [[train-validation-test-split]]). The book's `calc_loss_loader` averages per-batch losses (§5.1.3, book pp. 144–145):

$$\bar{\mathcal{L}} = \frac{1}{K}\sum_{k=1}^{K}\mathcal{L}_k$$

Symbol by symbol:
- $\bar{\mathcal L}$ = the number you plot (one point on the curve).
- $K$ = how many batches you average over — `num_batches`, which is `eval_iter` during training (e.g. 5) to keep it cheap.
- $\mathcal L_k$ = the mean cross-entropy of batch $k$, i.e. $-\frac{1}{T}\sum_t \log p_\theta(\text{correct token}_t)$ over the tokens in that batch.
- $\sum_{k=1}^{K}$ = "add up for $k = 1, 2, \ldots, K$".

### How the book records the curves — `evaluate_model` (§5.2, book pp. 147–148)
```python
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()                     # dropout off -> stable, reproducible
    with torch.no_grad():            # no gradients -> cheaper
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()                    # back to training mode
    return train_loss, val_loss
```
Inside `train_model_simple` it's called every `eval_freq` optimizer steps, and the losses are appended to `train_losses`, `val_losses` along with `tokens_seen`. Note: the *training* loss here is re-measured in eval mode on `eval_iter` batches — not the noisy per-batch loss used for the gradient.

The plot (`plot_losses`, book p. 150; Ch. 6 generalizes it as `plot_values`, book p. 198) puts **epochs** on the bottom axis and **tokens seen** (or examples seen) on a second top axis made with `ax1.twiny()`. The x-positions come from `torch.linspace(0, num_epochs, len(train_losses))`, which spreads the logged points evenly over fractional epochs.

### Case 1 — Pretraining on a tiny corpus: classic overfitting (Ch. 5)
GPT-2 124M config (context 256), AdamW `lr=0.0004`, `weight_decay=0.1`, 10 epochs, `eval_freq=5`, `eval_iter=5` (book p. 149):

| Epoch | Step | Train loss | Val loss |
|---|---|---|---|
| 1 | 0 | 9.781 | 9.933 |
| 1 | 5 | 8.111 | 8.339 |
| 2 | 10 | 6.661 | 7.048 |
| 2 | 15 | 5.961 | 6.616 |
| 9 | 80 | 0.541 | 6.393 |
| 10 | 85 | 0.391 | 6.452 |

Figure 5.12 (book p. 150): both drop sharply at first, but **after epoch 2** training keeps falling toward ~0.4 while validation **stagnates around 6**. The book's reading: the model is still learning but overfitting — expected, because the dataset is "very, very small" and we loop over it many times; normally you'd train on a much larger dataset for one [[epoch]] (book p. 151). The generated text agrees: the epoch-9 sample ("quite insensible to the irony") can be found verbatim in "The Verdict" (book p. 150). The book adds that training on 60,000 public-domain Project Gutenberg books makes this overfitting disappear (book p. 151).

To feel the gap, convert to [[perplexity]] $=e^{\mathcal L}$: train $e^{0.391}\approx 1.5$ (almost certain of each next token), validation $e^{6.452}\approx 634$ (still choosing among hundreds). *(Derived, not in the book.)*

Appendix D's fancier loop (warmup + cosine decay + clipping, 15 epochs) shows the same shape even more strongly: final **train 0.035 vs val 6.938** (§D.4, book p. 321).

### Case 2 — Classification fine-tuning: healthy curves (Ch. 6)
Spam classifier, AdamW `lr=5e-5`, `weight_decay=0.1`, 5 epochs (book p. 197). Start: train 2.153 / val 2.392; end (step 600): **train 0.083 / val 0.074**. Figure 6.16 (book p. 198): both lines fall sharply in epoch 1 and level off; "little to no indication of overfitting" because there's no noticeable gap (book p. 199). The same picture appears for LoRA fine-tuning in Figure E.5 (§E.4, book p. 335): sharp drop, then the curves level off — the model is converging. *Look closer at the LoRA log, though* (AdamW `lr=8e-4`, book pp. 334–335): validation loss bottoms out at **0.045** (step 150) and then drifts up to **0.252** (step 600) while training loss stays near 0.02 — a mild overfitting signal the caption doesn't mention *(observation from the printed log)*.

### Case 3 — Instruction fine-tuning: a gap starts to open (Ch. 7)
GPT-2 medium (355M), AdamW `lr=0.00005`, `weight_decay=0.1`, 2 epochs (book p. 231). Initial (before fine-tuning): train **3.8259**, val **3.7619** (book p. 230). Log (book pp. 231–232): step 0: 2.637 / 2.626 → step 115: 0.520 / 0.665 → step 230: **0.300 / 0.657**. Figure 7.17 (book p. 233): sharp decrease, then stabilization; in epoch 2 "the losses continue to decrease but at a slower rate" (book pp. 232–233). Because the model already learned effectively within two epochs, the book says a third epoch "is not essential and may even be counterproductive as it could lead to increased overfitting" (book p. 232). Reading the log yourself: in epoch 2 the validation loss stays at about 0.66–0.69 while training loss keeps dropping (0.438 → 0.300), so the gap is starting to widen *(observation)*.

### Case 4 — Pretrained GPT-2 on "The Verdict" (Exercise 5.5, Appendix C, book pp. 305–306)
Train **3.7547**, val **3.5596** — "in the same ballpark". Validation being *slightly lower* is "unusual in deep learning" but "likely due to random noise since the dataset is relatively small". But if the story was in GPT-2's training data, "we can't tell whether the model is overfitting", because the validation text was trained on too (contamination).

### Cheat sheet

| Pattern | Diagnosis | What the book suggests |
|---|---|---|
| Both fall, small gap | learning, generalizing | keep going / stop when flat |
| Train ↓, val flat or ↑, gap widens | overfitting | fewer epochs, more data, more [[dropout]] or [[weight-decay]] |
| Both high and flat | underfitting | more capacity, more epochs, higher [[learning-rate]] |
| Val slightly < train | usually noise (small val set) | don't over-read it |

## 🔴 Theory (Expert)

**Two estimators of two different things.** Write $\hat R_{\text{train}}(\theta_t)$ and $\hat R_{\text{val}}(\theta_t)$ for the losses at step $t$. Training minimizes $\hat R_{\text{train}}$ ([[empirical-risk-minimization]]), so it is an optimistically biased estimate of the true risk $R(\theta_t)$; $\hat R_{\text{val}}$ is unbiased for $R(\theta_t)$ as long as the validation data played no role in choosing $\theta_t$. The **generalization gap** $\hat R_{\text{val}} - \hat R_{\text{train}}$ is therefore the observable proxy for overfitting. Ch. 5's gap of ≈ 6 nats per token is huge; Ch. 6's ≈ 0 is ideal.

**The uniform baseline.** For a vocabulary of $V$ tokens, a model that assigns $1/V$ to everything has loss $\ln V$. For GPT-2's $V = 50{,}257$, $\ln V \approx 10.82$. The book's untrained losses (10.988 / 10.981, §5.1, book p. 145; 10.934 / 10.939, §D.4) sit right at — slightly above — this baseline, as they should for random weights. For the 2-class spam head the baseline is $\ln 2 \approx 0.693$; the initial 2.153 is higher because the fresh head outputs large uncalibrated logits. *(Derived.)*

**Mean of batch means vs token mean.** $\bar{\mathcal L}=\frac1K\sum_k \mathcal L_k$ equals the true per-token average only if all batches have the same number of (non-ignored) tokens. With `drop_last=False` on validation and `ignore_index=-100` masking in Ch. 7, batches differ slightly, so the plotted value is a close approximation. With `eval_iter=5` it is also a subsample estimate — the curves have sampling noise, which is why step-to-step wiggles (e.g. Ch. 6 step 450 → 500: val 0.132 → 0.137) shouldn't be interpreted.

**Early stopping as implicit regularization.** Choosing the step $t^\star = \arg\min_t \hat R_{\text{val}}(\theta_t)$ (or, as the book's epochs sidebar does, choosing `num_epochs` by eye from the curve, §6.7, book p. 199) is a form of [[regularization]]: for gradient descent on a quadratic loss, stopping after $t$ steps with rate $\eta$ behaves like an $\ell_2$ penalty with strength roughly $1/(\eta t)$. The usual caveat applies: once you select $t^\star$ on validation data, the validation loss at $t^\star$ is itself biased low — report a test number.

**Epochs vs tokens on the x-axis.** The book plots both. For pretraining, *tokens seen* is the more meaningful axis (compute ∝ tokens), and with one-pass training on web-scale data the training loss is itself an almost unbiased estimate of risk (every batch is new), so the two curves nearly coincide. The gap in Ch. 5 exists only because the same 4,608 training tokens are revisited ten times.

**Loss isn't the goal.** §7.6 ends: "the most crucial aspect is its performance in terms of response quality and correctness" (book p. 233) — so instruction-tuned models are judged on held-out responses ([[llm-evaluation]], [[llm-as-a-judge]]), not on loss curves alone.

## Connections
- Builds on: [[cross-entropy-loss]], [[train-validation-test-split]], [[loss-function]], [[empirical-risk-minimization]]
- Leads to: [[overfitting]], [[overfitting-and-underfitting]], [[epoch]] (choosing how many), [[perplexity]], [[llm-evaluation]]
- Used by: [[llm-training-loop]], [[training-loop]], [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Related: [[train-vs-eval-mode]], [[dropout]], [[weight-decay]], [[adamw-optimizer]], [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3 (Listing 5.2), book pp. 144–145 — `calc_loss_loader`, initial losses.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 (Listing 5.3, Fig. 5.12), book pp. 146–151 — `evaluate_model`, training log, `plot_losses`, overfitting reading.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7 (Figs. 6.16–6.17, epochs sidebar), book pp. 195–200.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6 (Fig. 7.17), book pp. 229–233.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.4, book pp. 319–321; §E.4 (Fig. E.5), book pp. 334–335; Appendix C Exercise 5.5, book pp. 305–306.
