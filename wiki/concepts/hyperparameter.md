---
title: Hyperparameter
type: concept
tags: [machine-learning, model-selection, optimization, training, llm-training]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Hyperparameter

> **One-liner:** A hyperparameter is a setting *you* choose before training (like the learning rate or batch size), as opposed to a parameter, which the model learns by itself.

## 🟢 Intuition (Dummy)
Think of baking bread. The dough "learns" its final shape in the oven — that's like the
model's **parameters** (its weights), which training adjusts automatically. But *you*
pick the oven temperature, the baking time, and the pan size before it goes in. Those
are the **hyperparameters**. Training can't pick them for you; you try a few settings,
look at how the bread (the loss) turns out, and adjust.

*Build a Large Language Model (From Scratch)* says it exactly this way: the learning rate
is "a tunable setting that we must experiment with based on observing the loss," and
"the number of epochs is another hyperparameter to choose" (§A.7, book p. 275). So the
everyday rule is: **parameters are learned, hyperparameters are chosen.**

## 🟡 Mechanics (Practitioner)
**Two kinds of numbers in every training run:**
| | Parameters $\theta$ | Hyperparameters $\lambda$ |
|---|---|---|
| Who sets them? | the optimizer, via [[backpropagation]] + [[gradient-descent]] | you, before (or around) training |
| Examples | weights and biases of every [[linear-layer]], [[embedding-layer]] rows | [[learning-rate]], batch size, number of [[epoch|epochs]], [[weight-decay]], dropout rate, model size |
| How many in GPT-2 small? | ~124 million ([[model-parameters]]) | a handful |
| Tuned on | the **training** set | the **validation** set |

**Hyperparameters the book actually sets** (a tour):
- **Optimization**: learning rate `lr` (0.5 for the toy MLP with SGD, §A.7; 0.0004 for
  GPT-2 pretraining, §5.2; 5e-5 for fine-tuning, §6.7/§7.6; 8e-4 for LoRA, §E.4), number
  of epochs (3 toy §A.7, 10 pretraining §5.2, 5 spam §6.7 and LoRA §E.4, 2 instruction §7.6), `weight_decay=0.1` for [[adamw-optimizer|AdamW]],
  schedule settings `peak_lr`, `initial_lr`, `min_lr`, `warmup_steps`, clipping
  `max_norm=1.0` (Appendix D).
- **Data**: **batch size** — "Small batch sizes require less memory during training but
  lead to more noisy model updates. Just like in regular deep learning, the batch size is
  a tradeoff and a hyperparameter to experiment with when training LLMs" (§2.6, book
  pp. 39–40); also `max_length` / [[context-length]] and `stride` of the
  [[sliding-window-sampling]] loader.
- **Architecture** ([[gpt-model-configuration]]): `vocab_size`, `context_length`,
  `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`. For GPT-2 small as first
  defined in §4.1 (book p. 95):
  ```python
  GPT_CONFIG_124M = {
      "vocab_size": 50257, "context_length": 1024, "emb_dim": 768,
      "n_heads": 12, "n_layers": 12, "drop_rate": 0.1, "qkv_bias": False
  }
  ```
  When loading the OpenAI weights (§7.5, Listing 7.7, book p. 227) the book instead uses
  `BASE_CONFIG` with `"drop_rate": 0.0` and `"qkv_bias": True`, and picks the size via
  `model_configs` (e.g. GPT-2 medium 355M: `emb_dim` 1024, `n_layers` 24, `n_heads` 16).
  The OpenAI checkpoint ships its settings in a file literally called `hparams.json` (§7.5, p. 227).
- **Fine-tuning method**: for [[low-rank-adaptation|LoRA]], the rank $r$ — "The inner
  dimension $r$ is a hyperparameter" (Fig. E.1, book p. 323) — and the scaling $\alpha$;
  the book suggests $r=16,\ \alpha=16$ as good defaults (§E.4, book p. 332); it is also common to raise $r$, and $\alpha$ is "usually chosen to be half,
  double, or equal to the rank" (book p. 332). See
  [[lora-rank-and-alpha]].

**How to tune them: the three-way split.** Use a third dataset, the **validation set**
(§A.7, book p. 275):
- **training set** → fit the parameters $\theta$;
- **validation set** → compare hyperparameter choices; you may use it many times;
- **test set** → use it **precisely once**, at the very end, "to avoid biasing the evaluation".

In symbols (a wiki formalization of that recipe):
$$\theta^{*}(\lambda) = \arg\min_{\theta}\ \mathcal L_{\text{train}}(\theta;\lambda),\qquad
\lambda^{*} = \arg\min_{\lambda\in\Lambda}\ \mathcal L_{\text{val}}\big(\theta^{*}(\lambda)\big)$$
Reading it symbol by symbol:
- $\lambda$ = one choice of hyperparameters (e.g. "lr = 5e-5, batch size 8, 5 epochs").
- $\Lambda$ = the set of choices you're willing to try (a grid, or random draws).
- $\theta^{*}(\lambda)$ = the weights you end up with after training *with that choice*.
- $\arg\min_\theta$ = "the $\theta$ that makes the following smallest".
- $\mathcal L_{\text{train}}$, $\mathcal L_{\text{val}}$ = the loss on the training / validation set.
- $\lambda^{*}$ = the winning choice: the one whose trained model does best on **validation**.

**What to adjust when results are off** (book's own advice):
- Instruction-tuned model scores poorly → adjust learning rate, batch size, or number of
  epochs; or use more data, other prompt formats, or a bigger model (§7.8, book p. 247).
- Gap between validation and test → increase `drop_rate` or `weight_decay`, both
  [[regularization]] hyperparameters (§6.7, book p. 200).
- Out of GPU memory → lower batch size 8 → 4/2/1 or `allowed_max_length` 1024 → 512/256
  (Exercise 7.3, book p. 233).

**Tiny example: why "watch the loss".** The toy MLP with `lr=0.5` for 3 epochs reaches
training loss 0.75 → 0.00 (§A.7, book p. 275). A much smaller lr would still be at a high
loss after 3 epochs (needs more epochs); a much larger one can overshoot and diverge.
The two hyperparameters (lr, epochs) trade off against each other.

## 🔴 Theory (Expert)
- **Bilevel optimization.** Hyperparameter tuning is an outer optimization over
  $\lambda$ wrapped around the inner training problem over $\theta$ (formula above).
  The outer objective is usually non-differentiable in $\lambda$ (discrete choices such
  as `n_layers`, or the noisy outcome of an SGD run), which is why grid/random search are
  used rather than gradient descent on $\lambda$.
- **Why validation accuracy beats test accuracy.** The book notes that validation
  accuracy (97.32%) is usually a bit higher than test (95.67%) because the settings were
  tuned to do well on the validation set (§6.7, book p. 200). Formally: if
  $\hat A_{\text{val}}(\lambda)$ is a noisy, unbiased estimate of true accuracy for each
  fixed $\lambda$, then $\mathbb E\big[\max_\lambda \hat A_{\text{val}}(\lambda)\big] \ge
  \max_\lambda \mathbb E\big[\hat A_{\text{val}}(\lambda)\big]$ (max of noisy estimates is
  optimistically biased). A held-out test set used once removes that selection bias —
  the reason for "use the test set precisely once" (book p. 275).
- **Regularization strengths are hyperparameters.** Weight decay $\lambda_{\text{wd}}$ and
  dropout rate $p$ control effective [[model-capacity]] and the bias–variance trade-off
  (see [[overfitting-and-underfitting]], [[regularization]]); they cannot be learned on
  the training set because minimizing training loss would always pick "no regularization".
- **With small data, use [[cross-validation]]** (average the validation score over $K$
  folds) to reduce the variance of $\hat A_{\text{val}}$; LLM work usually has enough data
  for a single split (the book uses 70/10/20 for spam, §6.2, book p. 175).
- **Coupled hyperparameters.** With mean-reduced losses, the SGD step is
  $\eta\cdot\frac1B\sum\nabla\ell_b$, so batch size and learning rate interact (larger
  batches → less gradient noise → often tolerate larger $\eta$); likewise LoRA's $\alpha$
  and $r$ enter only through the ratio $\alpha/r$ in the forward pass (§E.4, book p. 329).
- **Run-to-run variance.** Even the evaluation is noisy: the book suggests repeating the
  Ollama-judge scoring several times and averaging (§7.8, book p. 247). Hyperparameter
  comparisons smaller than this noise are not meaningful.

## Connections
- Builds on: [[model-parameters]], [[loss-function]], [[train-validation-test-split]], [[training-and-validation-loss]]
- Leads to: [[cross-validation]], [[learning-rate-schedule]], [[lora-rank-and-alpha]], [[gpt-model-configuration]]
- Examples: [[learning-rate]], [[mini-batch|batch size]], [[epoch]], [[weight-decay]], [[dropout]], [[context-length]], [[embedding-dimension]]
- Used by: [[stochastic-gradient-descent]], [[adamw-optimizer]], [[llm-training-loop]], [[fine-tuning]], [[low-rank-adaptation]]
- Related: [[overfitting]], [[overfitting-and-underfitting]], [[regularization]], [[model-capacity]], [[random-seed-reproducibility]]
- People: [[sebastian-raschka]]

> **From [[llm-appD-training-loop-bells-and-whistles]]:** Appendix D adds new learning-rate hyperparameters to the training loop: `n_epochs = 15`, `initial_lr = 0.0001`, `peak_lr = 0.01`, `warmup_steps = int(0.2 * total_steps)` (prints 27) and `min_lr = 0.1 * initial_lr` (book pp. 315–316), plus the clipping threshold `max_norm = 1.0` (book pp. 318–320). None of these is learned; you choose them before training. See [[learning-rate-schedule]], [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]].

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6 (batch size as a hyperparameter), book pp. 35–41 (quote on pp. 39–40).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1 (`GPT_CONFIG_124M`), book p. 95; §6.2 (70/10/20 split), book p. 175.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7 (validation vs test accuracy; drop_rate, weight_decay), book pp. 195–200.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5 (model configs, `hparams.json`), book pp. 226–229; §7.6 (Exercise 7.3), book pp. 229–233; §7.8 (adjusting lr, batch size, epochs), book pp. 238–247.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7 (definition, validation set, test once), book pp. 274–278 (quotes on p. 275).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1 (rank $r$ is a hyperparameter, Fig. E.1), book pp. 322–324; §E.4 (r = 16, α = 16 defaults), book pp. 328–336.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1–D.4, book pp. 315–320 (warmup, decay and clipping hyperparameters).
