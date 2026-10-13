---
title: Mini-Batch and Batch Size
type: concept
tags: [optimization, training, pytorch, llm, hyperparameter]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-ch03-coding-attention-mechanisms, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Mini-Batch and Batch Size

> **One-liner:** A mini-batch is a small handful of training examples processed together for one learning step, and the batch size is how many examples are in that handful.

## 🟢 Intuition (Dummy)
Imagine you are tasting soup to decide how much salt to add.
- Taste the **whole pot** every time: very accurate, but slow and you need a huge spoon.
- Taste **one drop**: fast, but one drop can mislead you.
- Taste **a spoonful**: a good compromise. That spoonful is a **mini-batch**.

Training a [[neural-network]] works the same way. To decide how to nudge the weights, the
model looks at a few examples at once (say 8 text messages), averages how wrong it was on
them, and takes one step. Then it grabs the next 8.

The book says it plainly: small batches need **less memory** but give **noisier updates**.
So the batch size is a trade-off, and a [[hyperparameter]] you experiment with (book pp. 39–40).
Batching also makes things fast, because a GPU is great at doing the same operation on many
examples in parallel.

If your GPU runs out of memory, the first thing the book suggests is to lower the batch size,
for example from 8 to 4, 2, or 1 (Exercise 7.3 solution, book p. 310).

## 🟡 Mechanics (Practitioner)

### The update rule
Pick a mini-batch $\mathcal{B}$ of $B$ examples. Then

$$\theta \leftarrow \theta - \eta\,\underbrace{\frac{1}{B}\sum_{i\in\mathcal{B}} \nabla_\theta\, \ell\big(f_\theta(\mathbf{x}_i), y_i\big)}_{\hat g\ \text{(mini-batch gradient)}}$$

Symbol by symbol:
- $\theta$ = all trainable [[model-parameters]]; $\leftarrow$ = "replace with the new value".
- $\eta$ (eta) = the [[learning-rate]] (the book's toy loop uses `lr=0.5`, book p. 275).
- $\mathcal{B}$ = the set of example indices in this batch; $B=|\mathcal{B}|$ = the batch size.
- $\ell(f_\theta(\mathbf{x}_i), y_i)$ = the loss on example $i$ (prediction vs. label).
- $\nabla_\theta$ = the [[gradient]]: the direction of steepest increase of the loss.
- $\frac{1}{B}\sum_{i\in\mathcal{B}}$ = average over the batch. PyTorch losses such as `F.cross_entropy` average by default (`reduction='mean'`).
- $\hat g$ ("g-hat") = the estimated gradient. The hat says it is an estimate of the full-data gradient.

This is [[stochastic-gradient-descent]] with mini-batches. In code, one inner-loop iteration
is one mini-batch, one call to `loss.backward()` and one `optimizer.step()`.

### What a batch looks like as a tensor
The [[pytorch-dataset-and-dataloader|`DataLoader`]] stacks $B$ examples along a new first axis:

| Where | `batch_size` | Input shape | Target shape | Book |
|---|---|---|---|---|
| Toy classifier (App. A) | 2 | `[2, 2]` (batch, features) | `[2]` | p. 272 |
| Ch. 2 sliding window | 8 | `[8, 4]` (batch, tokens) | `[8, 4]` | pp. 40–41 |
| Ch. 6 spam (and App. E) | 8 | `[8, 120]` | `[8]` | p. 180 (App. E reuses the same `SpamDataset` and `batch_size=8`, p. 325; it does not reprint the shape) |
| Ch. 7 instructions | 8 | `[8, 61]`, `[8, 76]`, … | same as input | pp. 225–226 |
| Default `create_dataloader_v1` | 4 | `[4, 256]` | `[4, 256]` | p. 38 |

In Ch. 7 the second axis changes from batch to batch because every batch is padded only to its
own longest example ([[dynamic-padding]]). Inside one batch all rows must have the same length.

### Batches and epochs
One **[[epoch]]** = one pass over the training data, visiting every example exactly once
(book p. 272). With $N$ examples the epoch has $\lceil N/B\rceil$ batches, or $\lfloor N/B\rfloor$
with `drop_last=True`:
- Toy set: $N=5$, $B=2$ → batches of size 2, 2, 1; with `drop_last=True`, 2 batches (book pp. 272–273).
  Training for 3 epochs × 2 batches = 6 updates; the loss falls 0.75 → 0.65 → 0.44 → 0.13 → 0.03 → 0.00 (book p. 275).
- Spam data: $B=8$ → 130 training, 19 validation, 38 test batches (book p. 181).

Why drop a short last batch? A batch that is much smaller than the others "can disturb
convergence" (book p. 273) and can cause "loss spikes during training" (book p. 38).

### Batches at evaluation time too
Even when not training, you usually evaluate in batches. `compute_accuracy` loops over a loader
because with large datasets the model can only be called on a small part at a time, due to memory
limits; using the training batch size makes it scale to any dataset size (book p. 278).

### Batches in the sliding-window loader (Ch. 2)
```python
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4,
                                  stride=4, shuffle=False)
inputs, targets = next(iter(dataloader))   # both torch.Size([8, 4])
```
Batch size 1 is good for illustration, but real training uses bigger batches; real input lengths
are "at least 256" tokens (book p. 39). Exercise 2.2 asks you to try other settings; for
`batch_size=4, max_length=2, stride=2` the first batch is `[[40, 367], [2885, 1464], [1807, 3619], [402, 271]]`
(book p. 301).

## 🔴 Theory (Expert)
- **Unbiasedness and variance.** If $\mathcal{B}$ is drawn uniformly, $\mathbb{E}[\hat g]=\nabla_\theta \hat R_N(\theta)$,
  the full-data gradient (an [[expected-value-and-mean|expectation]] argument). For i.i.d. sampling with
  per-example gradient covariance $\Sigma$,
  $$\operatorname{Cov}[\hat g]=\tfrac{1}{B}\Sigma,$$
  so the noise standard deviation falls like $1/\sqrt{B}$. This is the precise form of "smaller batches,
  noisier updates" (book pp. 39–40); see [[covariance-and-variance]]. (Sampling without replacement within
  an epoch shrinks it slightly more, by a factor $\frac{N-B}{N-1}$.)
- **Memory cost.** Activation memory for the backward pass scales linearly in $B$ (and, for transformers, in
  $B\cdot T$ for the MLP parts and $B\cdot T^2$ for attention scores). That is why lowering $B$ or the
  maximum length are the two out-of-memory remedies the book gives (book p. 310).
- **Large-batch equivalences.** Gradients are linear, so the average of $W$ per-shard mini-batch gradients
  equals the gradient of the mean loss over the combined batch of size $WB$. That is why
  [[distributed-data-parallel]] with $W$ GPUs, each getting a different non-overlapping batch and averaging
  gradients before the step, behaves like single-GPU training with a $W$-times larger batch (book p. 283
  states the averaging; the equivalence is the math behind it). [[gradient-accumulation]] uses the same
  identity over time instead of over devices.
- **Batch size and learning rate interact.** Larger $B$ means less noise per step but fewer steps per epoch;
  the common heuristic is to scale $\eta$ roughly linearly with $B$ (with [[learning-rate-warmup]]) up to a
  critical batch size, beyond which extra batch size buys little. Noise from small batches also acts as an
  implicit [[regularization]].
- **Mean reduction with masking.** When targets contain ignored positions (−100 in Ch. 7), the mean is over
  the set $\mathcal{I}$ of non-ignored tokens in the batch, not over $B\cdot T_b$, so batches with lots of
  padding are not down-weighted per real token (see [[ignore-index-loss-masking]]).
- **Batch-dependent layers.** [[batch-normalization]] computes statistics across the batch axis, so its
  behaviour depends on $B$; LLMs use [[layer-normalization]] instead, which normalises each token on its own
  and is independent of batch size (Ch. 4).

### From llm-ch03-coding-attention-mechanisms
In §3.5.3 the book makes the attention class work on mini-batches. It simulates a batch of two texts by stacking the same input twice: `torch.stack((inputs, inputs), dim=0)`, shape `(2, 6, 3)` (book p. 80). The batch axis stays at position 0, so the score product transposes only axes 1 and 2: `queries @ keys.transpose(1, 2)` (Listing 3.3, book p. 81). Every text in the batch gets its own $6 \times 6$ weight matrix; texts in a batch never attend to each other. See [[causal-attention]], [[batched-matrix-multiplication]].

## Connections
- Builds on: [[gradient]], [[gradient-descent]], [[expected-value-and-mean]], [[covariance-and-variance]], [[tensor]]
- Core of: [[stochastic-gradient-descent]], [[training-loop]], [[llm-training-loop]]
- Produced by: [[pytorch-dataset-and-dataloader]], [[collate-function]], [[sliding-window-sampling]]
- Related: [[epoch]], [[hyperparameter]], [[learning-rate]], [[dynamic-padding]], [[gradient-accumulation]], [[distributed-data-parallel]], [[batch-normalization]], [[layer-normalization]], [[gpu-computing]], [[overfitting]]
- People: [[sebastian-raschka]]

> **From [[llm-appD-training-loop-bells-and-whistles]]:** In Appendix D every `optimizer.step()` is one mini-batch update with `batch_size=2`; the step counter `global_step` counts mini-batches, so the [[learning-rate-schedule]] changes the learning rate once per mini-batch (book pp. 314–320).

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** In §5.1.3 the book makes mini-batches of shape `[2, 256]`: `batch_size=2`, `max_length=stride=256` (the context length). *The Verdict* (5,145 tokens) then gives **9 training batches and 1 validation batch** (book pp. 143–144). The book says real LLMs often use batch sizes of 1,024 or larger (book p. 143). `calc_loss_batch` gives the mean cross-entropy over all $2 \times 256 = 512$ tokens of one batch, and `calc_loss_loader` averages these batch losses (book pp. 144–145). The §5.2 training loop does one AdamW update per mini-batch (book pp. 147–148), which is mini-batch [[stochastic-gradient-descent]].

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book pp. 38–41 (batch size as memory/noise trade-off and hyperparameter; `drop_last` and loss spikes; batch tensors of shape 8×4).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book pp. 179–181 (batch size 8, `[8, 120]` inputs, 130/19/38 batches).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.4, book pp. 225–226 (batch size 8, variable-length batches).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6, book pp. 272–273 (epoch definition, uneven last batch, convergence); §A.7, book pp. 275–278 (training output, batch-wise accuracy); §A.9.3, book p. 283 (per-GPU minibatches, gradient averaging).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2, book p. 325 (batch size 8 for LoRA).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book p. 301 (Exercise 2.2) and book p. 310 (Exercise 7.3: lower batch size to fix out-of-memory).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5.3, book pp. 80–82 (batch of 2 via `torch.stack`, shape (2, 6, 3), Listing 3.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D intro, book p. 314 (batch size 2); §D.4, book pp. 319–320 (per-batch `global_step`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3, book pp. 143–145 (batch size 2, 9/1 batches, Listing 5.2); §5.2, book pp. 147–148 (one update per batch, Listing 5.3).
