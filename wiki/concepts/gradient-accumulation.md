---
title: Gradient Accumulation
type: concept
tags: [optimization, deep-learning, pytorch, training, llm]
difficulty: intermediate
sources: [llm-appA-introduction-to-pytorch, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Gradient Accumulation

> **One-liner:** Gradient accumulation means adding up the gradients from several small batches before you take one update step, so a small GPU can act like it trained on one big batch.

## 🟢 Intuition (Dummy)
Picture yourself tasting a big pot of soup to decide how much salt to add.
The ideal is to taste a big ladle, but your spoon is tiny. So you take four
small sips, keep a running mental note of "how salty" each one was, and only
*then* add salt, once, based on all four sips together. That's gradient
accumulation: each small batch is one sip, the running note is the gradient
total, and adding salt is the parameter update.

There's a flip side, and it's the reason the book brings this up at all. In
PyTorch, that running note is **always on**. Every time you compute gradients,
PyTorch *adds* them to whatever's already stored. If you forget to wipe the
note clean between updates, yesterday's sips leak into today's decision. That's
why every training loop in the book calls `optimizer.zero_grad()` before each
update.

## 🟡 Mechanics (Practitioner)

### What the book actually says
The book covers this idea in two short places:
- In the basic training loop (Listing A.9, §A.7), there's a NOTE: *"To prevent
  undesired gradient accumulation, it is important to include an
  `optimizer.zero_grad()` call in each update round to reset the gradients to 0.
  Otherwise, the gradients will accumulate, which may be undesired"* (book p. 276).
- In Appendix B (further reading for Appendix A), the author asks *why PyTorch
  doesn't call `zero_grad()` for us automatically*. His answer: *"In some
  instances, it may be desirable to accumulate the gradients, and PyTorch will
  leave this as an option for us"*, and he points to his article "Finetuning
  Large Language Models on a Single GPU Using Gradient Accumulation"
  (https://mng.bz/8wPD) (book p. 298).

So the book gives you the *mechanism* (PyTorch adds gradients up) and the
*reason it's a manual call* (you might want accumulation on purpose). The
recipe and the math below are standard practice that build on that, not text
from the book.

### The book's training loop (where accumulation gets switched off)
```python
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()   # reset gradients from the previous round to 0
        loss.backward()         # compute gradients; PyTorch ADDS them into .grad
        optimizer.step()        # update the parameters
```
(Listing A.9, book pp. 274–275. Here `batch_size=2`, `lr=0.5`, 3 epochs, 2
batches per epoch, so 6 updates in total; the loss falls 0.75 → 0.65 → 0.44 →
0.13 → 0.03 → 0.00.) The same `zero_grad → backward → step` pattern shows up in
the GPT pretraining loop in chapter 5 (book p. 147), the classification
fine-tuning loop in chapter 6 (book p. 196), and the Appendix D loops (book
pp. 315–319).

### How PyTorch stores gradients
Every trainable tensor $\theta$ has a `.grad` slot. A call to `loss.backward()`
does this:

$$\theta.\texttt{grad} \leftarrow \theta.\texttt{grad} + \nabla_\theta \mathcal{L}$$

Reading it symbol by symbol:
- $\leftarrow$ = "gets overwritten with"
- $\theta.\texttt{grad}$ = the gradient number(s) PyTorch already has stored for this parameter
- $\nabla_\theta \mathcal{L}$ = "the gradient of the loss $\mathcal{L}$ with respect to $\theta$", i.e. how much the loss changes if you nudge $\theta$ (see [[gradient]])
- the $+$ is the whole story: it **adds**, it doesn't replace.

A tiny illustration (mine, not the book's): take one weight $w = 2$ and the
loss $\mathcal{L} = w^2$. The derivative is $2w = 4$. Call `backward()` once and
`w.grad` is `4.`. Call it again on a fresh forward pass *without* zeroing, and
`w.grad` becomes `8.`, which is wrong for a normal update. `zero_grad()` resets
it to 0 (or `None`) so the next `backward()` starts clean.

### Using accumulation on purpose
Say your GPU only fits batches of 2, but you want the update to behave like a
batch of 8. Split the "big batch" into $K = 4$ **micro-batches**, run
`backward()` on each one, and only call `step()` after the fourth. That gives
an **effective batch size** of $K \times B = 4 \times 2 = 8$.

The one thing you must get right is scaling. With `reduction='mean'`,
`F.cross_entropy` returns the *average* loss over the micro-batch. To end up
with the average over the whole big batch, divide each micro-batch loss by $K$:

$$
\mathbf{g} \;=\; \sum_{k=1}^{K} \nabla_\theta\!\left(\frac{1}{K}\,\mathcal{L}_k(\theta)\right)
\;=\; \nabla_\theta\!\left(\frac{1}{K}\sum_{k=1}^{K}\mathcal{L}_k(\theta)\right),
\qquad \theta \leftarrow \theta - \eta\,\mathbf{g}
$$

Symbol by symbol:
- $K$ = the number of micro-batches you accumulate (the "accumulation steps")
- $k$ = which micro-batch you're on, counting $1, 2, \dots, K$
- $\mathcal{L}_k(\theta)$ = the mean cross-entropy loss on micro-batch $k$ (see [[cross-entropy-loss]])
- $\frac{1}{K}$ = the scaling factor, so the total is an average and not a sum
- $\sum_{k=1}^{K}$ = "add up the terms for $k = 1$ through $K$"
- $\mathbf{g}$ = the accumulated gradient sitting in `.grad` after the $K$-th `backward()`
- $\eta$ = the [[learning-rate]] (`lr`, e.g. 0.5 in Listing A.9)
- the middle "$=$" holds because the gradient is **linear**: the gradient of a sum is the sum of the gradients.

In code (standard pattern, not a book listing):
```python
accum_steps = 4                          # K
optimizer.zero_grad()
for batch_idx, (features, labels) in enumerate(train_loader):
    logits = model(features)
    loss = F.cross_entropy(logits, labels) / accum_steps   # scale by 1/K
    loss.backward()                                        # adds into .grad
    if (batch_idx + 1) % accum_steps == 0:
        optimizer.step()                                   # one update per K micro-batches
        optimizer.zero_grad()                              # now wipe the slate
```

### Where it matters for LLMs
LLM training is memory-hungry: activations grow with batch size × [[context-length]]
× [[embedding-dimension]]. If a batch of 8 sequences of 1,024 tokens won't fit
in memory, accumulation lets you keep the *optimization behaviour* of batch size
8 on hardware that can only hold 1 or 2 sequences at a time. The price is time:
$K$ forward/backward passes per update instead of one. The book's article
title says it plainly: fine-tuning an LLM "on a single GPU using gradient
accumulation".

## 🔴 Theory (Expert)
- **Exact equivalence (with caveats).** By linearity of $\nabla$, the
  accumulated gradient $\mathbf{g}$ equals the full-batch gradient of the mean
  loss whenever (a) every micro-batch has the same number of examples and (b)
  each loss is a per-example mean. If micro-batches differ in size, or if the
  loss is averaged over a variable number of *tokens* (as with
  [[ignore-index-loss-masking]] in instruction fine-tuning), dividing by $K$ is
  only approximate; the exact fix is to sum per-token losses and divide by the
  total token count across all $K$ micro-batches.
- **Where equivalence breaks.** Layers whose forward pass depends on batch
  statistics, such as [[batch-normalization]], see micro-batch statistics, not
  big-batch statistics, so the result differs. GPT-style models use
  [[layer-normalization]], which normalizes per token, so this issue doesn't
  arise for them. [[dropout]] masks are drawn independently per micro-batch,
  which is harmless in expectation.
- **Statistical view.** A minibatch gradient is a Monte Carlo estimate of the
  true risk gradient ([[stochastic-gradient-descent]], [[monte-carlo-estimation]]).
  Averaging $K$ independent micro-batch estimates cuts the variance by a factor
  of $K$, exactly like a $K$-times-larger batch. What accumulation doesn't give
  you is a speed-up: the number of examples processed per second is unchanged.
- **Interaction with other tricks.** [[gradient-clipping]] (Appendix D) should be
  applied to the *accumulated* gradient, just before `optimizer.step()`, not
  per micro-batch. Learning-rate schedules ([[learning-rate-warmup]],
  [[cosine-decay]]) should count *optimizer steps*, not micro-batches.
  [[adamw-optimizer]] state is updated once per step, so it also sees the
  accumulated gradient.
- **Relation to data parallelism.** [[distributed-data-parallel]] does the same
  averaging, but *across GPUs at the same time* (via all-reduce) instead of
  *across time on one GPU*. The two combine: $W$ GPUs × $K$ accumulation steps ×
  micro-batch $B$ gives an effective batch of $W K B$.

## Connections
- Builds on: [[gradient]], [[backpropagation]], [[automatic-differentiation]], [[mini-batch]], [[stochastic-gradient-descent]], [[gradient-descent]], [[pytorch]]
- Leads to: [[distributed-data-parallel]], [[llm-training-loop]], [[training-loop]]
- Related: [[gradient-clipping]], [[learning-rate-schedule]], [[gpu-computing]], [[parameter-count-and-memory]], [[expected-value-and-mean]]
- Used by: [[pretraining]], [[fine-tuning]], [[instruction-fine-tuning]] on memory-limited GPUs
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7 (Listing A.9 and the NOTE on `optimizer.zero_grad()`), book pp. 274–276 (PDF pp. 296–298).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, "Appendix A" further-reading block (why `zero_grad()` is manual; article "Finetuning Large Language Models on a Single GPU Using Gradient Accumulation", https://mng.bz/8wPD), book p. 298 (PDF p. 320).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 (`optimizer.zero_grad()` in the GPT training loop), book p. 147.
- The accumulation recipe, the $1/K$ scaling formula and the Theory notes are standard practice added for completeness; the book itself doesn't derive them.
