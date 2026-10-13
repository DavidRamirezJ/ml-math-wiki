---
title: Stochastic Gradient Descent
type: concept
tags: [optimization, gradient-descent, machine-learning, large-scale, deep-learning, llm-training]
difficulty: intermediate
sources: [mml-ch07-continuous-optimization, llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-06-04
updated: 2026-10-13
---

# Stochastic Gradient Descent

> **One-liner:** Instead of measuring the whole hillside before each step, you sample a few spots, guess the downhill direction, and step — much faster, and the guessing even helps you escape bad ditches.

## 🟢 Intuition (Dummy)
Ordinary [[gradient-descent]] looks at *all* your training data to decide each step. If
you have a million examples, that's a million measurements per step — incredibly slow.
The trick: you don't need a perfect downhill direction, just a *roughly* correct one. So
grab a small random handful of examples ("mini-batch"), estimate the slope from those,
and step. Repeat with a new handful. On average the random guesses point the right way,
so you still get to the bottom — and far faster. A bonus: the randomness jiggles you out
of shallow ditches you might otherwise get stuck in.

### In LLMs
Here's the nice surprise from *Build a Large Language Model (From Scratch)*: every
single model in that book — a tiny toy classifier, a GPT-2 pretrained on a short story,
a spam detector, an instruction-following assistant — is trained with the *same* little
loop, and that loop **is** SGD. You shuffle the data, chop it into small batches
(often just 2 or 8 examples!), and for each batch you ask "which way is downhill?" and
take one small step. The book even says it plainly about batch size: small batches need
**less memory** but give **noisier updates** — that's exactly the SGD trade-off, in
everyday words (§2.6, book pp. 39–40).

## 🟡 Mechanics (Practitioner)
ML objectives are usually a **sum over data points**:
$$ L(\boldsymbol\theta) = \sum_{n=1}^{N} L_n(\boldsymbol\theta), $$
e.g. the negative log-likelihood $L(\boldsymbol\theta)=-\sum_{n=1}^N \log p(y_n\mid \mathbf{x}_n,\boldsymbol\theta)$.

**Batch** gradient descent uses the full sum each step:
$$ \boldsymbol\theta_{i+1} = \boldsymbol\theta_i - \gamma_i \sum_{n=1}^{N}\big(\nabla L_n(\boldsymbol\theta_i)\big)^\top. $$
Evaluating all $N$ gradients is expensive. **SGD** replaces the sum with a sum over a
random subset (mini-batch) — in the extreme, a single example. This works because the
full sum is an **empirical estimate of the expected gradient**, and *any* unbiased
estimate (such as a random subsample) suffices for convergence.

**The trade-off (batch size):**
| Large mini-batch | Small mini-batch |
|---|---|
| accurate gradient, low variance | noisy gradient, high variance |
| stable convergence | can escape bad local optima |
| each step costs more | each step is cheap |
| exploits vectorized matrix ops | — |

**ML usage:** SGD (and adaptive variants like Adam) is **the** training algorithm for
large-scale ML — deep nets on millions of images, topic models, reinforcement learning,
large Gaussian processes. The size of central goal is generalization (Ch. 8), so a
precise minimum isn't even needed, making approximate gradients doubly attractive.

### In LLMs (and in PyTorch)
**The update rule, as PyTorch's `torch.optim.SGD` does it.** The LLM book describes
`optimizer.step()` as "multiplying the gradients with the learning rate and adding the
scaled negative gradient to the parameters" (§A.7, book p. 276). In symbols:
$$\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}_{\mathcal B}(\theta)$$
Reading it symbol by symbol:
- $\theta$ = all the trainable numbers of the model (every weight and bias, i.e. `model.parameters()`).
- $\leftarrow$ = "gets replaced by" — this is an update, not an equation.
- $\eta$ ("eta") = the **[[learning-rate]]** (`lr`), the step size. In the toy example $\eta = 0.5$.
- $\nabla_\theta$ = "the [[gradient]] with respect to $\theta$": a list of [[partial-derivative|partial derivatives]], one per parameter.
- $\mathcal{L}_{\mathcal B}$ = the loss computed on the current [[mini-batch]] $\mathcal B$ only (not the whole dataset).
- The minus sign = go *downhill* (against the gradient).

The mini-batch loss is an **average** over the batch, because PyTorch's
`F.cross_entropy` uses `reduction='mean'` by default:
$$\mathcal{L}_{\mathcal B}(\theta) = \frac{1}{B}\sum_{b\in\mathcal B} \ell_b(\theta),\qquad
\nabla_\theta\mathcal{L}_{\mathcal B} = \frac{1}{B}\sum_{b\in\mathcal B}\nabla_\theta \ell_b(\theta)$$
- $B = |\mathcal B|$ = the batch size (how many examples are in the batch).
- $\ell_b$ = the loss on one example $b$ (e.g. its [[cross-entropy-loss]]).
- $\sum_{b\in\mathcal B}$ = "add up over every example $b$ in the batch".

> Note on notation: the MML book (above) writes the step size as $\gamma$ and sums the
> per-example gradients; the LLM book / PyTorch writes $\eta$ and *averages* them. Same
> algorithm — averaging just rescales the effective learning rate by $1/B$.

**The training loop that implements SGD** (Listing A.9, §A.7, book pp. 274–275):
```python
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()   # reset old gradients (PyTorch adds them up otherwise)
        loss.backward()         # backpropagation: fill every param.grad
        optimizer.step()        # theta <- theta - lr * grad
    model.eval()
```
- The toy dataset has 5 examples, `batch_size=2`, `shuffle=True`, `drop_last=True` →
  $\lfloor 5/2\rfloor = 2$ batches per epoch, so 3 epochs = **6 SGD updates** (§A.6–A.7, book pp. 272–275).
- The printed training loss falls **0.75 → 0.65 → 0.44 → 0.13 → 0.03 → 0.00**; the book
  says the model "converged" on the training set (book p. 275).
- `optimizer.zero_grad()` matters because PyTorch **accumulates** gradients into `.grad`;
  without the reset, step $t$ would use the sum of all past gradients (book p. 276).
- Where the gradients come from: `loss.backward()` runs [[backpropagation]].

**Where the "stochastic" comes from: the DataLoader.** Shuffling + batching *is* the
random sampling of mini-batches (see [[pytorch-dataset-and-dataloader]]):
- `shuffle=True` for the training loader only; validation/test loaders keep a fixed
  order (§A.6, book p. 272; §6.3, book p. 180; §7.4, book p. 225).
- `drop_last=True` for training: a much smaller last batch "can disturb convergence"
  (§A.6, book p. 273) and causes "loss spikes during training" (§2.6, book p. 38). In SGD
  language: a batch of size 1 gives a high-variance gradient estimate.
- Number of updates per [[epoch]]: $\lfloor N/B\rfloor$ with `drop_last=True`,
  $\lceil N/B\rceil$ without. Spam example: the §6.2 split has 1,045 training rows, and the book prints
  "130 training batches" = $\lfloor 1045/8\rfloor$ (§6.3, book p. 181; §6.7, book p. 195). The
  1,045 → 130 arithmetic is a wiki consistency check, not printed in the book.

**Batch sizes the book actually uses** (all small!):
| Where | Batch size | Optimizer | Learning rate |
|---|---|---|---|
| Toy MLP (§A.7, p. 274) | 2 | `SGD` | 0.5 |
| GPT-2 124M pretraining on "The Verdict" (§5.1, p. 143; §5.2, p. 149) | 2 | `AdamW`, weight decay 0.1 | 0.0004 |
| Spam classifier fine-tuning (§6.3, §6.7, pp. 180, 197) | 8 | `AdamW`, weight decay 0.1 | 5e-5 |
| Instruction fine-tuning, GPT-2 355M (§7.4, §7.6, pp. 225, 231) | 8 | `AdamW`, weight decay 0.1 | 5e-5 |
| LoRA spam fine-tuning (§E.2, §E.4, pp. 325, 334) | 8 | `AdamW`, weight decay 0.1 | 8e-4 |
| Appendix D pretraining with warmup + cosine (§D.4, p. 320) | 2 | `AdamW`, weight decay 0.1 | peak 0.001 |

Everything after Appendix A uses **[[adamw-optimizer|AdamW]]** — still a *stochastic*
mini-batch method (one update per batch from a noisy batch gradient), but with a smarter
per-parameter step than plain SGD (§5.2, book p. 148). Exercise 7.3 even tells you to
shrink the batch size **8 → 4, 2, or 1** if you run out of GPU memory — pure
memory-vs-noise trade-off (book p. 233).

**Making the step size a function of time.** Appendix D turns the constant $\eta$ into a
schedule $\eta_t$: linear **[[learning-rate-warmup]]** from `initial_lr` up to `peak_lr`,
then **[[cosine-decay]]** down to `min_lr`, plus **[[gradient-clipping]]** to cap the
gradient's length at `max_norm=1.0` (§D.1–D.4, book pp. 313–321):
$$\theta_{t+1} = \theta_t - \eta_t\,\mathbf g_t$$
- $t$ = the global step counter (how many updates so far).
- $\eta_t$ = the learning rate at step $t$ (see [[learning-rate-schedule]]).
- $\mathbf g_t$ = the (possibly clipped) mini-batch gradient at step $t$.

## 🔴 Theory (Expert)
The "stochastic" label means we admit we only have a **noisy approximation** of the
gradient. Convergence is guaranteed by constraining the distribution of the approximate
gradients: it suffices that the estimate is **unbiased**,
$\mathbb{E}[\widehat{\nabla L}] = \nabla L$.

- **Robbins–Monro / Bottou (1998):** when the learning rate $\gamma_i$ decreases at an
  appropriate rate (e.g. $\sum_i\gamma_i=\infty$, $\sum_i\gamma_i^2<\infty$) and under
  mild assumptions, SGD converges **almost surely** to a local minimum.
- The mini-batch gradient is exactly the empirical mean of per-example gradients
  (Section 6.4.1), so reducing batch size raises the **variance** of the estimate — the
  same variance/sample-size relationship as in Monte Carlo mean estimation.
- [[gradient-descent|Momentum]] is especially valuable here, since averaging successive
  noisy estimates reduces variance in the parameter update.

### In LLMs — deeper notes (wiki derivations, labelled as such)
- **Variance falls like $1/B$.** If the per-example gradients are i.i.d. draws with
  covariance $\Sigma$, then
  $\mathrm{Cov}\big[\tfrac1B\sum_b \nabla\ell_b\big] = \tfrac1B\Sigma$. That is the formal
  version of the book's "small batches → noisier updates" (§2.6, book pp. 39–40); it is
  a [[monte-carlo-estimation]] fact, not a formula from the book. See
  [[covariance-and-variance]], [[expected-value-and-mean]].
- **Unbiasedness and `drop_last`.** With shuffling, each batch is a uniformly random
  subset, so $\mathbb E[\nabla\mathcal L_{\mathcal B}] = \nabla\mathcal L$ still holds;
  dropping the final partial batch removes a *high-variance* (not a biased) step. Because
  the dropped example changes with each shuffle, every example is still seen on average
  (§A.6–A.7: one random example of the 5 is skipped per epoch).
- **Multi-GPU = bigger batch.** In DistributedDataParallel each of $W$ GPUs computes the
  gradient on its own non-overlapping mini-batch, then the gradients are averaged
  (§A.9, book pp. 282–283):
  $$\mathbf g = \frac1W\sum_{r=0}^{W-1}\nabla_\theta L_r(\theta),\qquad \theta\leftarrow\theta-\eta\,\mathbf g$$
  $W$ = number of GPUs (`world_size`), $r$ = GPU index (rank), $L_r$ = loss on GPU $r$'s
  batch. By linearity of the gradient this equals one SGD step on the combined batch of
  size $W\cdot B$ (a math annotation; the book says only "averaged and synchronized").
  See [[distributed-data-parallel]].
- **Clipping interacts with the optimizer.** With plain SGD, clipping to `max_norm`
  caps the step length at $\eta\cdot$`max_norm`; with AdamW's per-parameter
  normalisation the cap is not as direct (the book does not discuss this; §D.3).
- **Schedules vs. Robbins–Monro.** The book's warmup + cosine decay is a practical
  schedule, not one meeting $\sum\gamma_i^2<\infty$ on an infinite horizon; warmup is used
  for stability early on (the book: smaller early updates decrease "the risk of the model
  encountering large, destabilizing updates", §D.1, book p. 314) and decay for fine
  convergence late.
- **Fine-tuning is SGD from a warm start.** In Chapters 6–7 and Appendix E the iterates
  start at the pretrained weights $\theta_0 = \theta_{\text{pretrained}}$ rather than at a
  random [[weight-initialization]] — that is what [[fine-tuning]] means at the optimizer level.

## Connections
- Builds on: [[gradient-descent]], [[gradient]], [[mini-batch]], [[expected-value-and-mean]], [[covariance-and-variance]]
- Leads to: neural-network optimization, [[adamw-optimizer]], [[learning-rate-schedule]], [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]], [[gradient-accumulation]], [[distributed-data-parallel]]
- Used by: deep learning, [[linear-regression]], large-scale ML, [[llm-training-loop]], [[training-loop]], [[pretraining]], [[fine-tuning]], [[low-rank-adaptation]]
- Related: [[backpropagation]] (computes the gradients SGD consumes), [[pytorch-dataset-and-dataloader]] (does the random batching), [[epoch]], [[learning-rate]], [[hyperparameter]], [[monte-carlo-estimation]]
- People: —

## Sources
- `raw/mml-book.pdf` §7.1.3 (Eqs. 7.13–7.15), printed pp. 231–233 (PDF pp. 236–238).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6 (batch size, `drop_last`, noisy updates), book pp. 35–41.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 (training loop, AdamW, lr 0.0004), book pp. 146–151.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3 (spam data loaders, batch size 8, 130 batches), book pp. 175–181; §6.7 (classification training loop, lr 5e-5), book pp. 195–200.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.4 (instruction data loaders), book pp. 223–226; §7.6 (instruction fine-tuning, Exercise 7.3), book pp. 229–233.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.6 (shuffle, `drop_last`, convergence), book pp. 270–274; §A.7 (Listing A.9, `torch.optim.SGD`, lr 0.5), book pp. 274–278; §A.9 (DDP gradient averaging), book pp. 279–288.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1 (learning-rate warmup), book pp. 313–316; §D.4 (full training function), book pp. 319–321.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2 (LoRA data loaders), book pp. 324–325.
