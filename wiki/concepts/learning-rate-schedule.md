---
title: Learning-Rate Schedule
type: concept
tags: [optimization, deep-learning, llm, training]
difficulty: intermediate
sources: [llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Learning-Rate Schedule

> **One-liner:** Instead of using one fixed learning rate, you let it change over the course of training according to a plan.

## 🟢 Intuition (Dummy)
Think about how you'd run a long race: an easy start to warm up, a strong pace through the middle,
and a careful finish. You don't run the whole thing at a single speed.

The [[learning-rate]] is how big a step the model takes each time it learns from a batch. A
**learning-rate schedule** is the race plan for that step size: a rule that says "at step $t$, use
this learning rate." Raschka's book builds the most popular LLM plan: a gentle
[[learning-rate-warmup|warmup]] ramp at the start, then a smooth [[cosine-decay|cosine decay]]
down to a small value at the end.

## 🟡 Mechanics (Practitioner)
**Definition.** A schedule is a function from the training step to a learning rate:
$$\eta : t \mapsto \eta_t, \qquad t = 0, 1, \dots, T-1$$
- $\eta_t$ = the learning rate used at step $t$.
- $t$ = the global step counter, counted across all epochs (`global_step`).
- $T$ = total number of steps $= B \cdot E$, where $B$ = batches per epoch (`len(train_loader)`) and $E$ = epochs (`n_epochs`) (book p. 319).
- $t \mapsto \eta_t$ reads "$t$ is sent to $\eta_t$".

The optimizer update then uses the current value: $\theta_{t+1} = \theta_t - \eta_t \cdot (\text{update direction})$
(for plain [[gradient-descent]] the direction is the [[gradient]] $\nabla_\theta L$).

**The book's schedule (warmup + cosine), §D.2/§D.4:**
$$\eta_t = \begin{cases} \eta_0 + t\,\dfrac{\eta_{\max} - \eta_0}{W} & t < W \quad\text{(linear warmup)}\\[8pt]
\eta_{\min} + (\eta_{\max} - \eta_{\min})\cdot \tfrac12\Big(1 + \cos\!\big(\pi\,\tfrac{t-W}{T-W}\big)\Big) & t \ge W \quad\text{(cosine decay)}\end{cases}$$
- $\eta_0$ = start learning rate (`initial_lr`); $\eta_{\max}$ = peak (`peak_lr`); $\eta_{\min}$ = floor (`min_lr`).
- $W$ = number of warmup steps (`warmup_steps`).
- Top line: a straight ramp up. Bottom line: half a cosine wave sliding from the peak to the floor. See
  [[learning-rate-warmup]] and [[cosine-decay]] for the symbol-by-symbol walk-throughs.

**How PyTorch applies it.** The book sets the rate by hand at every step by writing into the
optimizer's parameter groups (book pp. 315, 320):
```python
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
track_lrs.append(lr)
```
Recording the values in `track_lrs` and plotting them is the book's way to check the schedule is right
(Figures D.1 and D.2).

**The three schedules shown in the book:**

| Schedule | Where | Settings | Shape |
|---|---|---|---|
| Constant | Ch. 5 `train_model_simple` (book p. 149) | AdamW `lr=0.0004`, 10 epochs | flat line |
| Warmup then constant | §D.1, Fig. D.1 | 15 epochs (~135 steps); 0.0001 → 0.01 over 27 steps | ramp, then flat |
| Warmup then cosine | §D.2, Fig. D.2 | 15 epochs (~135 steps); 0.0001 → 0.01, then → `min_lr = 0.00001` | ramp, then half-cosine slide |

**Full training function (§D.4, book pp. 319–321).** `train_model(model, train_loader, val_loader, optimizer, device,
n_epochs, eval_freq, eval_iter, start_context, tokenizer, warmup_steps, initial_lr=3e-05, min_lr=1e-6)`:
- reads the peak from the optimizer: `peak_lr = optimizer.param_groups[0]["lr"]` ("assuming we use it as the peak learning rate");
- computes `total_training_steps = len(train_loader) * n_epochs` and `lr_increment = (peak_lr - initial_lr) / warmup_steps`;
- per batch: zero grads → step +1 → compute lr (warmup or cosine) → write lr → loss → `backward()` → [[gradient-clipping|clip]] (only after warmup) → `optimizer.step()`;
- returns `train_losses, val_losses, track_tokens_seen, track_lrs`.

The book's run: seed 123, `GPT_CONFIG_124M`, `peak_lr = 0.001`, AdamW with `weight_decay=0.1`,
`n_epochs=15`, `eval_freq=5`, `eval_iter=1`, `warmup_steps=27`, `initial_lr=1e-5`, `min_lr=1e-5`. It takes about
5 minutes on a MacBook Air. Losses: step 0 train 10.934 / val 10.939; step 130 train 0.035 / val 6.938 — the
model [[overfitting|overfits]] the tiny "The Verdict" dataset, but the function works (book pp. 320–321).

> Note: the printed run code defines `peak_lr = 0.001` but does not pass `lr=peak_lr` to AdamW. It still works
> because PyTorch's AdamW default learning rate is also 0.001 (validation observation, not stated in the book).

## 🔴 Theory (Expert)
- **Why schedules at all.** Early on, large steps can destabilise a randomly initialised network (motivating warmup);
  late in training, stochastic-gradient noise keeps a constant-$\eta$ iterate bouncing in a region whose size scales with
  $\eta$ (motivating decay). Classic SGD theory needs $\sum_t \eta_t = \infty$ and $\sum_t \eta_t^2 < \infty$ for
  convergence; finite-horizon schedules imitate the "shrink at the end" part (background).
- **Continuity.** The book's piecewise schedule is continuous at $t = W$ up to one increment: the warmup's last value is
  $\eta_{\max} - \Delta$ and the cosine starts at $\eta_{\max}$. At the end, $t \le T-1$ so $\eta_{\min}$ is approached, not reached.
- **Other families.** Step decay (multiply by $\gamma$ every $k$ epochs), exponential decay $\eta_0\gamma^t$, inverse-square-root
  decay (original Transformer), linear decay, cosine with warm restarts (SGDR), and "warmup–stable–decay". PyTorch's
  `torch.optim.lr_scheduler` module implements many; the book codes the schedule by hand for transparency.
- **Interaction with adaptive optimisers.** In [[adamw-optimizer|AdamW]], $\eta_t$ multiplies the normalised moment ratio and
  (in PyTorch) also scales the decoupled [[weight-decay]] term, so the schedule shapes both the gradient step and the
  regularisation strength.
- **The schedule is a hyperparameter object.** Its shape, $W$, $\eta_{\max}$, $\eta_{\min}$ are all
  [[hyperparameter|hyperparameters]] tuned on validation loss.

## Connections
- Builds on: [[learning-rate]], [[gradient-descent]], [[stochastic-gradient-descent]], [[epoch]], [[mini-batch]], [[hyperparameter]]
- Leads to: [[learning-rate-warmup]], [[cosine-decay]]
- Used by: [[llm-training-loop]], [[training-loop]], [[pretraining]], [[adamw-optimizer]]
- Related: [[gradient-clipping]], [[weight-decay]], [[overfitting]], [[training-and-validation-loss]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1, book pp. 314–316 (warmup, Figure D.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.2, book pp. 316–317 (cosine decay, Figure D.2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.4, book pp. 319–321 (`train_model`, run and output).
