---
title: Learning-Rate Warmup
type: concept
tags: [optimization, deep-learning, llm, training]
difficulty: intermediate
sources: [llm-appD-training-loop-bells-and-whistles, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Learning-Rate Warmup

> **One-liner:** Instead of taking big training steps from the very first moment, you start with tiny steps and grow them gradually up to your chosen maximum.

## 🟢 Intuition (Dummy)
Picture driving a car out of an icy parking lot. If you floor the gas pedal right away, the
wheels spin and the car slides sideways. If you press the pedal gently and speed up bit by bit,
you get going safely — and *then* you can drive at full speed.

A freshly created neural network is that icy parking lot. Its weights are random, so the first
gradients it computes can be huge or just noisy. If the [[learning-rate]] (the "gas pedal") is
already at full strength, one bad early update can throw the model somewhere awful and training
becomes unstable. **Warmup** says: begin with a really small learning rate and turn it up
step by step over the first stretch of training. Once the model has "found its footing", you
use the full rate. Raschka's book uses it as one of three "bells and whistles" that make LLM
training more stable (the other two are [[cosine-decay]] and [[gradient-clipping]]).

## 🟡 Mechanics (Practitioner)
**The idea in words.** For the first $T_{\text{warm}}$ steps, raise the learning rate in a
straight line from a small `initial_lr` up to the maximum `peak_lr`. After that, hold it at
the peak (or hand over to a decay, see [[cosine-decay]]).

**Step 1 — how much to add each step:**
$$\Delta = \frac{\eta_{\text{peak}} - \eta_{\text{init}}}{T_{\text{warm}}}$$

Reading it symbol by symbol:
- $\Delta$ ("delta") = the increment, i.e. how much the learning rate grows per step (`lr_increment` in code).
- $\eta$ (Greek "eta") = a learning rate. $\eta_{\text{peak}}$ = the top value (`peak_lr`), $\eta_{\text{init}}$ = the starting value (`initial_lr`).
- $T_{\text{warm}}$ = the number of warmup steps (`warmup_steps`).
- So: "the total climb, split into equal slices — one slice per warmup step."

**Step 2 — the schedule:**
$$\eta(t) = \begin{cases} \eta_{\text{init}} + t\,\Delta & \text{if } t < T_{\text{warm}} \\[2pt] \eta_{\text{peak}} & \text{otherwise} \end{cases}$$

- $\eta(t)$ = "the learning rate used at step $t$" — the learning rate is now a *function* of time.
- $t$ = the global step counter (`global_step`), counted across all epochs, starting at 0.
- The big curly brace means "pick the line whose condition is true". Top line: still warming up, so start value plus $t$ slices. Bottom line: warmup over, use the peak.
- This is a straight-line ramp followed by a flat line — a *piecewise linear* function.

**How many warmup steps?** The book's rule of thumb: usually **0.1% to 20% of the total
number of steps** (book p. 315). Total steps $=$ (batches per epoch) $\times$ (epochs).

**Worked numbers from the book (§D.1).** `n_epochs = 15`, `initial_lr = 0.0001`, `peak_lr = 0.01`,
and `warmup_steps = int(0.2 * total_steps)`, which prints **27**. (That implies 135 total steps,
i.e. 9 batches per epoch on the small "The Verdict" dataset.)
- $\Delta = (0.01 - 0.0001)/27 = 0.0099/27 \approx 0.000367$
- $t = 0$: $\eta = 0.0001$
- $t = 13$: $\eta \approx 0.0001 + 13 \times 0.000367 \approx 0.00487$
- $t = 26$: $\eta \approx 0.00963$ (last warmup step)
- $t \ge 27$: $\eta = 0.01$

> ⚠️ **Book inconsistency:** the code prints 27, but the prose ("we have 20 warmup steps"), a code
> annotation and the Figure D.1 caption all say **20** warmup steps. The plotted curve peaks at about
> step 27, matching the code — read "20" as a typo for 27 (book pp. 315–316).

**PyTorch code (book p. 315).** PyTorch lets you overwrite an optimizer's learning rate at
any time by writing into each `param_group["lr"]`:

```python
n_epochs = 15
initial_lr = 0.0001
peak_lr = 0.01
total_steps = len(train_loader) * n_epochs
warmup_steps = int(0.2 * total_steps)      # 20% warmup -> 27

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1
track_lrs = []
for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1
        if global_step < warmup_steps:       # still warming up
            lr = initial_lr + global_step * lr_increment
        else:
            lr = peak_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr          # apply the new rate
        track_lrs.append(optimizer.param_groups[0]["lr"])
        # loss calculation and model updates omitted for simplicity
```

Notes on the code: `global_step` starts at −1 so the first step is 0; the optimizer is
[[adamw-optimizer|AdamW]] with `weight_decay=0.1` and no `lr` argument, because the loop sets the rate
by hand. Plotting `track_lrs` gives **Figure D.1** (book p. 316): a straight ramp from ~0.0001 to
0.01, then flat for the rest of the ~135 steps.

**In the full training function (§D.4, book pp. 319–321).** Warmup is combined with
[[cosine-decay]] and [[gradient-clipping]] in `train_model(..., warmup_steps, initial_lr=3e-05, min_lr=1e-6)`.
There `peak_lr` is read from the optimizer (`optimizer.param_groups[0]["lr"]`), and the book runs it with
`peak_lr = 0.001`, `initial_lr = 1e-5`, `warmup_steps = 27`, 15 epochs. Clipping is switched on **only
after** warmup (`if global_step >= warmup_steps`).

**Where it is used.** Chapter 5 keeps its training loop simple and only *names* warmup, cosine
annealing and gradient clipping, pointing to appendix D (book p. 146). The appendix B reading list adds
that linear warmup plus cosine annealing also work for **continued pretraining** of already-pretrained LLMs
(Ibrahim et al., 2024; book p. 294).

## 🔴 Theory (Expert)
- **Why early steps are dangerous.** In plain [[stochastic-gradient-descent|SGD]] the update is
  $\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L$; its length is $\eta_t\,\lVert\nabla_\theta L\rVert$
  (a [[norm]]). At random initialisation the [[gradient]] norm and the local curvature can both be large,
  so a fixed large $\eta$ can exceed the stability limit $\eta < 2/\lambda_{\max}$ of the local quadratic
  model ($\lambda_{\max}$ = largest [[hessian]] eigenvalue) and blow up. Ramping $\eta$ lets the network move
  into flatter regions first. (Standard optimisation background, not stated in the book.)
- **Adaptive optimisers.** For Adam/[[adamw-optimizer|AdamW]], the second-moment estimate $\hat v_t$ is
  built from very few samples in the first steps, so the per-parameter scaling $1/\sqrt{\hat v_t}$ is noisy;
  warmup is the common remedy (background; the book only gives the stability argument).
- **Boundary detail of the book's code.** The last warmup step $t = T_{\text{warm}}-1$ uses
  $\eta_{\text{peak}} - \Delta$; the peak is first used at $t = T_{\text{warm}}$. The schedule is continuous at the
  switch point to within one increment.
- **Variants.** Linear warmup is the usual choice; exponential or constant ("low then jump") warmups exist.
  Warmup length is a [[hyperparameter]]; the book's 0.1%–20% range spans tiny-fraction warmups used for huge
  pretraining runs and generous ones for small datasets.
- **Relation to the schedule as a whole.** Warmup is the first piece of a
  [[learning-rate-schedule]] $\eta: \{0,\dots,T-1\} \to \mathbb R_{>0}$; it is usually followed by a decay such
  as [[cosine-decay]].

## Connections
- Builds on: [[learning-rate]], [[gradient-descent]], [[stochastic-gradient-descent]], [[gradient]], [[hyperparameter]], [[epoch]]
- Leads to: [[learning-rate-schedule]], [[cosine-decay]], [[gradient-clipping]]
- Used by: [[llm-training-loop]], [[pretraining]], [[adamw-optimizer]], [[training-loop]]
- Related: [[vanishing-and-exploding-gradients]], [[weight-decay]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D (intro) and §D.1 Learning rate warmup, book pp. 313–316 (Figure D.1, p. 316).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.2, book pp. 316–317 (warmup + cosine combined), §D.4, book pp. 319–321 (`train_model`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2, book p. 146 (pointer to appendix D).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, chapter 5 references, book p. 294 (Ibrahim et al. 2024, continued pretraining).
