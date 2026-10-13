---
title: Cosine Decay (Cosine Annealing)
type: concept
tags: [optimization, deep-learning, llm, training]
difficulty: intermediate
sources: [llm-appD-training-loop-bells-and-whistles, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Cosine Decay (Cosine Annealing)

> **One-liner:** After warmup, you lower the learning rate smoothly along the shape of half a cosine wave, so training ends with small, careful steps.

*Also called **cosine annealing** (the name used in the book's code annotations).*

## 🟢 Intuition (Dummy)
Imagine rolling a marble into the bottom of a bowl. At first you push it firmly so it gets
moving. But as it nears the bottom, if you keep pushing just as hard it shoots past the bottom and
up the other side, back and forth. To make it settle, you push more and more gently.

Training works the same way. The [[learning-rate]] is the push. Big steps are great early on, but
near the end they make the model **overshoot** the low point of the loss. Cosine decay turns the push
down gradually: slowly at first, faster in the middle, then gently again as it approaches a small floor
value. The curve looks like a smooth slide — half of a cosine wave. Raschka's book says this slower
update speed lowers the risk of overshooting the loss minima and keeps the late phase of training stable
(book p. 316).

## 🟡 Mechanics (Practitioner)
The book always pairs cosine decay with [[learning-rate-warmup]]: warm up for $T_w$ steps, then decay.

**Step 1 — how far along the decay phase are we?**
$$p_t = \frac{t - T_w}{T - T_w}$$
- $p_t$ = the "progress" (`progress` in code), a number from 0 (decay just started) towards 1 (training finished).
- $t$ = the current global step (`global_step`).
- $T_w$ = number of warmup steps (`warmup_steps`); $T$ = total training steps (`total_training_steps`).
- In words: "steps done since warmup ended, divided by the number of steps the decay phase lasts."

**Step 2 — the cosine-shaped learning rate:**
$$\eta_t = \eta_{\min} + \left(\eta_{\text{peak}} - \eta_{\min}\right)\cdot \frac{1}{2}\Big(1 + \cos(\pi\, p_t)\Big)$$
- $\eta_t$ = the learning rate at step $t$ (`lr`).
- $\eta_{\min}$ = the floor it decays towards (`min_lr`); $\eta_{\text{peak}}$ = the top value reached after warmup (`peak_lr`).
- $(\eta_{\text{peak}} - \eta_{\min})$ = the total "height" the rate will fall.
- $\cos(\pi p_t)$ = the cosine of an angle that sweeps from $0$ to $\pi$ (that is 0° to 180°) as progress goes from 0 to 1. Over that range the cosine falls from $+1$ to $-1$ — exactly **half a cosine cycle**.
- $\tfrac12(1 + \cos(\cdot))$ = a squeeze that turns the range $[-1, 1]$ into $[0, 1]$.
- Altogether: "start at the floor, then add back a fraction of the height; that fraction slides from 1 down to 0 along a cosine curve."

**Sanity checks at the end points.**

| progress $p_t$ | $\cos(\pi p_t)$ | factor $\tfrac12(1+\cos)$ | learning rate |
|---|---|---|---|
| 0 (decay starts) | 1 | 1 | $\eta_{\text{peak}}$ — smooth hand-over from warmup |
| 0.5 (halfway) | 0 | 0.5 | midpoint of $\eta_{\min}$ and $\eta_{\text{peak}}$ |
| → 1 (end) | −1 | 0 | $\eta_{\min}$ |

**Worked numbers from the book (§D.2).** Reusing D.1's settings: `initial_lr = 0.0001`, `peak_lr = 0.01`,
15 epochs, `warmup_steps = 27`, about 135 total steps. The floor is
`min_lr = 0.1 * initial_lr` $= 0.00001$ (book p. 316). The decay phase is $135-27 = 108$ steps long.
- $t = 27$: $p = 0$, $\eta = 0.01$
- $t = 81$: $p = 54/108 = 0.5$, $\eta = 0.00001 + 0.00999 \times 0.5 \approx 0.0050$
- $t = 134$ (last step): $p = 107/108 \approx 0.991$, $\eta$ just above $0.00001$ — close to zero

**PyTorch code (book pp. 316–317):**

```python
import math

min_lr = 0.1 * initial_lr
track_lrs = []
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1

for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1
        if global_step < warmup_steps:            # applies linear warmup
            lr = initial_lr + global_step * lr_increment
        else:                                     # uses cosine annealing after warmup
            progress = ((global_step - warmup_steps) /
                        (total_training_steps - warmup_steps))
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])
```

Plotting `track_lrs` gives **Figure D.2** (book p. 317): a straight rise to 0.010 at about step 27, then
an S-shaped half-cosine fall to nearly zero by about step 135.

> ⚠️ **Book inconsistency:** the text and the Figure D.2 caption describe **20** warmup steps, while the
> code prints 27 and the plot peaks at about step 27. The same 20-vs-27 slip appears in §D.1 (book pp. 315–317).

**In the full training function (§D.4, book pp. 319–321).** `train_model` uses exactly this
warmup-then-cosine logic, with defaults `initial_lr=3e-05`, `min_lr=1e-6`, and `peak_lr` taken from the
optimizer. The book's run uses `peak_lr = 0.001`, `initial_lr = 1e-5`, `min_lr = 1e-5`, 15 epochs; losses go
from 10.934 (train) / 10.939 (val) at step 0 to 0.035 / 6.938 at step 130 — the tiny dataset still overfits,
but the function works (book p. 321).

**Where it is used.** Chapter 5 names cosine annealing as an "advanced technique" covered in appendix D
(book p. 146). Appendix B cites Ibrahim et al. (2024) showing warmup + cosine annealing also helps
**continued pretraining** of already-pretrained LLMs (book p. 294).

## 🔴 Theory (Expert)
- **Why decaying step sizes help.** With stochastic gradients, SGD with a constant $\eta$ does not
  converge to a minimiser; it hovers in a noise ball whose radius scales with $\eta$. Classic
  Robbins–Monro conditions $\sum_t \eta_t = \infty,\ \sum_t \eta_t^2 < \infty$ guarantee convergence;
  practical schedules like cosine decay approximate this "shrink to small" behaviour over a finite horizon $T$.
  (Background; the book gives only the overshooting argument.)
- **Shape properties.** $\tfrac12(1+\cos\pi p)$ is [[monotonic-function|monotonically decreasing]] on $[0,1]$
  with zero slope at both ends ($\tfrac{d}{dp} = -\tfrac{\pi}{2}\sin\pi p$, which is 0 at $p=0$ and $p=1$).
  So the rate lingers near the peak at first, drops fastest at $p = \tfrac12$, and lingers near the floor at the end —
  unlike linear decay, which has constant slope.
- **Affine structure.** The formula is a composition of affine rescalings: $p \mapsto \pi p$ (angle),
  $x \mapsto \tfrac12(1+x)$ maps $[-1,1]\to[0,1]$, and $y \mapsto \eta_{\min} + (\eta_{\text{peak}}-\eta_{\min})y$
  maps $[0,1] \to [\eta_{\min},\eta_{\text{peak}}]$.
- **Boundary detail of the book's code.** Since $t$ ends at $T-1$, $p_t \le (T-1-T_w)/(T-T_w) < 1$, so $\eta_{\min}$ is
  approached but never exactly reached.
- **Origin and variants.** Cosine annealing is usually credited to Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent
  with Warm Restarts* (2017), which also restarts the cosine periodically (background, not cited in the book).
  PyTorch ships it as `torch.optim.lr_scheduler.CosineAnnealingLR`; the book implements it by hand for transparency.

## Connections
- Builds on: [[learning-rate]], [[learning-rate-warmup]], [[gradient-descent]], [[stochastic-gradient-descent]], [[monotonic-function]]
- Leads to: [[learning-rate-schedule]]
- Used by: [[llm-training-loop]], [[pretraining]], [[training-loop]], [[adamw-optimizer]]
- Related: [[gradient-clipping]], [[overfitting]], [[hyperparameter]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.2 Cosine decay, book pp. 316–317 (Figure D.2, p. 317).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1, book pp. 314–316 (reused hyperparameters); §D.4, book pp. 319–321 (`train_model`, run output).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2, book p. 146 (pointer to appendix D).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, chapter 5 references, book p. 294 (Ibrahim et al. 2024).
