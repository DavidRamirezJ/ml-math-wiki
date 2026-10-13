---
title: Gradient Descent
type: concept
tags: [optimization, gradient-descent, machine-learning, deep-learning, llm]
difficulty: beginner
sources: [mml-ch07-continuous-optimization, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-06-04
updated: 2026-10-13
---

# Gradient Descent

> **One-liner:** To find the bottom of a valley you can't see, keep taking small steps in whatever direction is most steeply downhill from where you stand.

## 🟢 Intuition (Dummy)
Imagine you're on a foggy hillside and want to reach the lowest point. You can't see
far, but you can feel which way the ground tilts under your feet. So you take a step
downhill, feel again, step again. Repeat until the ground is flat — you've reached a
valley bottom.

That's gradient descent. The "objective function" is the landscape, and "training a
model" means rolling a ball down it to the lowest point. One catch: if there are
several valleys, you only reach the one whose slope you happened to be standing on. And
your step size matters — tiny steps take forever, giant steps can launch you across the
valley and back out the other side.

### In LLMs
Training a GPT model is this same foggy-hillside walk, just in a landscape with about
124 million directions (one per weight of GPT-2 small). The "height" is the
[[cross-entropy-loss]] — how badly the model predicts the next word. Each step: show the
model a small batch of text, measure how wrong it was, let PyTorch work out which way is
downhill for every weight ([[backpropagation]]), and nudge all weights a little that way.
The book does this thousands of times to pretrain its LLM (Chapter 5) and again to
fine-tune it (Chapters 6–7). Appendix D adds three safety habits for the walk:
start with tiny steps (**warmup**), take smaller steps as you near the bottom
(**cosine decay**), and never take a giant leap even if the slope looks huge
(**gradient clipping**).

## 🟡 Mechanics (Practitioner)
We want to solve $\min_{\mathbf{x}} f(\mathbf{x})$ for a differentiable
$f:\mathbb{R}^d\to\mathbb{R}$. The **gradient** $\nabla f$ points in the direction of
steepest *ascent*, so we step the opposite way. Starting from a guess $\mathbf{x}_0$,
iterate:
$$ \mathbf{x}_{i+1} = \mathbf{x}_i - \gamma_i\,\big((\nabla f)(\mathbf{x}_i)\big)^\top $$
For a small enough **step-size** $\gamma_i\ge 0$, $f(\mathbf{x}_{i+1})\le f(\mathbf{x}_i)$,
so the sequence $f(\mathbf{x}_0)\ge f(\mathbf{x}_1)\ge\cdots$ converges to a *local*
minimum. (The transpose appears because gradients are row vectors by the book's
convention; the parameters are columns.)

**Step-size = learning rate.** Too small → painfully slow; too large → overshoot,
oscillate, or diverge. Adaptive heuristic: if a step *increases* $f$, undo it and shrink
$\gamma$; if it decreases $f$, try a bigger $\gamma$ next time.

**Momentum** (Eq. 7.11) gives the ball memory so it doesn't bounce between valley walls:
$$ \mathbf{x}_{i+1} = \mathbf{x}_i - \gamma_i\big((\nabla f)(\mathbf{x}_i)\big)^\top + \alpha\,\Delta\mathbf{x}_i,\qquad \Delta\mathbf{x}_i = \mathbf{x}_i-\mathbf{x}_{i-1} $$
with $\alpha\in[0,1]$. This implements a moving average of past gradients, dampening
oscillations and averaging out noise.

**Worked example (Ex. 7.1):** minimize the quadratic
$f(\mathbf{x})=\tfrac12\mathbf{x}^\top\big[\begin{smallmatrix}2&1\\1&20\end{smallmatrix}\big]\mathbf{x}-\big[5,3\big]\mathbf{x}$
with gradient $\nabla f(\mathbf{x})=\mathbf{x}^\top\big[\begin{smallmatrix}2&1\\1&20\end{smallmatrix}\big]-\big[5,3\big]$.
From $\mathbf{x}_0=\big[-3,-1\big]^\top$ with $\gamma=0.085$, one step gives
$\mathbf{x}_1=\big[-1.98,1.21\big]^\top$, then $\mathbf{x}_2=\big[-1.32,-0.42\big]^\top$, converging to the minimum.

**ML usage:** the default training loop for almost every model. The objective is the
loss; the parameters are weights. For large data, use [[stochastic-gradient-descent]].

### In LLMs: the training loop (Build an LLM §5.2, §A.4)
The LLM book writes the same update in deep-learning notation:
$$\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal L(\theta;\ \text{batch})$$
Symbol by symbol:
- $\theta$ ("theta") = *all* trainable weights of the model, flattened into one long list
  (`model.parameters()`; ~124M numbers for GPT-2 small).
- $\leftarrow$ = "replace with" (an update, not an equation).
- $\eta$ ("eta") = the [[learning-rate]] — the same thing MML calls the step-size $\gamma$.
  The book uses `lr=0.0004`.
- $\mathcal L(\theta;\ \text{batch})$ = the loss (average cross-entropy) on the current
  [[mini-batch]] of text, as a function of the weights.
- $\nabla_\theta\mathcal L$ = the [[gradient]]: the vector of [[partial-derivative|partial
  derivatives]] $\partial\mathcal L/\partial\theta_j$, one per weight. Computed by
  `loss.backward()`.

(Notation note, not a contradiction: MML writes $\mathbf x_i - \gamma_i(\nabla f)^\top$
because it treats gradients as *row* vectors; the LLM book and PyTorch store the gradient
in the same shape as the parameter, so no transpose appears.)

**Figure 5.11's eight-step loop (book p. 146).** 1) loop over epochs → 2) loop over
batches → 3) reset gradients → 4) compute the loss → 5) backward pass → 6) update the
weights → 7) print train/val losses (optional) → 8) generate sample text (optional).
Steps 3–6 are the gradient-descent core:
```python
for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()                     # 3. reset old gradients
        loss = calc_loss_batch(input_batch, target_batch, model, device)  # 4.
        loss.backward()                           # 5. compute ∇θ L
        optimizer.step()                          # 6. θ ← θ − (step)
```
(Listing 5.3 `train_model_simple`, book p. 147.) `zero_grad()` is needed because PyTorch
*adds* new gradients to old ones by default. Training call (book p. 149):
`torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)`, 10 epochs,
`GPT_CONFIG_124M` (context length shortened to 256) — about 5 minutes on a MacBook Air.

**Worked step by hand (Appendix A, Listing A.3, book p. 264).** A one-neuron logistic
regression: $x_1=1.1$, $w_1=2.2$, $b=0$, label $y=1$.
Forward: $z=x_1w_1+b=2.42$, $a=\sigma(z)\approx0.9183$, loss
$L=-\ln a\approx0.0852$ ([[binary-cross-entropy]]). Autograd gives
$\partial L/\partial w_1=-0.0898$ and $\partial L/\partial b=-0.0817$ (book p. 264).
One plain gradient-descent step with $\eta=0.1$ (our choice, not the book's):
$$w_1 \leftarrow 2.2-0.1\cdot(-0.0898)=2.20898,\qquad b\leftarrow 0-0.1\cdot(-0.0817)=0.00817$$
New loss $\approx0.0837$ — smaller, as promised. (Negative gradient → the weight goes *up*.)

**AdamW instead of plain GD (book p. 148).** The book's optimizer is AdamW, "a variant
of Adam that improves the weight decay approach". Its standard update (not printed in
the book) is
$$\theta_t=\theta_{t-1}-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_{t-1}\Big)$$
- $\hat m_t$ = a running average of recent gradients (momentum, like MML's Eq. 7.11);
- $\hat v_t$ = a running average of *squared* gradients, so $\sqrt{\hat v_t}$ gives each
  weight its own step size;
- $\epsilon$ = a tiny number to avoid dividing by zero;
- $\lambda$ = [[weight-decay]] (`weight_decay=0.1`), which gently pulls weights toward 0.
See [[adamw-optimizer]]. Because AdamW keeps $\hat m,\hat v$ per weight, the book saves
`optimizer.state_dict()` alongside the model when checkpointing; otherwise the optimizer
"resets" and training may fail to converge properly (§5.4, book p. 160).

**Updating only some weights (layer freezing, §6.5, book p. 186).** For spam
classification the book sets `param.requires_grad = False` for every weight, then
re-enables the new 768→2 classification head, the last transformer block and the final
LayerNorm. Gradient descent then updates only $\theta_{\text{train}}$:
$\theta_{\text{train}}\leftarrow\theta_{\text{train}}-\eta\nabla_{\theta_{\text{train}}}\mathcal L$,
and the 11 frozen blocks and embeddings stay fixed ([[layer-freezing]]).

### In LLMs: the "bells and whistles" (Appendix D, book pp. 313–319)
All three make gradient descent more stable; none changes the model.

*1. Linear [[learning-rate-warmup]] (D.1).* Start with a tiny learning rate and raise it
linearly:
$$\eta(t)=\eta_{\text{init}}+t\cdot\frac{\eta_{\text{peak}}-\eta_{\text{init}}}{T_{\text{warm}}}\quad(t<T_{\text{warm}})$$
$t$ = global step counter; $\eta_{\text{init}}=10^{-4}$ (`initial_lr`);
$\eta_{\text{peak}}=10^{-2}$ (`peak_lr`); $T_{\text{warm}}$ = number of warmup steps
= 20% of all steps = **27** (code output, book p. 315; 15 epochs × 9 batches = 135 steps,
inferred). Each step adds $0.0099/27\approx3.7\times10^{-4}$. Why: at the start the weights
are random and the gradients large and noisy, so big steps could destabilize training.
Rule of thumb: warmup = 0.1%–20% of total steps (book p. 315).
> ⚠️ Book inconsistency: the prose and Figure D.1/D.2 captions say "20 warmup steps", but
> the code prints 27 and the plotted peak is at step ~27 (book pp. 315–317).

*2. [[cosine-decay]] (D.2).* After warmup, glide down a half-cosine:
$$\eta(t)=\eta_{\min}+(\eta_{\text{peak}}-\eta_{\min})\cdot\tfrac12\big(1+\cos(\pi p_t)\big),\qquad p_t=\frac{t-T_{\text{warm}}}{T-T_{\text{warm}}}$$
$p_t$ = fraction of the post-warmup phase done (0 → 1); $T$ = total steps;
$\eta_{\min}=0.1\,\eta_{\text{init}}=10^{-5}$ (book p. 316). At $p_t=0$ you get
$\eta_{\text{peak}}$, at $p_t=\tfrac12$ the midpoint, near $p_t=1$ about $\eta_{\min}$.
With the book's numbers: $\eta(27)=0.01$, $\eta(81)\approx0.0050$, $\eta(134)\approx1.2\times10^{-5}$.
Why: smaller steps late in training reduce the risk of "overshooting the loss minima"
(book p. 316) — exactly the "step too large → bounce across the valley" problem above.

*3. [[gradient-clipping]] (D.3).* If the gradient is too long, shrink it — same direction,
capped length:
$$\mathbf g\leftarrow\mathbf g\cdot\min\!\Big(1,\frac{\text{max\_norm}}{\|\mathbf g\|_2}\Big),\qquad \|\mathbf g\|_2=\sqrt{g_1^2+\cdots+g_n^2}$$
$\|\mathbf g\|_2$ = the L2 ([[norm|Euclidean]]) length of all gradients together;
`max_norm` = 1.0 in the book. Code: `torch.nn.utils.clip_grad_norm_(model.parameters(),
max_norm=1.0)` between `loss.backward()` and `optimizer.step()`. On one batch of the
124M model, the largest single gradient entry dropped from 0.0411 to 0.0185 (book p. 319).
> ⚠️ Book erratum (D.3, p. 318): the worked example $\mathbf G=\big[\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\big]$
> computes $\|\mathbf G\|_2=\sqrt{1^2+2^2+2^2+4^2}=5$; the correct value is
> $\sqrt{1+4+9+16}=\sqrt{30}\approx5.48$, so the scale factor should be $1/\sqrt{30}\approx0.183$, not $1/5$.

## 🔴 Theory (Expert)
Gradient descent is a **first-order** method: it ignores curvature. Its asymptotic
convergence rate is poor on **ill-conditioned** problems — long, thin valleys where the
gradient points nearly orthogonal to the direction toward the optimum, causing
characteristic "zigzag" (Fig. 7.3).

For the linear-system least-squares objective $\|A\mathbf{x}-\mathbf{b}\|^2$ (Ex. 7.2,
gradient $2(A\mathbf{x}-\mathbf{b})^\top A$), the convergence speed depends on the
**condition number** $\kappa = \sigma_{\max}(A)/\sigma_{\min}(A)$, the ratio of largest
to smallest [[singular-value-decomposition|singular value]] — the ratio of the most-
to least-curved direction. **Preconditioning** solves $P^{-1}(A\mathbf{x}-\mathbf{b})=0$
with $P^{-1}$ chosen so $P^{-1}A$ is better conditioned yet $P^{-1}$ is cheap.

- **Stationary points** are roots of $\nabla f=0$; classify via the **Hessian**
  $\nabla^2 f$ (positive definite → min, negative definite → max, indefinite → saddle).
- The chapter notes (via the **Abel–Ruffini theorem**, [[niels-henrik-abel]]) that
  degree $\ge 5$ polynomials have no general algebraic root formula — hence iterative
  methods even in 1-D.
- Beyond first order: **Newton/quasi-Newton** (L-BFGS) use Hessian information;
  **conjugate gradient** reuses past directions; **mirror descent** and **natural
  gradient** change the descent metric (§7.4).
- For [[convex-optimization|convex]] $f$, every local minimum is global, removing the
  starting-point dependence entirely.

### In LLMs: why the deep-learning variants look the way they do
- **Non-convexity.** The GPT loss surface is highly non-convex, so none of the convex
  guarantees above apply; in practice gradient methods still find low-loss regions, and
  "convergence" is judged empirically by the training and validation losses
  ([[training-and-validation-loss]]).
- **Stochastic gradients.** Each step uses the gradient of a mini-batch loss, an unbiased
  [[monte-carlo-estimation|Monte Carlo estimate]] of the full-data gradient — strictly,
  this is [[stochastic-gradient-descent]]. Classical SGD convergence (Robbins–Monro)
  requires a decaying step size with $\sum_t\eta_t=\infty$, $\sum_t\eta_t^2<\infty$;
  cosine decay is a practical, finite-horizon version of "shrink the step over time".
- **Adam as preconditioning.** Dividing by $\sqrt{\hat v_t}$ is a *diagonal*
  preconditioner $P^{-1}=\operatorname{diag}(1/(\sqrt{\hat v_t}+\epsilon))$ — the cheap
  version of the preconditioning idea above: it equalizes step sizes across directions
  with very different gradient scales, which is what ill-conditioning punishes.
- **Decoupled weight decay.** Plain GD on $\mathcal L+\tfrac\lambda2\|\theta\|^2$ ([[regularization|L2
  regularization]]) gives $\theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla\mathcal L$, so
  for GD "L2 penalty" and "weight decay" coincide. With Adam they do not: the penalty
  gradient $\lambda\theta$ would be divided by $\sqrt{\hat v_t}$ too. AdamW applies
  $\lambda\theta$ outside the adaptive scaling — the "improved weight decay" the book
  mentions (Loshchilov & Hutter; not cited by name in the book).
- **Why warmup helps Adam.** Early on, $\hat v_t$ is estimated from very few gradients and
  the per-weight steps are unreliable; small $\eta$ while those statistics settle avoids
  large, destabilizing updates (book p. 314 gives the stability motivation; the
  Adam-variance explanation is standard background).
- **Clipping and the step length.** For plain GD, clipping caps the step at
  $\eta\cdot\text{max\_norm}$. With AdamW the per-weight normalization means the step
  is not capped this directly; clipping mainly protects $\hat m_t,\hat v_t$ from being
  wrecked by a single exploding batch ([[vanishing-and-exploding-gradients]]). The book
  doesn't discuss this nuance.

## Connections
- Builds on: [[gradient]], [[vector]], [[partial-derivative]], [[chain-rule]], [[norm]]
- Gradients computed by: [[backpropagation]], [[automatic-differentiation]], [[computational-graph]]
- Leads to: [[stochastic-gradient-descent]], [[convex-optimization]], [[adamw-optimizer]], [[learning-rate-schedule]]
- Refinements in LLM training: [[learning-rate]], [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]], [[weight-decay]], [[layer-freezing]], [[model-checkpointing]]
- Used by: neural-network training, [[linear-regression]], [[quadratic-programming]], [[llm-training-loop]], [[pretraining]], [[classification-fine-tuning]], [[logistic-regression]]
- People: [[niels-henrik-abel]], [[sebastian-raschka]]

## Sources
- `raw/mml-book.pdf` §7.1 (Eqs. 7.4–7.12, Examples 7.1–7.2), printed pp. 227–232 (PDF pp. 232–237).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2, book pp. 146–151 (Figure 5.11 training loop, Listing 5.3 `train_model_simple`, AdamW sidebar p. 148, training run p. 149).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.4, book pp. 159–160 (saving/restoring optimizer state).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 183–190 (layer freezing, trainable subset; p. 186).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4, book pp. 263–264 (Figure A.8, Listing A.3 autograd gradients −0.0898 / −0.0817, sidebar naming gradient descent).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1–D.3, book pp. 313–319 (warmup, cosine decay, gradient clipping).
