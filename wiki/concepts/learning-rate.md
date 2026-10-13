---
title: Learning Rate
type: concept
tags: [optimization, training, hyperparameter, deep-learning]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles, mml-ch07-continuous-optimization, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Learning Rate

> **One-liner:** The learning rate is the size of the step the model takes each time it adjusts its weights to reduce its mistakes.

## 🟢 Intuition (Dummy)
You are walking down a foggy hill at night, trying to reach the bottom. You can feel which way
the ground slopes under your feet (that's the [[gradient]]), but you have to decide how big a
step to take.

- **Tiny steps:** you will get there safely, but it takes forever.
- **Huge steps:** you may leap right over the valley and land on the other side, or even climb
  higher. Training "blows up".
- **Just right:** you get down quickly and settle at the bottom.

That step size is the **learning rate**. It is not learned by the model; *you* pick it, so it is a
[[hyperparameter]]. The LLM book describes it as "a tunable setting that we must experiment with
based on observing the loss" (§A.7, book p. 275). For big models like LLMs people also *change*
the step size during training: start gently (warmup), then gradually take smaller steps (decay).

## 🟡 Mechanics (Practitioner)

### Where it appears in the update
Plain [[stochastic-gradient-descent]] (what `optimizer.step()` does for `torch.optim.SGD`):
$$\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}(\theta)$$
Symbol by symbol:
- $\theta$ ("theta") = all trainable parameters (every weight and bias).
- $\leftarrow$ = "replace with" (an update, not an equation).
- $\eta$ ("eta") = the **learning rate**, a small positive number (`lr` in PyTorch).
- $\nabla_\theta \mathcal{L}$ = the gradient of the loss with respect to the parameters, the vector of
  [[partial-derivative|partial derivatives]] that `loss.backward()` puts in each parameter's `.grad`.
- $-$ = go *against* the gradient (downhill).

The book says it in words: SGD works by "multiplying the gradients with the learning rate and
adding the scaled negative gradient to the parameters" (book p. 276). MML writes the same thing
with step-size $\gamma$ instead of $\eta$ (see [[gradient-descent]]).

**Tiny numeric example (one step).** In the logistic-regression demo of §A.4, $\partial L/\partial w_1 = -0.0898$
with $w_1 = 2.2$ (book p. 264). With $\eta = 0.5$: $w_1 \leftarrow 2.2 - 0.5\times(-0.0898) = 2.2449$. With $\eta = 0.01$ the
same gradient only moves it to $2.2009$. (Illustration computed by hand; the book does not do this step.)

### The book's first training run (Listing A.9, book pp. 274–275)
```python
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)   # learning rate 0.5
num_epochs = 3
```
Training loss per batch: 0.75, 0.65, 0.44, 0.13, 0.03, 0.00 — "a sign that the model converged on the
training set" (book p. 275). "Ideally, we want to choose a learning rate such that the loss converges after
a certain number of epochs"; the number of epochs is another hyperparameter. Tune both on a **validation set**, and
touch the test set only once (book p. 275; see [[train-validation-test-split]]).

### Learning rates used across the LLM book
| where | model / task | optimizer | `lr` |
|---|---|---|---|
| §A.7, p. 275 | toy MLP, 2 classes | SGD | $0.5$ |
| §5.2, p. 149 | GPT-2 124M pretraining on "The Verdict" | AdamW, `weight_decay=0.1` | $4\times10^{-4}$ |
| §5.4, p. 160 | resume from checkpoint | AdamW, `weight_decay=0.1` | $5\times10^{-4}$ |
| §6.7, p. 197 | spam [[classification-fine-tuning]] | AdamW, `weight_decay=0.1` | $5\times10^{-5}$ |
| §7.6, p. 231 | [[instruction-fine-tuning]] | AdamW, `weight_decay=0.1` | $5\times10^{-5}$ |
| App. D.1, p. 315 | warmup demo | AdamW, `weight_decay=0.1` | $10^{-4} \to 10^{-2}$ |

Pattern: fine-tuning a pretrained model uses a **10× smaller** step than pretraining, so the useful
pretrained weights are only nudged, not wrecked.

### Learning-rate warmup (Appendix D.1, book pp. 313–316)
A [[learning-rate-schedule]] makes $\eta$ depend on the step $t$. Warmup raises it linearly from
`initial_lr` to `peak_lr`:
$$\Delta = \frac{\eta_{\text{peak}} - \eta_{\text{init}}}{T_{\text{warm}}},\qquad
\eta(t) = \begin{cases}\eta_{\text{init}} + t\,\Delta & t < T_{\text{warm}}\\ \eta_{\text{peak}} & \text{otherwise}\end{cases}$$
- $t$ = `global_step`, the training-step counter across all epochs, starting at 0.
- $\eta_{\text{init}} = 0.0001$, $\eta_{\text{peak}} = 0.01$ (book p. 315).
- $T_{\text{warm}}$ = `warmup_steps` $= \lfloor 0.2 \times T_{\text{total}} \rfloor = 27$, where $T_{\text{total}}$ = batches per epoch × `n_epochs` (15). The book's rule of thumb: warmup is usually 0.1% to 20% of all steps.
- $\Delta = 0.0099/27 \approx 3.667\times10^{-4}$ per step; $\eta(0) = 0.0001$, $\eta(26) \approx 0.009633$, then $0.01$ flat.

```python
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
lr_increment = (peak_lr - initial_lr) / warmup_steps
global_step = -1
track_lrs = []
for epoch in range(n_epochs):
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()
        global_step += 1
        if global_step < warmup_steps:
            lr = initial_lr + global_step * lr_increment
        else:
            lr = peak_lr
        for param_group in optimizer.param_groups:   # how PyTorch changes lr at runtime
            param_group["lr"] = lr
        track_lrs.append(optimizer.param_groups[0]["lr"])
```
Why warm up: early weights are random and gradients can be large or noisy; starting with smaller weight updates
"decreases the risk of the model encountering large, destabilizing updates" and can stabilize LLM training (book p. 314).

> ⚠️ **Inconsistency in the book (not between sources):** the code prints `warmup_steps = 27`, but the
> prose, a code annotation, and the Figure D.1 caption say "20 warmup steps". The plotted ramp peaks near
> step 27, matching the code. Read "20" as a typo (book pp. 315–316).

After warmup the book lowers $\eta$ again with [[cosine-decay]] (D.2) and protects updates with
[[gradient-clipping]] (D.3); see [[learning-rate-warmup]] for the full schedule.

## 🔴 Theory (Expert)
- **Stability bound (quadratic case).** For $f(\theta) = \tfrac12\theta^\top H\theta$ with [[hessian]] $H \succ 0$, gradient
  descent gives $\theta_{t+1} = (I - \eta H)\theta_t$, which converges iff $|1 - \eta\lambda_i| < 1$ for every eigenvalue
  $\lambda_i$, i.e. $0 < \eta < 2/\lambda_{\max}$. The best fixed step is $\eta^* = 2/(\lambda_{\min} + \lambda_{\max})$, and the
  convergence speed depends on the condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ (see
  [[eigenvalues-and-eigenvectors]]). For an $L$-smooth function, $\eta \le 1/L$ guarantees descent.
- **Stochastic setting.** With noisy minibatch gradients, a constant $\eta$ only reaches a noise ball around
  the minimum whose size scales with $\eta$. Robbins–Monro conditions $\sum_t\eta_t = \infty$, $\sum_t\eta_t^2 < \infty$
  guarantee convergence, which motivates decaying schedules (see [[stochastic-gradient-descent]]).
- **Adaptive optimizers rescale $\eta$ per parameter.** [[adamw-optimizer]] uses
  $\theta \leftarrow \theta - \eta\big(\hat m_t/(\sqrt{\hat v_t}+\epsilon) + \lambda\theta\big)$: the effective step for weight $j$ is
  roughly $\eta/\sqrt{\hat v_{t,j}}$, so $\eta$ sets a global scale rather than an exact step. The book notes that
  AdamW "uses historical data to adjust learning rates for each model parameter dynamically" (§5.4, p. 160).
  Because Adam's $\hat v_t$ is unreliable in the first steps, warmup is especially helpful for Adam-type
  optimizers in transformers.
- **Interaction with batch size.** Larger batches reduce gradient noise; a common heuristic scales $\eta$
  linearly with batch size (with warmup), up to a critical batch size.
- **Interaction with weight decay.** In AdamW the decay term is multiplied by $\eta$, so changing $\eta$ also
  changes the effective regularization strength (see [[weight-decay]], [[regularization]]).

## Connections
- Builds on: [[gradient]], [[partial-derivative]], [[gradient-descent]], [[loss-function]]
- Leads to: [[learning-rate-schedule]], [[learning-rate-warmup]], [[cosine-decay]], [[adamw-optimizer]]
- Used by: [[stochastic-gradient-descent]], [[training-loop]], [[llm-training-loop]], [[pretraining]], [[fine-tuning]]
- Related: [[hyperparameter]], [[epoch]], [[mini-batch]], [[weight-decay]], [[gradient-clipping]], [[training-and-validation-loss]], [[train-validation-test-split]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7, book pp. 274–276 (Listing A.9, `SGD(lr=0.5)`; learning rate as hyperparameter; validation set; SGD update in words).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix D intro + §D.1, book pp. 313–316 (warmup; `initial_lr=0.0001`, `peak_lr=0.01`, `warmup_steps=27`; Figure D.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 p. 149 (`lr=0.0004`), §5.4 p. 160 (`lr=5e-4`; AdamW per-parameter adjustment), §6.7 p. 197 (`lr=5e-5`), §7.6 p. 231 (`lr=0.00005`).
- `raw/mml-book.pdf` §7.1, printed pp. 227–232 (step-size $\gamma$ = learning rate).
- Stability bound, Robbins–Monro, batch-size scaling: standard optimization background, not in the LLM book.
