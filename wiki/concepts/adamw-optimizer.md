---
title: AdamW Optimizer
type: concept
tags: [optimization, deep-learning, regularization, pytorch, llm]
difficulty: advanced
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# AdamW Optimizer

> **One-liner:** AdamW is the rule that decides how much to nudge each weight after every batch — it gives every weight its own adaptive step size and also gently shrinks all weights toward zero.

*Also known as:* AdamW, Adam with decoupled weight decay. *Optimizer* in general = the object that turns gradients into weight updates.

## 🟢 Intuition (Dummy)
Training a model is like walking downhill in fog to reach the lowest valley. The [[gradient]] tells you which way is downhill *right here*. An **optimizer** decides how big a step to take.

Plain [[gradient-descent]] takes the same-sized stride for every direction. **Adam** is smarter, in two ways:
1. **Momentum** — it remembers which way you've been going lately, so random bumps in the ground don't throw you off. Like a heavy ball rolling downhill.
2. **Per-weight step sizes** — for weights whose gradient is always big and jumpy, it takes small careful steps; for weights whose gradient is small but steady, it takes bigger steps. Each of the millions of weights gets its own personal "speed setting."

**AdamW** adds one more habit: after every step, it shrinks each weight by a tiny percent — like a slow leak. Weights only stay big if the data keeps pushing them up. That keeps the model simple and helps against [[overfitting]]. That's the "W": **weight decay**, done the right way.

Raschka's sidebar puts it simply: Adam is popular; AdamW "improves the weight decay approach"; so "AdamW is frequently used in the training of LLMs" (§5.2, book p. 148).

## 🟡 Mechanics (Practitioner)

### In the book: one line of code
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
```
(§5.2, book p. 149). `model.parameters()` = all trainable weights; `lr` = [[learning-rate]] $\eta$; `weight_decay` = $\lambda$ ([[weight-decay]]). Then inside the loop (book p. 147):
```python
optimizer.zero_grad()   # clear old gradients (PyTorch accumulates them)
loss = calc_loss_batch(input_batch, target_batch, model, device)
loss.backward()         # compute gradients g_t
optimizer.step()        # AdamW update of every weight
```

### The update rule
The book doesn't print the formulas; these are the standard ones (Kingma & Ba 2014 for Adam; Loshchilov & Hutter 2019 for AdamW), matching what `torch.optim.AdamW` does. For each weight $\theta$ at step $t$, with gradient $g_t = \partial \mathcal L / \partial \theta$:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\right)$$

Symbol by symbol:
- $g_t$ = the gradient for this weight at step $t$ (what `loss.backward()` stores in `.grad`).
- $m_t$ = the **first moment**: a running average of recent gradients (the "momentum" — which way we've been going). Starts at $m_0 = 0$.
- $v_t$ = the **second moment**: a running average of *squared* gradients (how big/jumpy the gradient usually is). Starts at $v_0 = 0$.
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ = how much memory each average keeps (PyTorch defaults; the book never changes them). $\beta_1=0.9$ ≈ "average over the last ~10 steps", $\beta_2 = 0.999$ ≈ "last ~1,000 steps".
- $\hat m_t$, $\hat v_t$ = **bias-corrected** averages. Because $m_0=v_0=0$, early averages are too small; dividing by $1-\beta^t$ fixes that.
- $\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ = the Adam step: direction from momentum, scaled down by the typical gradient size. This is the "per-weight learning rate."
- $\epsilon = 10^{-8}$ = a tiny number so we never divide by zero ([[numerical-stability]]).
- $\eta$ = learning rate (`lr`).
- $\lambda\,\theta_{t-1}$ = the **decoupled weight decay** term: pull the weight toward 0 in proportion to its size.

### Worked example: the very first step
One weight $\theta_0 = 1.0$, first gradient $g_1 = 0.5$, book's Ch. 5 settings $\eta = 0.0004$, $\lambda = 0.1$:
- $m_1 = 0.9\cdot 0 + 0.1\cdot 0.5 = 0.05$; $\hat m_1 = 0.05/(1-0.9) = 0.5$
- $v_1 = 0.999\cdot 0 + 0.001\cdot 0.25 = 0.00025$; $\hat v_1 = 0.00025/(1-0.999) = 0.25$, so $\sqrt{\hat v_1} = 0.5$
- Adam step: $0.5 / 0.5 = 1$ (ignoring $\epsilon$)
- Decay term: $\lambda\theta_0 = 0.1$
- $\theta_1 = 1.0 - 0.0004\,(1 + 0.1) = 1.0 - 0.00044 = \mathbf{0.99956}$

Notice the Adam part came out as exactly $1\times\eta$: on step 1, $\hat m_1/\sqrt{\hat v_1} = g_1/|g_1| = \pm 1$, **whatever** the gradient's size. Adam's steps are roughly scale-free — that's why the same `lr` works across very different layers.

### Every AdamW configuration in the book

| Where | Task | `lr` | `weight_decay` | epochs |
|---|---|---|---|---|
| §5.2, p. 149 | pretrain GPT-2 124M config on "The Verdict" | 0.0004 | 0.1 | 10 |
| §5.4, p. 160 | resume from checkpoint | 5e-4 | 0.1 | (+1, Ex. 5.4) |
| §6.7, p. 197 | spam classification fine-tune | 5e-5 | 0.1 | 5 |
| §7.6, p. 231 | instruction fine-tune GPT-2 medium 355M | 0.00005 | 0.1 | 2 |
| §D.1–D.4, pp. 315–320 | warmup + cosine schedule | default (1e-3) used as peak, overwritten every step | 0.1 | 15 |
| §E.4, p. 334 | LoRA fine-tune | 8e-4 | 0.1 | 5 |

Patterns: **`weight_decay=0.1` everywhere**; fine-tuning a pretrained model uses a ~10× smaller `lr` (5e-5) than training from scratch (4e-4); LoRA, which trains only small new matrices, can use a larger one (8e-4). Appendix A, by contrast, uses plain `torch.optim.SGD(model.parameters(), lr=0.5)` for its toy network (book p. 274).

In Appendix D the learning rate is set by hand at each step by writing into `optimizer.param_groups` (book pp. 315, 320):
```python
for param_group in optimizer.param_groups:
    param_group["lr"] = lr
```
— this is how [[learning-rate-warmup]] and [[cosine-decay]] are layered on top of AdamW. AdamW itself still does the per-weight scaling.

### The optimizer has memory — save it! (§5.4, book p. 160)
"Adaptive optimizers such as AdamW store additional parameters for each model weight. AdamW uses historical data to adjust learning rates for each model parameter dynamically." Those extra numbers are $m$ and $v$. If you reload only the model and start a fresh optimizer, "the optimizer resets, and the model may learn suboptimally or even fail to converge properly, which means it will lose the ability to generate coherent text." The reload code re-creates `torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)` and then calls `optimizer.load_state_dict(...)` (book p. 160; Exercise 5.4 solution, Appendix C p. 305). Note that this `lr=5e-4` differs from the `lr=0.0004` of the §5.2 run; the book does not comment on it. So the book saves both ([[model-checkpointing]]):
```python
torch.save({"model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()},
           "model_and_optimizer.pth")
```

**Memory cost (derived).** Two extra fp32 numbers per weight. The untied GPT model of Ch. 4–5 has 163,009,536 parameters (621.83 MB, §4.6, book pp. 120–122), so AdamW's state adds $2\times 621.83 \approx 1{,}244$ MB — the optimizer state is **twice** the size of the model itself. See [[parameter-count-and-memory]].

## 🔴 Theory (Expert)

**Adam as diagonal preconditioning.** Write the Adam step as $\theta_t = \theta_{t-1} - \eta\, D_t^{-1}\hat m_t$ with $D_t = \operatorname{diag}(\sqrt{\hat v_t}+\epsilon)$. $\hat m_t$ is an exponential moving average (EMA) of gradients ([[expected-value-and-mean]]) — momentum with effective window $1/(1-\beta_1)$. $\hat v_t$ estimates the uncentered second moment $\mathbb E[g^2]$ ([[covariance-and-variance]]) per coordinate. Dividing by $\sqrt{\hat v_t}$ approximates a diagonal, sign-preserving rescaling of the gradient: the step magnitude per coordinate is bounded by roughly $\eta$ (exactly $\eta$ at $t=1$ as computed above), independent of the gradient's scale. This invariance to per-parameter gradient scale is what makes Adam robust for transformers, whose embeddings, attention projections and LayerNorm gains have gradients differing by orders of magnitude.

**Bias correction.** With $m_0 = 0$, $\mathbb E[m_t] = (1-\beta_1^t)\,\mathbb E[g]$ if gradients are stationary, so $\hat m_t = m_t/(1-\beta_1^t)$ is unbiased; same for $v_t$. Without it, the ratio $m_t/\sqrt{v_t}$ would be off by a factor $(1-\beta_1^t)/\sqrt{1-\beta_2^t}$ — at $t=1$ that is $0.1/\sqrt{0.001}\approx 3.16$, so the first steps would be about **3× too large** (the $v_t$ bias, which makes the denominator too small, dominates) — exactly the early instability Kingma & Ba's correction removes. Warmup ([[learning-rate-warmup]]) addresses a related instability: $\hat v_t$ is a high-variance estimate when $t$ is small.

**Why "decoupled" matters.** In SGD, adding an $\ell_2$ penalty $\frac{\lambda}{2}\lVert\theta\rVert_2^2$ to the loss is *identical* to weight decay: the gradient gains $\lambda\theta$, giving $\theta \leftarrow (1-\eta\lambda)\theta - \eta g$. In Adam it is **not**: the $\lambda\theta$ term passes through the preconditioner, becoming $\eta\,\lambda\theta/\sqrt{\hat v_t}$, so weights with large historical gradients are barely regularized, and those with small gradients are over-regularized. Loshchilov & Hutter ("Decoupled Weight Decay Regularization", ICLR 2019) showed this coupling explains Adam's weaker generalization vs SGD, and fixed it by applying $\lambda\theta$ *outside* the adaptive scaling. That is precisely the book's "improves the weight decay approach … more effective regularization and better generalization" (book p. 148). Details in [[weight-decay]].

**PyTorch specifics (background).** `torch.optim.AdamW` applies $\theta \leftarrow \theta(1-\eta\lambda)$ *then* the Adam step — the same to first order as the formula above. Its defaults are `lr=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`; the book always overrides `weight_decay` to 0.1. The book passes `model.parameters()` directly, so biases and LayerNorm scale/shift are decayed too; many LLM codebases exclude those via parameter groups. In the original paper, $\lambda$ is multiplied by a schedule multiplier rather than by $\eta$ itself; PyTorch's coupling to `lr` means changing `lr` (e.g. during cosine decay) also changes the effective decay.

**Not a descent method on a fixed objective.** AdamW doesn't minimize "loss + penalty" for any fixed penalty; it's best viewed as a preconditioned momentum method with a separate shrinkage operator. Convergence guarantees for Adam in the non-convex setting require care (Reddi et al., 2018, constructed counterexamples and proposed AMSGrad), but in practice AdamW + warmup + cosine decay is the default recipe for LLM pretraining and fine-tuning.

**Beyond AdamW.** Appendix B lists GaLore (Zhao et al. 2024), which keeps the training loop unchanged and simply swaps PyTorch's AdamW for `GaLoreAdamW`, with the stated aim of making LLM pretraining more efficient (book p. 294). *(Background, not in the book:* GaLore projects gradients onto a low-rank subspace, which shrinks exactly the optimizer-state memory computed above.) See [[galore]], [[low-rank-approximation]]. [[jimmy-lei-ba]], co-author of Adam, is also the first author of "Layer Normalization" (Ba, Kiros & Hinton 2016; Appendix B, Ch. 4 references, book p. 292).

## Connections
- Builds on: [[gradient-descent]], [[stochastic-gradient-descent]], [[gradient]], [[backpropagation]], [[learning-rate]], [[expected-value-and-mean]], [[covariance-and-variance]]
- Leads to: [[weight-decay]], [[learning-rate-schedule]], [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]], [[galore]]
- Used by: [[llm-training-loop]], [[training-loop]], [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Related: [[regularization]], [[model-checkpointing]], [[parameter-count-and-memory]], [[numerical-stability]], [[hyperparameter]], [[pytorch]]
- People: [[jimmy-lei-ba]], [[jiawei-zhao]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 (AdamW sidebar, Listing 5.3), book pp. 146–149.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.4, book pp. 159–160 — per-weight optimizer state, checkpointing; §5.5 summary, book p. 168.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7, book p. 197; §7.6, book p. 231; §E.4, book p. 334 — fine-tuning configurations.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1, book p. 252 (optimizers as PyTorch building blocks); §A.7, book pp. 274–276 (SGD contrast).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.1, book p. 315; §D.2, book pp. 316–317; §D.4, book pp. 319–321 — manual `param_groups` learning rates.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book p. 294 (GaLore); Appendix C Exercise 5.4, book p. 305.
- Formulas: Kingma & Ba (2014), "Adam"; Loshchilov & Hutter (2019), "Decoupled Weight Decay Regularization" — not printed in the book.
