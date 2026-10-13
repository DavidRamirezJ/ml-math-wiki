---
title: Saving and Loading Models (state_dict, Checkpointing)
type: concept
tags: [pytorch, deep-learning, training, llm, engineering]
difficulty: beginner
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Saving and Loading Models (state_dict, Checkpointing)

> **One-liner:** After you spend hours (or dollars) training a model, you write its learned numbers to a file so you can load them back later instead of training all over again.

## 🟢 Intuition (Dummy)
Think of a trained model as a huge mixing desk with millions of knobs. Training is the slow
process of turning every knob to just the right spot. If you switch the computer off, all that
knob-turning is gone, unless you take a photo of where every knob sits.

That "photo" is the **state_dict**: a list that says "knob called `layer1.weight` sits at these
values, knob called `layer1.bias` sits at these values, …". To use the model again tomorrow you:

1. build an *empty* mixing desk with exactly the same layout (same architecture), and
2. set every knob back from the photo.

If you only want to *use* the model, the knob photo is enough. If you want to *keep training* it
later, you also need a photo of the **coach's notebook**, the optimizer's memory of how each knob
has been moving lately. Model photo + coach's notebook in one file = a **checkpoint**.

The book's motivation is simple: pretraining is computationally expensive, even for the small
model in Chapter 5, so you save the model and avoid rerunning training every session (book p. 159).

## 🟡 Mechanics (Practitioner)

### What a state_dict is, in math
A neural network computes a function $f_\theta(\mathbf x)$. Two things fix that function:

- the **architecture**: the shape of the formula (which layers, in which order, with which sizes), and
- the **parameters** $\theta$: the actual numbers inside it.

$$\theta = \{W^{(\ell)},\ \mathbf b^{(\ell)}\}_{\ell=1}^{L}$$

Reading it symbol by symbol:
- $\theta$ (theta) = "all the learnable numbers of the model, taken together";
- $W^{(\ell)}$ = the weight [[matrix]] of layer number $\ell$ (the superscript in brackets is a
  layer label, not a power);
- $\mathbf b^{(\ell)}$ = the [[bias-vector]] of layer $\ell$ (bold = a [[vector]]);
- $\{\dots\}_{\ell=1}^{L}$ = "collect these for every layer $\ell$ from 1 up to $L$".

A PyTorch `state_dict` is exactly this set, stored as a Python dictionary that maps each
layer's name to its parameter tensors (weights and biases) (book pp. 159, 279). Saving the model =
saving $\theta$. The architecture is *not* saved; it lives in your Python class.

### The basic save / load (appendix A toy network)
```python
torch.save(model.state_dict(), "model.pth")     # write θ to disk

model = NeuralNetwork(2, 2)                      # rebuild the SAME architecture
model.load_state_dict(torch.load("model.pth"))  # copy θ back in
```
(book pp. 278–279)

- `torch.save(obj, path)` serializes a Python object to a file.
- `.pth` and `.pt` are just the most common file-extension conventions; any name works (book p. 279).
- `torch.load` rebuilds the dictionary; `load_state_dict` copies its tensors into the model and
  restores the learned state (book p. 279).
- You need a model instance in memory first, and **its architecture must match the saved one
  exactly** (book p. 279). Here `NeuralNetwork(2, 2)` is the multilayer perceptron of Listing A.4
  (2 inputs → 30 → 20 → 2 outputs). Its weight shapes are $30\times2$, $20\times30$, $2\times20$
  (derived from Listing A.4, not printed in §A.8). A model with other sizes would have tensors that
  don't fit.

The book calls saving the `state_dict` (rather than the whole model object) **the recommended
way** (book pp. 159, 278).

### Saving the GPT model (Chapter 5)
```python
torch.save(model.state_dict(), "model.pth")

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```
(book pp. 159–160)

- `GPT_CONFIG_124M` rebuilds the 124M-parameter GPT-2-small layout from Chapter 4.
- `map_location=device` puts the loaded tensors on the device you choose (CPU or GPU), so a file
  saved on a GPU machine can load on a CPU-only laptop.
- `model.eval()` switches to evaluation mode, which turns dropout off for inference (book p. 160);
  see [[train-vs-eval-mode]].

### A full checkpoint: model + optimizer (to resume training)
If you plan to **continue pretraining** later, the book says to save the optimizer state too
(book p. 160):

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)

checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train();
```
(book p. 160)

Why? Adaptive optimizers such as [[adamw-optimizer|AdamW]] store **extra values for every model
weight**. AdamW uses this history to adjust each parameter's learning rate on the fly. Without it
the optimizer resets, and the model may learn suboptimally or even fail to converge, losing its
ability to generate coherent text (book p. 160). Note the order: build the optimizer on the new
model's parameters first, then load its state into it. At the end, `model.train()` turns dropout
back on because we are about to train again.

**The extra values, as formulas** (standard AdamW; the book names them but does not print them):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$$

- $g_t$ = the [[gradient]] of the loss for this weight at step $t$;
- $m_t$ = a running average of recent gradients (the "which way has this knob been moving" memory);
- $v_t$ = a running average of *squared* gradients (the "how jumpy has this knob been" memory);
- $\beta_1,\beta_2$ = how long the memory is (PyTorch defaults 0.9 and 0.999);
- $t$ = the step counter.

`optimizer.state_dict()` stores $m_t$, $v_t$ and $t$ for every weight, plus the hyperparameters
(learning rate $\eta$, weight decay $\lambda$). So a checkpoint for an AdamW-trained model is
roughly **three times** the size of the weights alone ($\theta$, $m$, $v$).

> Small inconsistency worth knowing: the §5.2 training run (book p. 149) uses `lr=0.0004`, while the §5.4
> restore code builds AdamW with `lr=5e-4`. The book does not comment. In practice
> `optimizer.load_state_dict` overwrites the constructor's `lr` with the saved value (PyTorch
> behaviour, not stated in the book).

**Exercise 5.4** (book p. 160): load model + optimizer in a fresh session and pretrain one more
epoch. The appendix C solution (book p. 305) is the restore code above (printed without
`map_location=device` and without the final `model.train()`), followed by a call to the training
function with `num_epochs=1` (the book prints the name as "train_simple_function";
it means `train_model_simple`).

### The same pattern in the fine-tuning chapters
| Where | File | Code | Page |
|---|---|---|---|
| Ch. 5 pretraining | `model.pth`, `model_and_optimizer.pth` | see above | pp. 159–160 |
| Ch. 6 spam classifier | `review_classifier.pth` | `torch.save(model.state_dict(), "review_classifier.pth")`; reload with `model_state_dict = torch.load("review_classifier.pth", map_location=device)` then `model.load_state_dict(model_state_dict)` | p. 202 |
| Ch. 7 instruction model | `gpt2-medium355M-sft.pth` | name built with `re.sub(r'[ ()]', '', CHOOSE_MODEL)` + `-sft.pth`; reload with `model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))` | p. 238 |
| App. A toy MLP | `model.pth` | `NeuralNetwork(2, 2)` | pp. 278–279 |

In Ch. 6 the architecture you rebuild must include the new 2-output [[classification-head]], not
the original 50,257-way output layer, because that is what was saved. "sft" in Ch. 7 means
supervised fine-tuning (standard expansion; not spelled out in §7.7).

## 🔴 Theory (Expert)
- **Function = architecture + parameters.** A network is a family of functions
  $\{f_\theta\}$ indexed by $\theta\in\mathbb R^{P}$ ($P$ = parameter count, e.g. 124M). The
  architecture fixes the family; the `state_dict` fixes the point $\theta$. `load_state_dict`
  works by *name and shape*: each key must exist in the new model and each tensor must have the
  identical shape, which is why a mismatch raises an error. This is the same safety idea as the
  `assign` shape check used when loading OpenAI's GPT-2 weights in §5.5 ([[pretrained-weights]]).
- **Why the optimizer state matters, precisely.** For AdamW the update is
  $\theta_t=\theta_{t-1}-\eta\big(\hat m_t/(\sqrt{\hat v_t}+\epsilon)+\lambda\theta_{t-1}\big)$ with
  bias corrections $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$. If you resume with
  $m=v=0$ and $t=0$, the bias correction treats the first few new gradients as the whole history,
  so the effective per-parameter step $\eta/\sqrt{\hat v_t}$ jumps away from the value training had
  settled into. That discontinuity is the "optimizer reset" the book warns about. Plain
  [[stochastic-gradient-descent|SGD]] without momentum has no such state, so for it the weights
  alone are a full checkpoint.
- **What a checkpoint does *not* capture.** Exact resumption also needs the data-loader position,
  the [[learning-rate-schedule]] step (warmup / cosine decay counters), and the random-number
  generator states ([[random-seed-reproducibility]]). The book's checkpoint saves only model +
  optimizer, which is enough to *continue* training, not to reproduce an uninterrupted run
  bit for bit.
- **Storage vs math.** Weights stored as 32-bit floats take $4P$ bytes ($\approx 0.5$ GB for
  124M parameters); with two AdamW moment tensors the checkpoint is $\approx 3\times$ that
  ([[parameter-count-and-memory]], [[floating-point-precision]]).
- **Transfer learning rides on this.** Every "load pretrained weights, then fine-tune" step in
  Chapters 6–7 is a `load_state_dict` (or an equivalent manual copy) into a matching architecture,
  followed by training from that $\theta$ as the starting point ([[transfer-learning]], [[fine-tuning]]).

## Connections
- Builds on: [[pytorch]], [[pytorch-nn-module]], [[model-parameters]], [[tensor]], [[matrix]], [[vector]]
- Leads to: [[pretrained-weights]], [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[transfer-learning]]
- Related: [[train-vs-eval-mode]], [[adamw-optimizer]], [[llm-training-loop]], [[dropout]], [[gradient-descent]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.4, book pp. 159–160 (state_dict, `.pth`, `map_location`, model + optimizer checkpoint, Exercise 5.4); §5.2, book p. 149 (training run `lr=0.0004`).
- Same PDF, §6.8, book p. 202 (`review_classifier.pth`).
- Same PDF, §7.7, book p. 238 (`gpt2-medium355M-sft.pth`).
- Same PDF, §A.8, book pp. 278–279 (recommended way, `.pth`/`.pt`, architecture must match).
- Same PDF, appendix C, Exercise 5.4 solution, book p. 305.
- AdamW moment formulas: standard (Loshchilov & Hutter 2019), not printed in the book.
