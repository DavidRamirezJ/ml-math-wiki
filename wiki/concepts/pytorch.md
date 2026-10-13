---
title: PyTorch
type: concept
tags: [software, deep-learning, pytorch, llm, tools]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# PyTorch

> **One-liner:** PyTorch is the Python toolbox the book uses to build and train neural networks: it stores numbers in tensors, computes gradients for you, and gives you ready-made layers.

## 🟢 Intuition (Dummy)
If a [[neural-network]] is a machine, PyTorch is the workshop. It gives you three things
(Figure A.1, book p. 252):

1. **Boxes for numbers** ([[tensor|tensors]]). Like spreadsheets that can have any number of
   dimensions, and that can live on a fast graphics card (GPU) instead of the normal CPU.
2. **An automatic "blame calculator"** (autograd). After the machine makes a mistake, it works
   out how much each knob was responsible, so you never have to do calculus by hand.
3. **Ready-made parts** (the deep learning library): layers, loss functions, optimizers,
   even pretrained models.

Why this one? Most LLMs today are implemented in PyTorch, which is why the book uses it
(book p. 5). It has been the most widely used deep learning library for research since 2019
(Papers With Code), and about 40% of respondents of the Kaggle 2022 survey use it
(book p. 251). People like it because it is easy to use without being rigid: beginners get
simple building blocks, experts can still tweak the low-level details (book pp. 251–252).

A bit of history: PyTorch continues the older **Torch** library, which was written in the
Lua language; PyTorch brought it to Python (book p. 256).

## 🟡 Mechanics (Practitioner)

### The three core components and the math behind each
**1. Tensor library.** Extends the NumPy array idea with GPU acceleration and an easy
switch between CPU and GPU (book p. 252). See [[tensor]]. A typical operation is the
matrix product
$$(\mathbf{A}\mathbf{B})_{ij}=\sum_{k=1}^{n}A_{ik}B_{kj}$$
- $\mathbf{A}$ is $m\times n$, $\mathbf{B}$ is $n\times p$; $(\mathbf{AB})_{ij}$ = the entry in row $i$, column $j$ of the result.
- $\sum_{k=1}^{n}$ = add up over the shared inner index $k$; written `A @ B` in PyTorch.

**2. Automatic differentiation engine (autograd).** Automatically computes gradients of
tensor operations, which makes [[backpropagation]] easy (book p. 252). For a scalar loss
$L(\theta)$ with parameters $\theta\in\mathbb{R}^n$ it computes the [[gradient]]

$$\nabla_\theta L=\left(\frac{\partial L}{\partial\theta_1},\dots,\frac{\partial L}{\partial\theta_n}\right)$$

- $\nabla_\theta L$ ("nabla", "gradient of $L$ with respect to $\theta$") = a list of slopes, one per parameter.
- $\frac{\partial L}{\partial\theta_i}$ = [[partial-derivative]]: how much $L$ changes if you nudge only $\theta_i$.

It gets these by applying the [[chain-rule]] $\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\frac{\partial y}{\partial x}$
backwards over the [[computational-graph]] that PyTorch records during the [[forward-pass]]
(see [[automatic-differentiation]]). In code this is one call: `loss.backward()`.

**3. Deep learning library.** Building blocks such as [[pytorch-nn-module|`torch.nn.Module`]],
[[linear-layer|`torch.nn.Linear`]], loss functions (`F.cross_entropy`) and optimizers
(`torch.optim.SGD`, `torch.optim.AdamW`) (book p. 252).

### Installing and checking (book pp. 254–257)
```bash
pip install torch          # CPU or CUDA build chosen automatically
pip install torch==2.4.0   # the version the book uses
```
```python
import torch
torch.__version__                    # '2.4.0'
torch.cuda.is_available()            # True if an NVIDIA (CUDA) GPU is usable
torch.backends.mps.is_available()    # True on Apple Silicon (M1/M2/M3)
```
- Two builds exist: a leaner CPU-only version and a CPU + GPU version (book p. 255).
- Recommended GPUs: NVIDIA T4, RTX 2080 Ti or newer; AMD GPUs have experimental ROCm support (book p. 255).
- Python tip: use one or two releases behind the newest (e.g. 3.11 or 3.12 when 3.13 is out) (book p. 254).
- No GPU? Google Colab offers time-limited GPU access (book p. 257). GPUs are not needed for the early chapters (book p. 256).

### The whole workflow in one picture
The pieces fit together like this in every training script of the book (Appendix A, Ch. 5–7):

```python
model = NeuralNetwork(num_inputs=2, num_outputs=2)          # nn.Module
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)     # optimizer
for epoch in range(3):
    model.train()
    for features, labels in train_loader:                   # DataLoader -> mini-batches
        logits = model(features)                            # forward pass (tensor ops)
        loss = F.cross_entropy(logits, labels)              # loss function
        optimizer.zero_grad()
        loss.backward()                                     # autograd
        optimizer.step()                                    # theta <- theta - lr * grad
    model.eval()
```
(book pp. 274–275). The update `optimizer.step()` does

$$\theta \leftarrow \theta - \eta\,\nabla_\theta L(\theta)$$

- $\theta$ = the parameters; $\leftarrow$ = "replace with"; $\eta$ (eta) = the [[learning-rate]] (`lr=0.5`);
  $\nabla_\theta L$ = the gradient that `loss.backward()` stored in each parameter's `.grad`.

See [[pytorch-dataset-and-dataloader]] for `train_loader` and [[mini-batch]] for what one iteration sees.

## 🔴 Theory (Expert)
- **Define-by-run (dynamic graphs).** PyTorch builds the computation graph on the fly while the
  Python code executes, then runs reverse-mode autodiff over it. That is why ordinary Python control
  flow works inside `forward`, and why gradients **accumulate** in `.grad` unless you call
  `optimizer.zero_grad()` (book NOTE, p. 276). Accumulation is also exactly what
  [[gradient-accumulation]] exploits on purpose.
- **Reverse mode is the right choice** because a training step has many inputs (billions of
  parameters) and one scalar output (the loss); reverse-mode AD costs a small constant multiple of
  one forward pass, independent of the number of parameters. See [[backpropagation]].
- **Devices.** Every tensor lives on one device and all operands of an op must share it, otherwise
  PyTorch raises `RuntimeError: Expected all tensors to be on the same device` (book p. 280). See
  [[gpu-computing]] and [[distributed-data-parallel]] for scaling to many GPUs (book pp. 282–288).
- **Numerics.** Default floats are 32-bit (`torch.float32`) as a trade-off between precision,
  memory and GPU throughput (book p. 259); LLM chapters later use lower precision (see
  [[floating-point-precision]]). `F.cross_entropy` takes logits and applies the softmax internally "for efficiency and
  numerical stability reasons" (book p. 276); in practice PyTorch fuses it as log-softmax + negative
  log-likelihood (implementation detail, not stated in the book). See [[numerical-stability]].
- **Why it matters for this wiki.** Every formula on the LLM pages (attention scores
  $\mathbf{Q}\mathbf{K}^\top$, layer norm, softmax, cross-entropy) is implemented as a short
  sequence of PyTorch tensor ops, so reading PyTorch code is a direct way to read the math.

## Connections
- Builds on: [[vector]], [[matrix]], [[gradient]], [[chain-rule]]
- Core pieces: [[tensor]], [[automatic-differentiation]], [[computational-graph]], [[pytorch-nn-module]], [[linear-layer]], [[pytorch-dataset-and-dataloader]]
- Used for: [[neural-network]], [[llm-training-loop]], [[training-loop]], [[gpt-architecture]], [[backpropagation]], [[stochastic-gradient-descent]], [[adamw-optimizer]]
- Related: [[gpu-computing]], [[distributed-data-parallel]], [[floating-point-precision]], [[model-checkpointing]], [[train-vs-eval-mode]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 5 (NOTE: most LLMs are implemented in PyTorch).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix A intro and §A.1–A.1.3, book pp. 251–257 (popularity, three components, Figure A.1, installation, Torch/Lua sidebar, Apple Silicon).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2.2, book p. 259 (float32 default); §A.7, book pp. 274–276 (training loop, zero_grad NOTE); §A.9.1, book p. 280 (same-device rule).
