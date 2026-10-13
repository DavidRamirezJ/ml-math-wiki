---
title: Automatic Differentiation
type: concept
tags: [calculus, deep-learning, differentiation, numerical-methods]
difficulty: advanced
sources: [mml-ch05-vector-calculus, llm-appA-introduction-to-pytorch, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-appD-training-loop-bells-and-whistles]
created: 2026-06-04
updated: 2026-10-13
---

# Automatic Differentiation

> **One-liner:** Automatic differentiation computes the exact gradient of any program by breaking it into elementary steps and chaining their known derivatives together.

## 🟢 Intuition (Dummy)
You don't need a clever closed-form formula for a function's derivative. Any
computation, no matter how tangled, is built from tiny known steps — add,
multiply, $\sin$, $\exp$. We know the derivative of each tiny step exactly. So
record the steps the program actually took (a **computation graph**), then walk
through it multiplying the little derivatives together using the chain rule. The
machine assembles the exact gradient automatically — no pen-and-paper calculus,
and no error-prone finite-difference guessing. This is what every modern deep
learning framework does under the hood.

**In LLMs.** In PyTorch this engine is called **autograd**. Raschka lists it
as one of PyTorch's three pillars — a tensor library, an automatic
differentiation engine, and deep-learning utilities built on top (book p. 252).
Think of autograd as a diligent note-taker: while your GPT model runs
forward, it quietly writes down every multiply and add. When you call
`loss.backward()`, it reads those notes in reverse and fills in a gradient for
every one of the model's 124 million weights — so you never derive a single
derivative by hand (book pp. 112, 264).

## 🟡 Mechanics (Practitioner)
Rewrite a function with **intermediate variables**. For
$f(x)=\sqrt{x^2+\exp(x^2)}+\cos\!\big(x^2+\exp(x^2)\big)$ (5.122):
$$a=x^2,\quad b=\exp(a),\quad c=a+b,\quad d=\sqrt{c},\quad e=\cos(c),\quad f=d+e.$$
This is the **computation graph** (Fig. 5.11). Each edge has an elementary
derivative:
$$\tfrac{\partial a}{\partial x}=2x,\ \tfrac{\partial b}{\partial a}=\exp(a),\
\tfrac{\partial c}{\partial a}=\tfrac{\partial c}{\partial b}=1,\
\tfrac{\partial d}{\partial c}=\tfrac{1}{2\sqrt c},\
\tfrac{\partial e}{\partial c}=-\sin(c),\
\tfrac{\partial f}{\partial d}=\tfrac{\partial f}{\partial e}=1.$$
**Reverse pass** (start at the output, work back), accumulating via the
[[chain-rule]]:
$$
\tfrac{\partial f}{\partial c}=\tfrac{\partial f}{\partial d}\tfrac{\partial d}{\partial c}+\tfrac{\partial f}{\partial e}\tfrac{\partial e}{\partial c},\quad
\tfrac{\partial f}{\partial b}=\tfrac{\partial f}{\partial c}\cdot1,\quad
\tfrac{\partial f}{\partial a}=\tfrac{\partial f}{\partial b}\exp(a)+\tfrac{\partial f}{\partial c}\cdot1,\quad
\tfrac{\partial f}{\partial x}=\tfrac{\partial f}{\partial a}\cdot2x.
$$
The result is exact (to machine precision), and computing it costs about the
same as evaluating $f$ — even though the explicit symbolic derivative (5.110) is
far longer than $f$ itself.

**ML usage:** this is exactly how PyTorch/TensorFlow/JAX compute gradients;
[[backpropagation]] is reverse-mode AD specialized to neural networks.

### In LLMs: PyTorch's autograd (Raschka, Appendix A and Ch. 3–4)

**1. The computation graph (§A.3, pp. 261–262).** A [[computational-graph]]
is "a directed graph that allows us to express and visualize mathematical
expressions." The book's example is a [[logistic-regression]] [[forward-pass]]
(Fig. A.7):
$$u=w_1\times x_1,\qquad z=u+b,\qquad a=\sigma(z),\qquad \text{loss}=L(a,y).$$
Read it as: $x_1$ = the input feature; $w_1$ = a trainable weight; $b$ = a
trainable bias; $u$ = an intermediate result; $z$ = the net input; $\sigma$ =
the [[sigmoid-function]]; $a$ = the output score in $(0,1)$; $y$ = the true
label; $L$ = the loss ([[binary-cross-entropy]]). PyTorch's graph is
**dynamic**: it is built on the fly as your Python code runs.

**2. When does PyTorch record the graph? (§A.4, p. 263).** Only if at least
one leaf (terminal) tensor has `requires_grad=True`. A **leaf** is a tensor
you created yourself (like `w1`, `b`), not the result of an operation. Layers
such as `nn.Linear` set `requires_grad=True` on their weights and biases by
default (§A.5, p. 267). In §3.4 the book even sets `requires_grad=False` on
the attention weight matrices, just "to reduce clutter" in the printouts
(p. 66).

**3. Reverse mode = "chain rule from right to left" (Fig. A.8, p. 263).** Each
edge of the graph carries a local derivative, and the gradients come from
multiplying them backward from the loss:
$$\frac{\partial L}{\partial w_1}=\frac{\partial u}{\partial w_1}\times\frac{\partial z}{\partial u}\times\frac{da}{dz}\times\frac{\partial L}{\partial a},\qquad
\frac{\partial L}{\partial b}=\frac{\partial z}{\partial b}\times\frac{da}{dz}\times\frac{\partial L}{\partial a}.$$
Symbol by symbol: $\tfrac{\partial L}{\partial a}$ = how the loss reacts to the
output; $\tfrac{da}{dz}$ = slope of the sigmoid; $\tfrac{\partial z}{\partial u}=1$ and
$\tfrac{\partial z}{\partial b}=1$ (an addition passes the signal through
unchanged); $\tfrac{\partial u}{\partial w_1}=x_1$. The caption names this
**reverse-mode automatic differentiation** or **backpropagation**.

**4. Two ways to ask for gradients (Listing A.3, p. 264).**
```python
import torch.nn.functional as F
from torch.autograd import grad

y  = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b  = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)   # (tensor([-0.0898]),)
grad_L_b  = grad(loss, b,  retain_graph=True)   # (tensor([-0.0817]),)

loss.backward()          # the everyday way: fills .grad of every leaf
print(w1.grad, b.grad)   # -0.0898, -0.0817
```
- `grad(...)` (manual) is good for "experimentation, debugging, and
  demonstrating concepts"; it returns a tuple.
- `retain_graph=True`: by default PyTorch **destroys the graph after
  computing gradients to free memory**; this flag keeps it for reuse.
- `loss.backward()` computes gradients for **all** leaf nodes and stores each
  one in that tensor's `.grad`. (The book prints `w1.grad` with a tuple
  wrapper; real PyTorch prints a plain `tensor([-0.0898])`.)

**5. The `grad_fn` breadcrumbs.** Every tensor produced by a tracked
operation remembers "the last-used function that computed it" (§A.5, p. 269).
You see these all through the book:

| printout | where | operation recorded |
|---|---|---|
| `grad_fn=<MmBackward0>` | §3.4 self-attention context vectors | matrix multiply |
| `grad_fn=<ReluBackward0>`, `<MeanBackward1>`, `<VarBackward0>`, `<DivBackward0>` | §4.2 layer-norm demo | ReLU, mean, variance, divide |
| `grad_fn=<UnsafeViewBackward0>` | §4.1 / §4.6 GPT logits | reshape (view) |
| `grad_fn=<AddmmBackward0>` | §A.5 MLP output | fused `bias + x @ W.T` |

**6. You never write `backward` yourself (§A.5, p. 265; §4.4, p. 112).** In an
`nn.Module` you write only `forward`; autograd derives the backward pass.
The book calls `loss.backward()` the convenience method that makes deep
networks "much more accessible."

**7. Switching it off: `torch.no_grad()` (§A.5, p. 269; §4.7).** For
inference (prediction, text generation, evaluation), building the graph wastes
memory and compute, so best practice is
```python
with torch.no_grad():
    logits = model(idx_cond)
```

**8. Gradients pile up (§A.7, p. 276).** PyTorch **adds** new gradients into
`.grad` instead of overwriting. That's why every [[training-loop]] calls
`optimizer.zero_grad()` each round ([[gradient-accumulation]] uses this on
purpose).

**9. Across GPUs (§A.9, pp. 283–286).** With
[[distributed-data-parallel|DistributedDataParallel]], each GPU runs its own
forward and backward pass on its slice of data; the gradients are then
**averaged and synchronized** so every model copy gets the same update.

## 🔴 Theory (Expert)
**Formalization (5.143–5.145).** Let $x_1,\dots,x_d$ be inputs,
$x_{d+1},\dots,x_{D-1}$ intermediates, $x_D=f$ the output, with each
$x_i=g_i(x_{\mathrm{Pa}(x_i)})$ a function of its parent nodes. Then with
$\partial f/\partial x_D=1$,
$$\frac{\partial f}{\partial x_i}=\sum_{x_j:\,x_i\in\mathrm{Pa}(x_j)}\frac{\partial f}{\partial x_j}\frac{\partial g_j}{\partial x_i}.$$
Equation 5.143 is **forward propagation** of values; 5.145 is **backpropagation**
of gradients.

- **AD ≠ symbolic ≠ numerical.** Unlike symbolic differentiation it never builds
  the giant expression; unlike finite differences it is exact, not an
  approximation.
- **Forward vs. reverse mode.** Both compute $\frac{dy}{dx}=\frac{dy}{db}\frac{db}{da}\frac{da}{dx}$;
  they differ in *association order* (5.120 reverse, 5.121 forward). Reverse mode
  is cheap when outputs $\ll$ inputs (scalar loss, many parameters) — the deep
  learning regime.
- Works on any program whose elementary operations are differentiable;
  branches/loops need care. References: Baydin et al. (2018); Griewank & Walther.

**In LLMs (deeper view).**
- **Define-by-run.** PyTorch's dynamic graph is a *tape* recorded during the
  forward pass (5.143 executed live). Because it is rebuilt every call, Python
  control flow (`if`, loops — e.g. the `use_shortcut` branch in Listing 4.5)
  is differentiated along the path actually taken. This answers MML's "control
  flow needs care" caveat in practice.
- **Vector–Jacobian products.** For a scalar loss, reverse mode never builds a
  full [[jacobian]]; each node maps the incoming row vector
  $\bar{\mathbf v}=\partial L/\partial\text{(output)}$ to
  $\bar{\mathbf v}\,\partial\text{(output)}/\partial\text{(input)}$. The `grad_fn`
  objects above are exactly these VJP rules (e.g. `MmBackward0` for $Y=XW$
  returns $\bar Y W^\top$ and $X^\top\bar Y$ — see [[matrix-transpose]]).
- **Memory, not just time.** Reverse mode must keep the forward activations
  until the backward pass uses them. That's why PyTorch frees the graph after
  `.backward()` by default (`retain_graph` overrides it), and why `no_grad()`
  saves so much memory at inference.
- **Accumulation is the sum rule.** Adding into `.grad` is the
  $\sum_{x_j:\,x_i\in\mathrm{Pa}(x_j)}$ of (5.145) spread over calls: gradients of a
  sum of losses are the sum of gradients. Hence the need for `zero_grad()`,
  and the legitimacy of averaging gradients across GPUs (linearity of the
  [[gradient]]).
- The derivations in these bullets are standard results, not stated in the
  book; the book stays at the "call `.backward()`" level on purpose.

## Connections
- Builds on: [[chain-rule]], [[jacobian]], [[derivative]], [[partial-derivative]], [[gradient]], [[computational-graph]]
- Leads to: [[backpropagation]] (its reverse-mode specialization), [[training-loop]]
- Used by: deep learning frameworks ([[pytorch]] autograd), gradient-based optimization (Ch. 7), [[stochastic-gradient-descent]], [[gradient-accumulation]], [[distributed-data-parallel]], [[gradient-clipping]]
- Related (LLM book): [[forward-pass]], [[tensor]], [[logistic-regression]], [[sigmoid-function]], [[binary-cross-entropy]], [[shortcut-connections]], [[train-vs-eval-mode]]
- People: [[sebastian-raschka]]

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** In the §5.2 training loop, one call `loss.backward()` computes the gradient of the cross-entropy loss with respect to all ≈124M weights of the GPT model, through every transformer block, by reverse-mode autodiff. Then `optimizer.step()` uses these gradients and `optimizer.zero_grad()` resets them before the next batch (Listing 5.3, book pp. 147–148). For evaluation, the code wraps the loss in `torch.no_grad()` so no graph is built (book p. 145).

## Sources
- `raw/mml-book.pdf` §5.6 & §5.6.2 (Example 5.14, 5.119–5.145), printed pp. 159–164 (PDF pp. 165–170).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1 (three PyTorch components, Fig. A.1), book pp. 251–252.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2 (autograd as PyTorch's addition to NumPy), book p. 258.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3 (Listing A.2, Fig. A.7), book pp. 261–262; §A.4 (Fig. A.8, Listing A.3), book pp. 263–264 (PDF pp. 285–286).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (`forward`/`backward`, `requires_grad`, `grad_fn`, `no_grad`), book pp. 265–269; §A.7 (gradient accumulation note), book p. 276; §A.9 (DDP gradient sync), book pp. 283–286.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (`requires_grad=False`, `MmBackward0`), book pp. 66–71; §4.1, book pp. 92–99; §4.4 (`loss.backward()`), book p. 112; §4.6, book p. 120; §4.7 (`no_grad` in generation), book pp. 122–127.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.3 (`.grad` filled by `.backward()`), book p. 318.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.3, book p. 145 (`torch.no_grad`); §5.2, book pp. 147–148 (Listing 5.3: `zero_grad`, `backward`, `step`).
