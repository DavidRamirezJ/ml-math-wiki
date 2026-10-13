---
title: Gradient
type: concept
tags: [calculus, differentiation, multivariate, optimization]
difficulty: intermediate
sources: [mml-ch05-vector-calculus, llm-appA-introduction-to-pytorch, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch06-fine-tuning-for-classification, llm-appD-training-loop-bells-and-whistles]
created: 2026-06-04
updated: 2026-10-13
---

# Gradient

> **One-liner:** The gradient is the list of all a function's partial slopes, bundled into one vector that points in the direction the function rises fastest.

## 🟢 Intuition (Dummy)
Stand somewhere on a hilly landscape. There is one direction that is *steepest
uphill*. The **gradient** is an arrow pointing exactly that way, and its length
tells you how steep the climb is. If you want to get *down* the hill fast (which
is what training a model does — going downhill on an error landscape), you walk
opposite the gradient. That single idea, "take a step against the gradient,"
powers almost all of machine learning's learning.

**In LLMs.** A GPT model has millions of knobs (the small GPT-2 has about 124
million). After every batch of text, PyTorch hands each knob its own little
gradient number: "if you go up a bit, the loss goes up (or down) this much."
The optimizer then nudges every knob a tiny step the other way. Raschka's
book keeps it friendly: you don't need deep calculus — you just need to know
that the gradient is the "which way and how hard" signal that training
follows (book pp. 99, 263).

## 🟡 Mechanics (Practitioner)
For $f:\mathbb R^n\to\mathbb R$, the **gradient** collects every
[[partial-derivative]] into a **row vector** (5.40):
$$
\nabla_{\mathbf x} f = \frac{df}{d\mathbf x}
= \Big[\ \tfrac{\partial f}{\partial x_1}\ \ \tfrac{\partial f}{\partial x_2}\ \cdots\ \tfrac{\partial f}{\partial x_n}\ \Big]\in\mathbb R^{1\times n}.
$$
(The book uses the **numerator layout**, so the gradient of a scalar function is
a *row* vector — this makes the [[chain-rule]] literal matrix multiplication.)

**Worked example.** $f(x_1,x_2)=x_1^2 x_2 + x_1 x_2^3$:
$$\nabla f = \big[\,2x_1 x_2 + x_2^3\ \ \ x_1^2 + 3x_1 x_2^2\,\big]\in\mathbb R^{1\times2}.$$

**Useful gradient identities (5.99–5.108)** that recur in ML (constant $\mathbf a$, matrix $B$):
$$
\frac{\partial \mathbf a^\top\mathbf x}{\partial \mathbf x}=\mathbf a^\top,\qquad
\frac{\partial \mathbf x^\top B\mathbf x}{\partial \mathbf x}=\mathbf x^\top(B+B^\top),\qquad
\frac{\partial}{\partial X}\operatorname{tr}(f(X))=\operatorname{tr}\!\Big(\tfrac{\partial f(X)}{\partial X}\Big).
$$

**ML usage — least-squares gradient (Example 5.11).** For a linear model
$\mathbf y=\Phi\boldsymbol\theta$ with loss $L=\lVert\mathbf e\rVert^2$,
$\mathbf e=\mathbf y-\Phi\boldsymbol\theta$, the chain rule gives
$$\frac{\partial L}{\partial\boldsymbol\theta}
=\frac{\partial L}{\partial\mathbf e}\frac{\partial\mathbf e}{\partial\boldsymbol\theta}
= (2\mathbf e^\top)(-\Phi) = -2(\mathbf y-\Phi\boldsymbol\theta)^\top\Phi\in\mathbb R^{1\times D}.$$
Setting this to zero yields the normal equations of linear regression (Ch. 9).

### In LLMs (Raschka, *Build a Large Language Model*)

**The book's definition (sidebar "Partial derivatives and gradients", §A.4, pp. 263–264).**
A [[partial-derivative]] measures how fast a function changes when you wiggle
*one* of its inputs; the gradient is "a vector containing all of the partial
derivatives of a multivariate function":
$$\nabla f(\mathbf x)=\Big[\tfrac{\partial f}{\partial x_1},\ \dots,\ \tfrac{\partial f}{\partial x_n}\Big].$$
Read it as: $\nabla$ ("nabla") = "the gradient of"; $f$ = the function (in
training, the loss); $\mathbf x$ = the point where we measure it (the current
weights); each $\tfrac{\partial f}{\partial x_i}$ = "slope of $f$ if only $x_i$
moves." The gradients tell us how to update each parameter to make the
[[loss-function]] smaller — the loss being "a proxy for measuring the model's
performance."

**A tiny worked example with the book's numbers (Listing A.3, p. 264).** A
one-input [[logistic-regression]] neuron: $z=w_1x_1+b$, $a=\sigma(z)$, loss
$L=-[y\ln a+(1-y)\ln(1-a)]$ ([[binary-cross-entropy]]). With $x_1=1.1$,
$w_1=2.2$, $b=0$, $y=1$:
- $z=2.42$, $a=\sigma(2.42)\approx0.9183$, $L\approx0.0852$;
- $\tfrac{\partial L}{\partial b}=a-y\approx-0.0817$, and
  $\tfrac{\partial L}{\partial w_1}=x_1(a-y)\approx1.1\times(-0.0817)\approx-0.0898$.

So the gradient of the loss w.r.t. the two parameters is
$\nabla_{(w_1,b)}L\approx[-0.0898,\ -0.0817]$. PyTorch prints exactly
`(tensor([-0.0898]),)` and `(tensor([-0.0817]),)`. Both entries are negative,
so gradient descent will *increase* $w_1$ and $b$ — which pushes $a$ toward
the true label 1. (The intermediate values above are our hand check; the book
prints only the final two numbers.)

**Where the gradient lives in code.** After `loss.backward()`, PyTorch stores
each parameter's gradient in its `.grad` attribute, with the **same shape as
the parameter** (a $3\times3$ weight matrix gets a $3\times3$ `.grad`) (pp. 112, 264, 318).
The optimizer then applies [[stochastic-gradient-descent]] (§A.7, p. 276):
$$\theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal L(\theta)$$
$\theta$ = all trainable weights and biases; $\leftarrow$ = "is replaced by";
$\eta$ (eta) = the [[learning-rate]] (`lr=0.5` in the toy example);
$\nabla_\theta\mathcal L$ = gradient of the mini-batch loss w.r.t. $\theta$. In
the book's words: "multiplying the gradients with the learning rate and adding
the scaled negative gradient to the parameters."

**Measuring gradients per layer (§4.4, pp. 111–113).** To compare layers, the
book prints the **mean absolute gradient** of each weight matrix
$W\in\mathbb R^{m\times k}$:
$$\bar g=\frac{1}{mk}\sum_{i=1}^{m}\sum_{j=1}^{k}\Big|\frac{\partial\mathcal L}{\partial W_{ij}}\Big|$$
($\bar g$ = one summary number per layer; $|\cdot|$ = absolute value; $mk$ =
number of entries, 9 for a $3\times3$ matrix). In a 5-layer toy network
(`layer_sizes=[3,3,3,3,3,1]`, input `[1., 0., -1.]`, MSE loss vs target 0):

| layer | no shortcut | with [[shortcut-connections]] |
|---|---|---|
| `layers.4` (last) | 0.00505 | 1.3259 |
| `layers.3` | 0.00140 | 0.2666 |
| `layers.2` | 0.00072 | 0.3290 |
| `layers.1` | 0.00012 | 0.2069 |
| `layers.0` (first) | 0.00020 | 0.2217 |

Without shortcuts the gradient shrinks toward the early layers — the
[[vanishing-and-exploding-gradients|vanishing-gradient problem]]. With
shortcuts it stays healthy (first layer ≈1100× larger; our ratio, not the book's).

**Keeping gradients healthy — the book's recurring theme.**
- *Scaled attention* (§3.4, p. 69): dividing attention scores by $\sqrt{d_k}$
  avoids "small gradients": big dot products make [[softmax]] act like a step
  function, whose gradient is near zero, so learning stalls
  ([[scaled-dot-product-attention]]).
- *Layer normalization* (§4.2, p. 99): stabilizes training against vanishing
  or exploding gradients ([[layer-normalization]]).
- *GELU* (§4.3, p. 106): unlike ReLU it has a **non-zero gradient for almost all
  negative inputs** (the exception is near $x\approx-0.75$, where the curve has its
  minimum) ([[gelu]]).
- *Gradient clipping* (§D.3, pp. 317–319): if the gradient's L2 [[norm]]
  $\lVert\mathbf G\rVert_2=\sqrt{\sum G_{ij}^2}$ is bigger than `max_norm`, multiply
  it by $c=\text{max\_norm}/\lVert\mathbf G\rVert_2$, so its length becomes exactly
  `max_norm` while its direction stays the same ([[gradient-clipping]]). On a
  freshly initialized GPT-124M, `clip_grad_norm_(..., max_norm=1.0)` drops the
  largest gradient entry from `0.0411` to `0.0185`.

  > ⚠️ Book erratum (p. 318): for $\mathbf G=\begin{bmatrix}1&2\\3&4\end{bmatrix}$ the book writes
  > $\sqrt{1^2+2^2+2^2+4^2}=5$. The correct norm is $\sqrt{1+4+9+16}=\sqrt{30}\approx5.477$,
  > so the scale factor is $1/\sqrt{30}\approx0.183$, not $1/5$. The method is right; the arithmetic is a typo.

**When there is no useful gradient (§6.6, p. 193).** Classification accuracy
is a count of correct `argmax` guesses — a step function whose gradient is
zero almost everywhere. You can't follow a slope that is flat, so the book
trains on [[cross-entropy-loss]] instead, as a smooth stand-in
([[surrogate-loss]]).

## 🔴 Theory (Expert)
- The gradient is the matrix of the best **linear approximation** of $f$ at a
  point: $f(\mathbf x)\approx f(\mathbf x_0)+(\nabla_{\mathbf x}f)(\mathbf x_0)(\mathbf x-\mathbf x_0)$
  (5.148) — the first-order [[taylor-series]] term.
- **Steepest ascent.** Among all unit directions, $\nabla f/\lVert\nabla f\rVert$
  maximizes the directional derivative $\nabla f\cdot \mathbf u$; this is why
  $-\nabla f$ is the steepest-descent direction used by gradient descent (Ch. 7).
- The gradient of $f:\mathbb R^n\to\mathbb R$ is the special $m=1$ case of the
  [[jacobian]] (a $1\times n$ matrix). Going to vector-valued $f$ promotes it to
  a full matrix; going to matrix arguments (§5.4) promotes it to a **gradient
  tensor** with entries $\partial A_{ij}/\partial B_{kl}$.
- The chain rule and these identities (Petersen & Pedersen, 2012) let you
  differentiate complicated objectives symbolically; at scale this is automated
  by [[automatic-differentiation]].

**In LLMs (connections to deeper structure).**
- **Layout note.** MML writes the gradient of a scalar function as a *row*
  vector $\in\mathbb R^{1\times n}$ (numerator layout). PyTorch stores `.grad`
  with the *same shape as the parameter* (a $768\times768$ weight gets a
  $768\times768$ gradient). Same numbers, different arrangement — not a
  contradiction, just a convention to keep in mind when translating formulas
  into code.
- **Why shortcuts save the gradient.** For a residual layer
  $\mathbf x_{\ell+1}=\mathbf x_\ell+f_\ell(\mathbf x_\ell)$, the [[jacobian]] is
  $I+\partial f_\ell/\partial\mathbf x_\ell$. The backward signal always has an
  identity path, so it is not forced through a product of many small Jacobians
  that shrinks toward $0$ (standard argument behind book p. 109; the book
  states only the conclusion).
- **Why $\sqrt{d_k}$.** If query/key components are independent with unit
  variance, $\operatorname{Var}(\mathbf q\cdot\mathbf k)=d_k$; dividing by $\sqrt{d_k}$
  resets the variance to 1 and keeps softmax out of its flat (zero-gradient)
  zone (our rationale; the book gives only the gradient argument).
- **Clipping as projection.** Norm clipping is
  $\mathbf g\leftarrow\mathbf g\cdot\min\!\big(1,\ \text{max\_norm}/\lVert\mathbf g\rVert_2\big)$ —
  projection onto the L2 ball of radius `max_norm`. PyTorch computes one norm
  over *all* parameters jointly. With plain SGD this caps the step length at
  $\eta\cdot\text{max\_norm}$; with [[adamw-optimizer|AdamW]] the per-parameter
  normalization means the cap is less direct (the book does not discuss this).
- **Gradients are linear.** The gradient of an average loss is the average of
  the per-example gradients, $\nabla\frac1B\sum_i\ell_i=\frac1B\sum_i\nabla\ell_i$.
  This is why a mini-batch gradient is an unbiased estimate of the full-data
  gradient, and why multi-GPU training can simply average gradients (§A.9;
  [[distributed-data-parallel]]).

## Connections
- Builds on: [[partial-derivative]], [[derivative]], [[vector]]
- Leads to: [[jacobian]], [[hessian]], [[taylor-series]], [[gradient-descent]], [[stochastic-gradient-descent]]
- Used by: gradient descent (Ch. 7), [[backpropagation]], linear regression (Ch. 9), [[automatic-differentiation]], [[training-loop]], [[gradient-clipping]], [[adamw-optimizer]]
- Related (LLM book): [[vanishing-and-exploding-gradients]], [[shortcut-connections]], [[scaled-dot-product-attention]], [[layer-normalization]], [[gelu]], [[surrogate-loss]], [[norm]], [[chain-rule]]
- People: [[sebastian-raschka]]

## Sources
- `raw/mml-book.pdf` §5.2 (5.40), §5.5 (identities 5.99–5.108), Example 5.11, printed pp. 146–148, 153–158 (PDF pp. 152–154, 159–164).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4 (sidebar "Partial derivatives and gradients", Listing A.3), book pp. 263–264 (PDF pp. 285–286).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7 (SGD update, `.grad`), book pp. 274–276.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (scaled-attention sidebar), book p. 69; §4.2, book p. 99; §4.3, book pp. 105–108; §4.4 (Listing 4.5, `print_gradients`, Fig. 4.12), book pp. 109–113.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6 (accuracy not differentiable), book p. 193.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.3 (gradient clipping, L2 norm), book pp. 317–319.
