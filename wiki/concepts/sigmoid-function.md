---
title: Sigmoid Function
type: concept
tags: [activation-function, probability, classification, calculus]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, mml-ch09-linear-regression, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Sigmoid Function

> **One-liner:** The sigmoid is an S-shaped curve that takes any number — huge, tiny, negative — and gently squeezes it into a value between 0 and 1.

## 🟢 Intuition (Dummy)
Think of a dimmer switch that can never go fully off or fully on. Turn the knob far to
the left and the light gets close to 0 but never quite reaches it; far to the right it gets
close to 1. In the middle (knob at 0) the light sits at exactly half brightness, 0.5.

That's the sigmoid. It's handy whenever you need a "how likely is it?" answer: a model
produces a raw score that can be anything, and the sigmoid turns it into something that
looks like a probability. In the LLM book it appears in the tiny [[logistic-regression]]
example of Appendix A as the **activation function** — the step that turns the net input
$z$ into the output $a$ (book p. 262).

## 🟡 Mechanics (Practitioner)

### The formula
$$\sigma(z) = \frac{1}{1+e^{-z}}$$

Symbol by symbol:
- $\sigma$ — the Greek letter sigma, the usual name for the sigmoid.
- $z$ — the input, any real number (in a neuron: the net input $w_1x_1 + b$).
- $e$ — Euler's number, about 2.718; $e^{-z}$ means "$e$ to the power $-z$".
- When $z$ is big and positive, $e^{-z}$ is nearly 0, so $\sigma(z)\approx 1/1 = 1$.
- When $z$ is big and negative, $e^{-z}$ is huge, so $\sigma(z)\approx 0$.
- When $z = 0$: $\sigma(0) = 1/(1+1) = 0.5$.

(The book writes $a = \sigma(z)$ in Figure A.7 and uses `torch.sigmoid`, but does not print the
formula; it's the standard definition.)

| $z$ | −2 | 0 | 2 | 2.42 (book) |
|---|---|---|---|---|
| $\sigma(z)$ | 0.1192 | 0.5 | 0.8808 | 0.9183 |

### The book's example (Listing A.2, book p. 262)
```python
y  = torch.tensor([1.0])    # true label
x1 = torch.tensor([1.1])    # input feature
w1 = torch.tensor([2.2])    # weight parameter
b  = torch.tensor([0.0])    # bias unit
z = x1 * w1 + b             # net input: 2.42
a = torch.sigmoid(z)        # activation and output: ~0.9183
loss = F.binary_cross_entropy(a, y)
```
The output $a \approx 0.9183$ is "a score between 0 and 1" that gets compared with the true
label $y = 1$ to compute the loss (book p. 262).

### Its derivative (used in Figure A.8)
In the chain-rule picture of Figure A.8 (book p. 263), one edge is labelled $\frac{da}{dz}$
— the slope of the sigmoid. It has a famously neat form:

$$\frac{da}{dz} = \sigma'(z) = \sigma(z)\,\big(1-\sigma(z)\big) = a(1-a)$$

Reading it: $\sigma'(z)$ ("sigma prime of $z$") = how steep the S-curve is at $z$; you can get it
from the output $a$ alone, without recomputing anything. For the book's numbers:
$0.9183 \times (1 - 0.9183) \approx 0.0750$. Multiply by $\partial L/\partial a \approx -1.089$
and you get $\partial L/\partial b \approx -0.0817$ — exactly what PyTorch prints (book p. 264).
See [[computational-graph]] and [[derivative]].

### Sigmoid vs softmax, and why models return logits
- **Binary vs multi-class.** Sigmoid gives one probability for "class 1"; [[softmax]] gives a
  whole probability distribution over $K$ classes. The book: PyTorch's loss functions combine
  "softmax (or sigmoid for binary classification)" with the negative log-likelihood loss in
  one class (book p. 269).
- **So don't apply it yourself before the loss.** Models usually return raw [[logits]]; the
  loss applies sigmoid/softmax internally "for numerical efficiency and stability"
  (book p. 269). For binary problems that's `F.binary_cross_entropy_with_logits`; the book's
  Listing A.2 instead applies `torch.sigmoid` explicitly and then `F.binary_cross_entropy`, which
  is fine for a teaching example.
- **In modern LLMs** the sigmoid hides inside other activations: the [[gelu]] is often
  approximated as $z\,\sigma(1.702z)$, and [[swiglu]] uses the Swish/SiLU gate $z\,\sigma(z)$.
  Plain sigmoid hidden layers have mostly been replaced by [[relu]] and GELU.

## 🔴 Theory (Expert)
- **Range and symmetry.** $\sigma:\mathbb{R}\to(0,1)$ is a smooth ($C^\infty$), strictly
  increasing bijection onto the **open** interval $(0,1)$ — it never actually reaches 0 or 1
  (the MML-based [[generalized-linear-model]] page writes $[0,1]$; the endpoints are only limits).
  It satisfies $\sigma(-z) = 1-\sigma(z)$, so it is point-symmetric about $(0, 0.5)$.
- **Inverse = logit (log-odds).** $\sigma^{-1}(p) = \ln\frac{p}{1-p}$. That is why the net input
  $z$ is called a **logit**: it is the log-odds of class 1. In GLM language the logit is the
  canonical link of the Bernoulli distribution ([[generalized-linear-model]]).
- **Relation to tanh and softmax.** $\tanh(z) = 2\sigma(2z)-1$, and for two classes
  $\text{softmax}(z_0,z_1)_1 = \frac{e^{z_1}}{e^{z_0}+e^{z_1}} = \sigma(z_1-z_0)$ — softmax is the
  multi-class generalisation of the sigmoid.
- **Derivative bound and vanishing gradients.** $\sigma'(z) = \sigma(z)(1-\sigma(z)) \le 1/4$,
  with the maximum at $z=0$, and $\sigma'(z)\to 0$ as $|z|\to\infty$ (saturation). In a deep stack
  of sigmoid layers, backpropagated gradients pick up a factor $\le 1/4$ per layer, so they shrink
  geometrically — the classic cause of [[vanishing-and-exploding-gradients]] and one reason
  [[relu]]/[[gelu]] and [[shortcut-connections]] took over.
- **Pairing with BCE cancels the saturation.** With $L = -[y\ln a + (1-y)\ln(1-a)]$, the chain rule
  gives $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\,a(1-a) = a - y$. The small
  $a(1-a)$ factor is exactly cancelled by the $1/a$ and $1/(1-a)$ in $\partial L/\partial a$, so the
  output layer still learns quickly even when it is confidently wrong. Check with the book's
  numbers: $a-y = 0.9183-1 = -0.0817$ ✓ (book p. 264).
- **Numerical stability.** Computing $\ln\sigma(z)$ naively fails for very negative $z$
  ($\sigma(z)$ underflows to 0, $\ln 0 = -\infty$). The stable form is
  $\ln\sigma(z) = -\operatorname{softplus}(-z) = -\ln(1+e^{-z})$, evaluated with a log-sum-exp style
  trick — the reason the "with logits" losses exist ([[numerical-stability]], book p. 269).
- **Probabilistic origin.** If the class-conditional densities of $\mathbf{x}$ are Gaussians with a
  shared covariance, Bayes' rule gives a posterior $P(y=1\mid\mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x}+b)$
  exactly — one theoretical reason the sigmoid is the "natural" squashing function
  ([[gaussian-distribution]]).

> **From [[llm-ch04-implementing-a-gpt-model]]:** the book names [[swiglu]] as a "sigmoid-gated linear unit" (book p. 105). Here the sigmoid $\sigma(z) = 1/(1+e^{-z})$ acts as a soft gate in $(0,1)$ that scales another linear projection. The [[gelu]] tanh approximation is also a sigmoid in disguise: $\tfrac12(1+\tanh z) = \sigma(2z)$.

## Connections
- Builds on: [[derivative]], [[probability-distribution]], [[monotonic-function]]
- Leads to: [[logistic-regression]], [[binary-cross-entropy]], [[softmax]], [[activation-function]]
- Used by: [[computational-graph]] (Fig. A.7–A.8 example), [[gelu]], [[swiglu]], [[generalized-linear-model]], [[neural-network]]
- Related: [[logits]], [[relu]], [[vanishing-and-exploding-gradients]], [[numerical-stability]], [[chain-rule]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3 (sigmoid as activation in the logistic-regression forward pass, Listing A.2, Figure A.7), book pp. 261–262
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4 (the $da/dz$ edge in Figure A.8; gradients −0.0898 / −0.0817 in Listing A.3), book pp. 263–264
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (losses combine softmax or sigmoid with NLL; models return logits), book p. 269
- `raw/mml-book.pdf` §9.5 Further Reading (logistic sigmoid in generalized linear models; Eq. 9.72), printed pp. 315–316
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book p. 105 (SwiGLU = "sigmoid-gated linear unit", named as a GELU alternative in modern LLMs).
