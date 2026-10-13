---
title: Logistic Regression
type: concept
tags: [machine-learning, classification, probability, neural-networks]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch, mml-ch09-linear-regression]
created: 2026-10-13
updated: 2026-10-13
---

# Logistic Regression

> **One-liner:** Logistic regression adds up the inputs with some weights, then squashes the total into a number between 0 and 1 that you can read as "the chance this is a yes".

## 🟢 Intuition (Dummy)
Say you want to guess whether an email is spam. You look at a few clues — how many times
it says "free", whether it has a weird link — and give each clue a weight: big positive
weights push toward "spam", negative ones toward "not spam". Add everything up and you get
a score that could be anything: −7, 0.3, 42…

That raw score isn't easy to read. So logistic regression feeds it through an S-shaped
"squasher", the [[sigmoid-function]], which turns any score into a number between 0 and 1.
Large positive score → close to 1 ("very likely spam"). Large negative → close to 0. A
score of exactly 0 → 0.5 ("coin flip").

Training means finding weights that make these numbers close to 1 for real spam and close
to 0 for real non-spam. The book uses logistic regression as the smallest possible neural
network — "a single-layer neural network" — to show how PyTorch computes gradients
(book p. 262).

## 🟡 Mechanics (Practitioner)

### The model
With one input feature (the book's example, §A.3, book p. 262):

$$z = w_1 x_1 + b,\qquad a = \sigma(z) = \frac{1}{1+e^{-z}}$$

Symbol by symbol:
- $x_1$ — the input feature (a number describing the example).
- $w_1$ — the **weight**: how much this feature matters (trainable).
- $b$ — the **bias**: a baseline shift, the score when $x_1 = 0$ (trainable).
- $z$ — the **net input** (the raw score, also called a [[logits|logit]]).
- $\sigma$ — the sigmoid function; $e$ ≈ 2.718 is Euler's number; $e^{-z}$ shrinks as $z$ grows.
- $a$ — the **activation and output**: a score between 0 and 1, read as $P(y=1\mid x_1)$.

With many features it's the same thing with a [[inner-product|dot product]]:
$z = \mathbf{w}^\top\mathbf{x} + b = w_1x_1 + \dots + w_dx_d + b$.

### The loss: binary cross-entropy
The output $a$ is "compared to the true class label (0 or 1) when computing the loss"
(book p. 262). PyTorch's `F.binary_cross_entropy(a, y)` computes (standard definition; the
book names the function):

$$L(a, y) = -\big[\,y\ln a + (1-y)\ln(1-a)\,\big]$$

Reading it: $y$ is the true label (0 or 1); $\ln$ is the natural logarithm. If $y=1$, only
the first term survives: $L = -\ln a$, small when $a$ is near 1. If $y=0$, only the second
term survives: $L = -\ln(1-a)$, small when $a$ is near 0. See [[binary-cross-entropy]].

### The book's worked example (Listings A.2–A.3, book pp. 262–264)
```python
import torch.nn.functional as F
from torch.autograd import grad

y  = torch.tensor([1.0])                       # true label
x1 = torch.tensor([1.1])                       # input feature
w1 = torch.tensor([2.2], requires_grad=True)   # weight parameter
b  = torch.tensor([0.0], requires_grad=True)   # bias unit
z = x1 * w1 + b                                # net input
a = torch.sigmoid(z)                           # activation and output
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)  # (tensor([-0.0898]),)
grad_L_b  = grad(loss, b,  retain_graph=True)  # (tensor([-0.0817]),)
```

Following the numbers (intermediate values computed here; the gradients are printed in the book):
1. $z = 1.1 \times 2.2 + 0 = 2.42$
2. $a = \sigma(2.42) = 1/(1+e^{-2.42}) \approx 0.9183$ → the model says "91.8% class 1".
3. True label $y = 1$, so $L = -\ln(0.9183) \approx 0.0852$ — a small loss, the guess is good.
4. Gradients: $\partial L/\partial b \approx -0.0817$, $\partial L/\partial w_1 \approx -0.0898$.

The minus signs mean "increasing $w_1$ or $b$ would *lower* the loss", so
[[gradient-descent]] ($\theta \leftarrow \theta - \eta\,\nabla_\theta L$) will nudge both up
— pushing $a$ even closer to 1.

### The famous shortcut
Combining the sigmoid with BCE gives a beautifully simple gradient:

$$\frac{\partial L}{\partial z} = a - y,\qquad \frac{\partial L}{\partial w_1} = (a-y)\,x_1,\qquad \frac{\partial L}{\partial b} = a - y$$

"Prediction minus truth, times the input". Check: $a - y = 0.9183 - 1 = -0.0817$ ✓ and
$(a-y)x_1 = -0.0817 \times 1.1 = -0.0898$ ✓ — exactly the book's printout.

### Where it shows up in the LLM book
- §A.3–A.4 use it as the **running example** of a [[computational-graph]] and of
  [[automatic-differentiation]]. The book stresses the goal is *not* to implement logistic
  regression, but to show how a sequence of computations becomes a graph (book p. 262).
- The multi-class version (softmax regression) is what every classifier head does:
  the MLP of §A.5 ends in logits that go through [[softmax]] + [[cross-entropy-loss]], and the
  spam classifier of chapter 6 is a 2-logit head on top of GPT-2 (see [[classification-head]]).
- PyTorch's habit of returning **logits** and letting the loss apply sigmoid/softmax
  internally ("for numerical efficiency and stability", book p. 269) comes straight from
  this model.

## 🔴 Theory (Expert)
- **Probabilistic model.** Logistic regression is a [[generalized-linear-model]]: a Bernoulli
  likelihood $y \mid \mathbf{x} \sim \text{Bernoulli}(\sigma(\mathbf{w}^\top\mathbf{x}+b))$ with the
  logit (log-odds) as canonical link: $\ln\frac{p}{1-p} = \mathbf{w}^\top\mathbf{x}+b$. So each
  weight is an additive change in **log-odds** per unit of its feature.
- **Loss = negative log-likelihood.** For data $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, the likelihood is
  $\prod_i a_i^{y_i}(1-a_i)^{1-y_i}$; taking $-\ln$ and averaging gives the mean BCE. Minimising
  BCE is therefore [[maximum-likelihood-estimation]] — and also [[empirical-risk-minimization]]
  with the log loss.
- **Gradient and Hessian.** With $\mathbf{a} = \sigma(X\mathbf{w})$ (bias absorbed),
  $\nabla_\mathbf{w} L = \frac1N X^\top(\mathbf{a}-\mathbf{y})$ and
  $\nabla^2_\mathbf{w} L = \frac1N X^\top S X$ with $S = \operatorname{diag}(a_i(1-a_i)) \succeq 0$. The
  Hessian is positive semidefinite, so the loss is **convex**: any local minimum is global, and
  Newton's method (IRLS) converges quickly. Deep networks lose this convexity.
- **Linearly separable data has no finite MLE.** If a hyperplane separates the classes
  perfectly, scaling $\mathbf{w}\to c\mathbf{w}$ with $c\to\infty$ keeps lowering the loss toward 0, so
  the weights diverge. In practice [[regularization]] (an L2 penalty, or AdamW's
  [[weight-decay]]) keeps them finite. The book's 5-example toy dataset in §A.6 (book p. 270) is
  linearly separable (class 1 exactly when the first feature is positive), and its MLP's last
  printed batch loss is 0.00 after 3 epochs (book p. 275).
- **Decision boundary.** $a > 0.5 \iff z > 0 \iff \mathbf{w}^\top\mathbf{x}+b > 0$: the boundary is a
  hyperplane, so logistic regression is a *linear classifier* despite the nonlinear sigmoid.
  Because $\sigma$ is [[monotonic-function|monotonic]], predicting with $\arg\max$ over logits or
  over probabilities gives the same label (book p. 277).
- **Multi-class generalisation.** Replace $\sigma$ by [[softmax]] over $K$ logits $\mathbf{z} = W\mathbf{x}+\mathbf{b}$:
  $p(k\mid\mathbf{x}) = e^{z_k}/\sum_j e^{z_j}$. For $K=2$ this reduces to the sigmoid of the logit
  difference: $\text{softmax}(\mathbf{z})_1 = \sigma(z_1 - z_0)$.
- **Neural-network view.** A logistic-regression unit is one neuron with a sigmoid
  [[activation-function]]; stacking such units with hidden layers gives a
  [[multilayer-perceptron]] (book p. 262, §A.5).

## Connections
- Builds on: [[linear-mapping]], [[inner-product]], [[sigmoid-function]], [[probability-distribution]]
- Leads to: [[softmax]], [[cross-entropy-loss]], [[multilayer-perceptron]], [[classification-head]], [[text-classification]]
- Used by: [[computational-graph]], [[automatic-differentiation]], [[backpropagation]] (as the book's running example)
- Related: [[binary-cross-entropy]], [[maximum-likelihood-estimation]], [[generalized-linear-model]], [[logits]], [[gradient-descent]], [[supervised-learning]], [[neural-network]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3 (logistic regression as a single-layer network, Listing A.2, Figure A.7), book pp. 261–262
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4 (gradients via autograd, Listing A.3, Figure A.8), book pp. 263–264
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (losses combine sigmoid/softmax with NLL), book p. 269; §A.6 (toy dataset), book p. 270; §A.7 (training log, argmax over logits), book pp. 275, 277
- `raw/mml-book.pdf` §9.5 Further Reading (generalized linear models, logistic sigmoid; Eq. 9.72), printed pp. 315–316 — see [[generalized-linear-model]]
