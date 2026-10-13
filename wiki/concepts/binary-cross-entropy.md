---
title: Binary Cross-Entropy
type: concept
tags: [loss-function, classification, probability, deep-learning, pytorch]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Binary Cross-Entropy

> **One-liner:** Binary cross-entropy is a score for yes/no predictions that is tiny when the model confidently says the right answer and huge when it confidently says the wrong one.

## 🟢 Intuition (Dummy)
Picture a weather app that says "90% chance of rain". If it rains, the app did well. If it
stays sunny, the app looks silly, and it looks much sillier than an app that only said
"55%". Binary cross-entropy (BCE) is a way of turning that "how silly do you look?"
feeling into a number.

- The model gives a probability $a$ between 0 and 1 ("how sure am I that the answer is *yes*?").
- The truth $y$ is either 1 (yes) or 0 (no).
- BCE asks: "How much probability did you put on what *actually* happened?" Put a lot on it and
  the penalty is close to 0. Put almost nothing on it and the penalty shoots up toward infinity.

That "confidently wrong = heavily punished" behaviour is exactly what you want for training a
classifier: it pushes the model hard away from bad, overconfident guesses. In the LLM book it is
the loss used in the tiny [[logistic-regression]] example that Appendix A uses to show how
PyTorch builds a [[computational-graph]] and computes gradients.

## 🟡 Mechanics (Practitioner)

### The formula
$$L(a, y) = -\big[\,y \ln a + (1-y)\ln(1-a)\,\big]$$

Reading it symbol by symbol:
- $L(a, y)$ = "the loss, which depends on the prediction $a$ and the label $y$".
- $a$ = the model's output, a probability in $(0, 1)$. In the book it comes out of the [[sigmoid-function]]: $a = \sigma(z)$.
- $y$ = the true label, either $0$ or $1$.
- $\ln$ = the natural logarithm (log base $e$). $\ln$ of a number close to 1 is close to 0; $\ln$ of a number close to 0 is a big negative number.
- $y \ln a$ = "if the truth is 1, look at how much probability you gave to 1". When $y = 0$ this term vanishes.
- $(1-y)\ln(1-a)$ = "if the truth is 0, look at how much probability you gave to 0". When $y = 1$ this term vanishes.
- The leading minus sign flips the (negative) log into a positive penalty.

So the formula is really two cases glued together:
$$L = \begin{cases} -\ln a & \text{if } y = 1 \\ -\ln(1-a) & \text{if } y = 0 \end{cases}$$

For a batch of $N$ examples, PyTorch's default (`reduction='mean'`) takes the average:
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\big[\,y_i \ln a_i + (1-y_i)\ln(1-a_i)\,\big]$$
- $N$ = number of examples, $i$ = which example, $\sum_{i=1}^{N}$ = "add up over all examples", $\frac{1}{N}$ = "then divide by $N$" (an average).

> The book uses `F.binary_cross_entropy` but does not write out the formula; the formula above is
> the standard definition (the book only calls it at book p. 262).

### The book's worked example (Listings A.2–A.3, book pp. 262–264)
The book builds a one-input logistic-regression "network" (Figure A.7):
$$u = w_1 x_1,\qquad z = u + b,\qquad a = \sigma(z),\qquad \text{loss} = L(a, y)$$

with $y = 1.0$, $x_1 = 1.1$, $w_1 = 2.2$, $b = 0.0$:

| quantity | value |
|---|---|
| $z = 1.1 \times 2.2 + 0$ | $2.42$ |
| $a = \sigma(2.42) = 1/(1+e^{-2.42})$ | $\approx 0.9183$ |
| $L = -\ln(0.9183)$ (because $y = 1$) | $\approx 0.0852$ |

(The intermediate values are computed by hand; the book prints only the gradients.)

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

### The gradient: why BCE + sigmoid is so neat
Differentiate $L$ with respect to $a$:
$$\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}$$
- $\frac{\partial L}{\partial a}$ = "how much the loss changes if we nudge the prediction $a$" (a [[partial-derivative]]).

The sigmoid's own [[derivative]] is $\frac{da}{dz} = \sigma(z)\,(1-\sigma(z)) = a(1-a)$. Multiplying them
with the [[chain-rule]], almost everything cancels:
$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a}\cdot\frac{da}{dz} = a - y$$
- In words: "the error signal at the net input is simply *prediction minus truth*."

Book numbers: $\frac{\partial L}{\partial a} = -1/0.9183 \approx -1.0890$, $\frac{da}{dz} \approx 0.0750$, so
$\frac{\partial L}{\partial z} = 0.9183 - 1 \approx -0.0817$. Then
- $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z}\cdot 1 \approx -0.0817$ ✓ (book prints `tensor([-0.0817])`)
- $\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z}\cdot x_1 \approx -0.0817 \times 1.1 \approx -0.0898$ ✓ (book prints `tensor([-0.0898])`)

Both gradients are negative, which tells [[gradient-descent]] to *increase* $w_1$ and $b$, pushing $a$ even
closer to the correct label 1.

### Where it shows up
- Binary (two-class) classification with a single sigmoid output, like the A.3/A.4 example.
- The multi-class version is [[cross-entropy-loss]] (softmax + negative log-likelihood), which is what
  the book uses everywhere else: `F.cross_entropy` in the A.7 training loop and for next-token
  prediction in the GPT chapters. A.5 notes that PyTorch's loss functions fuse sigmoid/softmax with
  the negative log-likelihood "for numerical efficiency and stability" (book p. 269), which is why models
  usually return [[logits]].

## 🔴 Theory (Expert)
- **It is a negative log-likelihood.** Model the label as a Bernoulli random variable,
  $P(y \mid a) = a^{y}(1-a)^{1-y}$. Then $-\ln P(y\mid a) = -[y\ln a + (1-y)\ln(1-a)]$. Minimizing
  mean BCE over a dataset is exactly [[maximum-likelihood-estimation]] for logistic regression, and
  an instance of [[empirical-risk-minimization]] with the log loss as [[loss-function]].
- **It is a cross-entropy.** For two distributions $p=(y, 1-y)$ (the truth) and $q=(a, 1-a)$ (the model),
  $H(p, q) = -\sum_k p_k \ln q_k$ gives BCE. Since $H(p,q) = H(p) + D_{\mathrm{KL}}(p\,\|\,q)$ and a hard
  label has $H(p)=0$, minimizing BCE minimizes the KL divergence from truth to model.
- **Convexity.** As a function of the logit $z$, $\ell(z) = \ln(1+e^{-z})$ for $y=1$ (and $\ln(1+e^{z})$ for
  $y=0$); its second derivative is $\sigma(z)(1-\sigma(z)) > 0$, so logistic regression with BCE is a
  convex problem in $(w, b)$ (see [[convex-optimization]]).
- **Gradient $a - y$.** This clean form is the canonical-link property of [[generalized-linear-model]]s
  (logit link for the Bernoulli member of the [[exponential-family]]): the gradient w.r.t. the natural
  parameter is "mean minus observation". It also explains why MSE + sigmoid trains worse: there the
  gradient carries an extra $a(1-a)$ factor that vanishes when the sigmoid saturates.
- **Numerical stability.** Computing $\sigma$ then $\ln$ separately can produce $\ln 0 = -\infty$ when
  $|z|$ is large. The stable form works on logits: $\ell(z, y) = \max(z, 0) - zy + \ln(1+e^{-|z|})$
  (this is what `F.binary_cross_entropy_with_logits` does). PyTorch's plain `F.binary_cross_entropy`
  clamps the log at $-100$ for this reason. See [[numerical-stability]].

## Connections
- Builds on: [[sigmoid-function]], [[logistic-regression]], [[probability-distribution]], [[loss-function]], [[maximum-likelihood-estimation]]
- Leads to: [[cross-entropy-loss]], [[backpropagation]], [[gradient-descent]]
- Used by: [[computational-graph]] example in Appendix A, [[automatic-differentiation]] demo (`torch.autograd.grad`), binary [[text-classification]]
- Related: [[log-probability]], [[logits]], [[softmax]], [[numerical-stability]], [[chain-rule]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.3, book p. 262 (Listing A.2, Figure A.7: logistic-regression forward pass with `F.binary_cross_entropy`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4, book pp. 263–264 (Figure A.8 chain rule; Listing A.3 gradients −0.0898 and −0.0817).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book p. 269 (losses fuse sigmoid/softmax with negative log-likelihood).
- BCE formula, Bernoulli/KL/convexity derivations: standard results, not written out in the book.
