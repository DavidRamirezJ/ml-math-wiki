---
title: Mean Squared Error
type: concept
tags: [loss-function, statistics, optimization, regression]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, mml-ch09-linear-regression]
created: 2026-10-13
updated: 2026-10-13
---

# Mean Squared Error

> **One-liner:** Mean squared error measures how wrong your predictions are by taking each miss, squaring it, and averaging the squares.

## 🟢 Intuition (Dummy)
You throw darts at a target. For each dart, measure how far it landed from the bullseye.
Square that distance (so a miss of 2 cm counts as 4, a miss of 10 cm counts as 100), then
take the average over all darts. That average is the **mean squared error (MSE)**.

Two things to notice:
- A perfect throw scores 0. Bigger numbers mean worse throws.
- Squaring punishes **big** misses much more than small ones, and it makes left and right
  misses count the same (the square is never negative).

In machine learning, MSE is a [[loss-function]]: a single number that says "how far are
the model's outputs from the targets?" Training tries to push that number down.

## 🟡 Mechanics (Practitioner)

### The formula
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^{n} (\hat y_i - y_i)^2$$
Symbol by symbol:
- $\mathcal L_{\text{MSE}}$ = the loss value (one number).
- $n$ = how many output values we compare.
- $\sum_{i=1}^{n}$ = "add up the following for $i=1,2,\dots,n$".
- $\hat y_i$ ("y-hat") = the model's prediction for item $i$.
- $y_i$ = the true target for item $i$.
- $(\hat y_i - y_i)^2$ = the miss for item $i$, squared.
- $\frac1n$ = divide by $n$ to get the **mean** ([[expected-value-and-mean]]).

**Tiny example.** Predictions $(2, 0, 5)$, targets $(1, 0, 2)$. Misses: $1, 0, 3$. Squares:
$1, 0, 9$. MSE $=(1+0+9)/3 = 10/3 \approx 3.33$.

### Its gradient
$$\frac{\partial \mathcal L_{\text{MSE}}}{\partial \hat y_i} = \frac{2}{n}(\hat y_i - y_i)$$
- $\partial/\partial\hat y_i$ = the [[partial-derivative]]: "how fast does the loss change if this one prediction moves?"
- The answer is proportional to the miss itself: a big miss gives a big push back, zero miss gives zero push. This is the starting signal for [[backpropagation]].

### How the LLM book uses it (§4.4, book pp. 111–112)
In the [[shortcut-connections]] experiment, the book needs *some* loss to compute
gradients, so it uses PyTorch's built-in MSE against a dummy target of 0:
```python
def print_gradients(model, x):
    output = model(x)                    # forward pass, shape (1, 1)
    target = torch.tensor([[0.]])
    loss = nn.MSELoss()
    loss = loss(output, target)          # how close output and target are
    loss.backward()                      # compute gradients
    ...
```
Here $n = 1$ (the toy network returns one number) and $y = 0$, so the loss is just
$\mathcal L = \hat y^2$ and its gradient is $2\hat y$. The target is chosen "for
simplicity" — the point is not to train, only to watch how big the gradients are in each
layer (book p. 112).

`nn.MSELoss()` averages over all elements by default (`reduction='mean'`), exactly the
$\frac1n\sum$ above.

### Where MSE is *not* used in the book
For predicting the next [[token]], the book uses [[cross-entropy-loss]] instead (Ch. 5).
The output is a choice among 50,257 tokens — a classification problem — and cross-entropy
fits probabilities far better than squared error does. MSE is the default for
**regression** (predicting real numbers), as in [[linear-regression]].

## 🔴 Theory (Expert)
- **MSE = Gaussian maximum likelihood.** If targets are $y_i = g(\mathbf x_i)+\varepsilon_i$
  with noise $\varepsilon_i\sim\mathcal N(0,\sigma^2)$, the negative log-likelihood is
  $\frac{1}{2\sigma^2}\sum_i (y_i-g(\mathbf x_i))^2 + \text{const}$. So minimizing MSE is
  the same as [[maximum-likelihood-estimation]] under a [[gaussian-distribution|Gaussian]]
  noise model (MML §9.2, see [[linear-regression]]). Cross-entropy plays the same role
  for a categorical model — both are "negative log-likelihood" losses.
- **The best constant is the mean.** $\min_c \mathbb E[(Y-c)^2]$ is attained at
  $c=\mathbb E[Y]$, and the minimum value is $\operatorname{Var}[Y]$
  ([[covariance-and-variance]]). More generally, the MSE-optimal predictor is the
  conditional mean $\mathbb E[Y\mid X]$.
- **Bias–variance decomposition.** For an estimator $\hat\theta$ of $\theta$:
  $\operatorname{MSE}(\hat\theta)=\operatorname{Var}[\hat\theta] + \big(\mathbb E[\hat\theta]-\theta\big)^2$.
  This is the formal root of the trade-off discussed in [[overfitting-and-underfitting]].
- **Geometry.** $n\cdot\mathcal L_{\text{MSE}}=\lVert\hat{\mathbf y}-\mathbf y\rVert_2^2$, the squared
  Euclidean [[norm]] of the residual vector. Least squares is therefore an
  [[orthogonal-projection]] of $\mathbf y$ onto the column space of the design matrix.
- **Sensitivity to outliers.** The quadratic penalty makes MSE dominated by large errors;
  mean absolute error (median-optimal) or Huber loss are robust alternatives.
- **Why not MSE for classification.** With a softmax/sigmoid output, MSE's gradient
  contains the factor $\sigma'(z)$, which vanishes when the model is confidently wrong;
  cross-entropy cancels that factor, giving gradient $\hat p - y$
  ([[binary-cross-entropy]], [[logistic-regression]]).
- It is an instance of [[empirical-risk-minimization]] with squared loss.

## Connections
- Builds on: [[loss-function]], [[expected-value-and-mean]], [[norm]], [[partial-derivative]]
- Leads to: [[linear-regression]], [[regularized-least-squares]], [[backpropagation]]
- Related: [[maximum-likelihood-estimation]], [[gaussian-distribution]], [[cross-entropy-loss]], [[empirical-risk-minimization]], [[covariance-and-variance]]
- Used by: [[shortcut-connections]] (gradient experiment), [[linear-regression]]
- People: [[carl-friedrich-gauss]] (least squares)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book pp. 111–112 (`print_gradients`, `nn.MSELoss`, target 0).
- `raw/mml-book.pdf` §9.2 (least squares as maximum likelihood with Gaussian noise) — via [[mml-ch09-linear-regression]].
- Gradient, bias–variance, outlier and classification remarks: standard results, not in the LLM book.
