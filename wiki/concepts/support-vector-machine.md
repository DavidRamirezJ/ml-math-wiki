---
title: Support Vector Machine
type: concept
tags: [classification, support-vector-machine, supervised-learning, machine-learning]
difficulty: intermediate
sources: [mml-ch12-classification-with-support-vector-machines]
created: 2026-06-04
updated: 2026-06-04
---

# Support Vector Machine

> **One-liner:** A support vector machine draws the straightest dividing line between two groups of dots so that the line sits as far as possible from the nearest dot on either side.

## 🟢 Intuition (Dummy)
Imagine red and blue marbles scattered on a table, and you must draw one straight line so all reds
land on one side and all blues on the other. Usually *many* lines work. Which is best? The one that
leaves the widest empty street down the middle — the line that keeps its distance from the closest
marbles of both colors. That widest-street line is what an SVM finds. A new marble is then classified
just by which side of the street it lands on. The few marbles that touch the edges of the street are
the only ones that matter; the rest could be moved around freely without changing the line.

## 🟡 Mechanics (Practitioner)
The SVM is a **binary classifier** $f:\mathbb R^D\to\{+1,-1\}$. Given example–label pairs
$\{(x_n,y_n)\}_{n=1}^N$ with $y_n\in\{+1,-1\}$, it learns a [[separating-hyperplane|hyperplane]]
$$ f(x)=\langle w,x\rangle + b, \qquad \hat y=\operatorname{sign} f(x), $$
choosing $w,b$ to **maximize the [[margin]]** — the distance to the closest example. After the
standard scaling that puts the closest point at $y(\langle w,x\rangle+b)=1$, the training problem is

- **Hard margin** (separable data): $\min_{w,b}\tfrac12\|w\|^2$ s.t. $y_n(\langle w,x_n\rangle+b)\ge 1$ — see [[hard-margin-svm]].
- **Soft margin** (the practical one): add slack $\xi_n\ge0$ and a penalty $C$ — see [[soft-margin-svm]].

Both are **convex** problems (a [[quadratic-programming|quadratic program]]); there is no closed-form
solution, so we use [[convex-optimization|numerical optimization]]. Two equivalent routes exist: solve
in the **primal** ($w,b$, size $D$) or in the **[[dual-support-vector-machine|dual]]** ($\alpha$, size $N$).
The dual depends only on **inner products** between examples, which enables the [[kernel-trick]] for
nonlinear boundaries.

**Worked example (1D).** Two points: $x_+ = 2$ ($y=+1$), $x_- = 0$ ($y=-1$). The max-margin separator
sits at the midpoint $x=1$, so $b=-w$ and $\langle w,x\rangle+b = w(x-1)$. Unit-margin scaling needs
$w(2-1)=1\Rightarrow w=1,\ b=-1$. Margin $=1/\|w\|=1$ (half the gap, as expected).

**ML usage.** State-of-the-art on many small/medium tabular and text problems; robust because only the
**[[support-vector|support vectors]]** (boundary points) determine the model. The hyperparameter $C$
trades margin width against training errors and is tuned by [[cross-validation]].

## 🔴 Theory (Expert)
The SVM is an instance of **regularized [[empirical-risk-minimization]]**: minimize the [[hinge-loss]]
(a convex surrogate upper bound for the non-convex zero-one loss) plus an $\ell_2$ [[regularization|regularizer]],
$$ \min_{w,b}\ \underbrace{\tfrac12\|w\|^2}_{\text{margin = regularizer}} + C\sum_{n=1}^N \max\{0,\,1-y_n(\langle w,x_n\rangle+b)\}. $$
Margin maximization and regularization are thus *the same term* — large margin = low function-class
complexity (Vapnik–Chervonenkis theory; [[vladimir-vapnik]]), which is why large margin generalizes.

By [[lagrange-multipliers|Lagrangian duality]] the optimal weight obeys the **representer theorem**
$w^\*=\sum_n\alpha_n y_n x_n$, so $w^\*$ lies in the span of the data and depends only on inner products —
the structural fact behind both the dual and kernelization. Unlike the maximum-likelihood classifiers of
Ch. 9, the SVM is **non-probabilistic**: its score $\langle w,x\rangle+b$ is not a calibrated probability
(post-hoc calibration, e.g. Platt scaling, is needed). It contrasts with logistic regression, which uses
the logistic loss and *does* model $P(Y=1\mid x)$.

## Connections
- Builds on: [[separating-hyperplane]], [[inner-product]], [[empirical-risk-minimization]], [[convex-optimization]]
- Leads to: [[hard-margin-svm]], [[soft-margin-svm]], [[dual-support-vector-machine]], [[kernel-trick]]
- Used by: binary classification (spam, cancer detection, text categorization)
- People: [[vladimir-vapnik]], [[joseph-louis-lagrange]]

## Sources
- `raw/mml-book.pdf` §12 intro & §12.2, printed pp. 370–382 (PDF pp. 376–388).
