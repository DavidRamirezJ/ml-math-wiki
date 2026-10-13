---
title: Surrogate (Proxy) Loss
type: concept
tags: [optimization, loss-function, classification, calculus, llm]
difficulty: intermediate
sources: [llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Surrogate (Proxy) Loss

> **One-liner:** A surrogate loss is a smooth stand-in you train on, because the score you actually care about (like accuracy) is too "blocky" for gradients to work with.

## 🟢 Intuition (Dummy)
Say you're looking for the lowest point in a valley, but you're blindfolded. You can
only feel the slope under your feet. If the ground is a smooth hill, the slope tells
you which way to step. If the ground is made of perfectly flat stair steps, your feet
feel *no* slope at all, almost everywhere. You'd have no idea which way to go.

**Accuracy is the staircase.** If you change the model's weights a tiny bit, almost
every prediction stays the same, so accuracy doesn't move at all. Its slope is zero.
[[gradient-descent|Gradient descent]] has nothing to work with.

So we train on a **smooth hill** that goes down whenever accuracy tends to go up:
[[cross-entropy-loss]]. The book says it directly: "Because classification
accuracy is not a differentiable function, we use cross-entropy loss as a proxy to
maximize accuracy" (book p. 193). The smooth stand-in is called a *surrogate* or
*proxy* loss.

## 🟡 Mechanics (Practitioner)

### The objective we want vs. the one we train on
What we *want*: maximise [[classification-accuracy]], or equivalently minimise the
average **0-1 loss**:

$$\mathcal{L}_{0\text{-}1}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\arg\max_k z^{(i)}_k(\theta) \ne y^{(i)}\right]$$

- $\theta$ = the model's trainable parameters.
- $z^{(i)}_k(\theta)$ = the [[logits|logit]] for class $k$ on example $i$. Writing $(\theta)$ is a reminder that it depends on the weights.
- $\arg\max_k$ = the predicted class, the one with the biggest logit.
- $\mathbb{1}[\ldots \ne y^{(i)}]$ = 1 if the prediction is **wrong**, 0 if it's right.
- The average of that = the fraction of mistakes = $1 - \text{accuracy}$.

What we *train on*: the cross-entropy surrogate (book pp. 193–194):

$$\mathcal{L}_{\text{CE}}(\theta) = \frac{1}{N}\sum_{i=1}^{N} -\log p^{(i)}_{y^{(i)}}, \qquad p^{(i)}_{k} = \frac{e^{z^{(i)}_k}}{\sum_{j} e^{z^{(i)}_j}}$$

- $p^{(i)}_k$ = the [[softmax]] probability the model gives class $k$ on example $i$.
- $p^{(i)}_{y^{(i)}}$ = the probability it gives the **correct** class.
- $-\log$ = turns "probability of the right answer" into a penalty. A probability of 1
  gives a penalty of 0. A probability near 0 gives a huge penalty. See [[log-probability]].
- $\frac{1}{N}\sum$ = average over the examples.

### Why one works with gradients and the other doesn't
- $\mathcal{L}_{0\text{-}1}$ is **piecewise constant**. Its [[gradient]] is $0$ almost everywhere,
  and it jumps (it's undefined) exactly where two logits tie. Useless for
  [[backpropagation]].
- $\mathcal{L}_{\text{CE}}$ is smooth. For one example, its gradient with respect to the logits is
  very simple:

$$\frac{\partial \ell}{\partial z_k} = p_k - \mathbb{1}[k = y]$$

  - $\frac{\partial \ell}{\partial z_k}$ = how much the loss changes if you nudge logit $k$ a little ([[partial-derivative]]).
  - $p_k - \mathbb{1}[k=y]$ = "predicted probability minus target". For the correct class
    that's $p_y - 1$ (negative, so push that logit up). For a wrong class it's $p_k$
    (positive, so push it down).

### Worked example with the book's numbers
The untrained classifier gives the last-token logits $\mathbf{z} = [-3.5983,\ 3.9902]$ for
"Do you have time" and predicts class 1, "spam" (book pp. 188, 192). The true label is 0, "not spam": in
Fig. 6.14 the same text "Do you have time" is predicted "spam", and the caption says the
untrained model "predicts the spam labels incorrectly" (book p. 191). (Fig. 6.14 prints
illustrative logits `[-3.9846, 5.2940]` for this text; we use the code output from p. 192.)

- Softmax: $p_0 = \dfrac{e^{-3.5983}}{e^{-3.5983} + e^{3.9902}} = \dfrac{1}{1 + e^{7.5885}} \approx 0.0005$, $\ p_1 \approx 0.9995$.
- 0-1 loss: $1$ (wrong). If you nudge the logits by $\pm 0.1$, it's still exactly $1$, so there's no signal.
- Cross-entropy: $-\log p_0 \approx 7.59$. That's large, and it shrinks smoothly as $z_0$ goes up.
- Gradient: $\partial \ell / \partial \mathbf{z} \approx [0.0005 - 1,\ 0.9995] = [-0.9995,\ +0.9995]$, a strong push toward "not spam".

### In the book's code
The surrogate is computed on the [[last-token-pooling|last token's]] logits only. That's
the only change from the pretraining loss in chapter 5 (book pp. 193–194):

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]          # [B, 2]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
```

`cross_entropy` takes the **raw logits** and applies log-softmax inside. The initial
losses (5 batches each) are **2.453 / 2.583 / 2.322** for train / val / test
(book p. 194). Compare that to $\ln 2 \approx 0.693$, which is what a model that
always says 50/50 would get. So the random head is *confidently wrong* on a lot of
examples. After training, the loss falls and accuracy rises along with it: the
book's goal of minimising the training-set loss "will help increase the
classification accuracy" (book p. 194). During the 5-epoch run, the training loss
(5-batch estimate) falls from 2.153 at step 0 to 0.083 at step 600, while the
5-batch training accuracy rises from 70.00% (epoch 1) to 100.00% (epoch 5)
(book p. 197). On the full sets, accuracy goes from ~46% before training to 95.67%
test after (book pp. 193, 200). The chapter summary repeats the point: classification
fine-tuning "uses the same cross entropy loss function as when pretraining the LLM"
(book p. 203).

**Why two output nodes, not one?** For a binary task a single output node would
work, but the book says that "would require modifying the loss function". It keeps one
output node per class so the standard multiclass cross-entropy applies unchanged
(sidebar "Output layer nodes", book p. 183). The single-node version would normally use
a sigmoid with [[binary-cross-entropy]] (our note, not the book's). With $K=2$ the two
are equivalent: $p_1 = \sigma(z_1 - z_0)$.

## 🔴 Theory (Expert)
- **Upper bound.** Measure the loss in bits ($\log_2$). On any misclassified example,
  some other class $k$ has $p_k \ge p_y$, and $p_k + p_y \le 1$, so $p_y \le \tfrac12$. That
  means $-\log_2 p_y \ge 1 = \ell_{0\text{-}1}$. On correct examples, $-\log_2 p_y \ge 0 = \ell_{0\text{-}1}$.
  So, for any number of classes $K$:
  $$\mathcal{L}_{0\text{-}1}(\theta) \le \frac{1}{\ln 2}\,\mathcal{L}_{\text{CE}}(\theta).$$
  Pushing the surrogate down pushes down a guaranteed ceiling on the error rate. (This
  derivation is ours, not the book's.) For the untrained model, $2.453/\ln 2 \approx 3.5$,
  which is a vacuous bound. After training, a small CE loss gives a tight one.
- **Convexity where possible.** Cross-entropy is convex in the logits $\mathbf{z}$
  (log-sum-exp is convex, and $-z_y$ is linear). If the features are fixed and only a
  linear head is trained (see [[layer-freezing]]), the problem is convex in the head
  weights: multinomial [[logistic-regression]]. The 0-1 loss isn't convex, and
  minimising it exactly is NP-hard in general. In deep networks the composition makes
  the whole objective non-convex anyway, but smoothness is what gradient methods need.
- **The same idea in other models.** The [[hinge-loss]] of support vector machines is
  another convex surrogate for 0-1 loss (MML §12.2.5). Logistic, exponential (AdaBoost)
  and squared losses are others. They all replace the step
  $\mathbb{1}[y f(x) < 0]$ with a continuous upper bound.
- **Classification-calibration (consistency).** A surrogate is useful in principle if
  minimising its *expected* value also minimises the expected 0-1 risk. Cross-entropy
  qualifies: its population minimiser is the true conditional probability
  $p(y \mid x)$, whose argmax is the Bayes-optimal classifier. This result is general
  theory, not from this book (e.g. Bartlett, Jordan & McAuliffe, 2006).
- **The two can disagree.** On a finite dataset, lower CE doesn't *guarantee* higher
  accuracy. CE can drop by making already-correct predictions more confident while a
  few borderline examples flip the wrong way. That's why the book tracks *both*
  losses and accuracies during training (book pp. 195–200). Cross-entropy is also
  the [[maximum-likelihood-estimation|negative log-likelihood]] of a categorical
  model, so minimising it is MLE. That's a meaningful objective on its own, not just
  a trick.
- **Empirical risk view.** Training minimises the empirical surrogate risk, and we
  measure the empirical 0-1 risk on held-out data. That gap between training on one
  quantity and reporting another is what [[empirical-risk-minimization]] with
  surrogate losses formalises.

## Connections
- Builds on: [[loss-function]], [[cross-entropy-loss]], [[softmax]], [[gradient]], [[partial-derivative]], [[classification-accuracy]]
- Leads to: [[training-loop]], [[classification-fine-tuning]]
- Related: [[empirical-risk-minimization]], [[hinge-loss]], [[logistic-regression]], [[maximum-likelihood-estimation]], [[log-probability]], [[binary-cross-entropy]], [[convex-optimization]], [[last-token-pooling]]
- Used by: [[gradient-descent]], [[backpropagation]], [[fine-tuning]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book pp. 193–194 ("classification accuracy is not a differentiable function … cross-entropy loss as a proxy"; `calc_loss_batch` on pp. 193–194; Listing 6.9 `calc_loss_loader`; initial losses 2.453 / 2.583 / 2.322)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5–6.6, book pp. 188, 192 (logits `[-3.5983, 3.9902]` used in the worked example)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book p. 183 (sidebar "Output layer nodes": 2 nodes + cross-entropy instead of 1 node + modified loss); §6.6, book p. 191 (Fig. 6.14, untrained predictions are wrong)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7, book pp. 195–200 (training log p. 197: loss 2.153 → 0.083 while accuracy rises; 95.67% test on p. 200); §6.8 summary, book p. 203
- `raw/mml-book.pdf` §12.2.5 (hinge loss as a convex surrogate for zero-one loss), via [[hinge-loss]]
- Derived (not in the book): the $\log_2$ upper bound, gradient formula, worked CE numbers, calibration remark.
