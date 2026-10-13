---
title: Cross-Entropy Loss
type: concept
tags: [deep-learning, llm, loss-function, probability, information-theory]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Cross-Entropy Loss

> **One-liner:** Cross-entropy loss is one number that says how surprised the model was by the right answer. You take the probability it gave to the correct token, take its log, and flip the sign.

## 🟢 Intuition (Dummy)
Picture a quiz show. Before each answer comes out, the model has to bet its chips
across all 50,257 possible next words. When the real answer shows up, we only
look at how many chips the model put on it.

- Lots of chips on the right word means a small penalty.
- Almost no chips on it means a big penalty.
- Zero chips on it means an infinite penalty, so the model learns never to be
  that sure of the wrong thing.

Cross-entropy is that penalty, averaged over every word in the text. Training an
LLM really just means "move the chips toward the right words", over and over,
until the average penalty is small. The same penalty trains the GPT model in
pretraining (predict the next word), in spam classification (predict "spam" or
"not spam"), and in instruction tuning (predict the response). Only *what counts
as the right answer* changes.

## 🟡 Mechanics (Practitioner)

### The six-step recipe (Raschka Fig. 5.7, book p. 137)
The book builds the loss by hand first, using two 3-token examples
(`"every effort moves" → " effort moves you"` and `"I really like" → " really like chocolate"`):

| Step | What happens | Book's numbers |
|---|---|---|
| 1 | Model outputs **[[logits]]** | `[[[0.1113, -0.1057, -0.3666, ...]]]` |
| 2 | **[[softmax]]** turns them into probabilities | `[[[1.8849e-05, 1.5172e-05, ...]]]` |
| 3 | Pick the probability of each **target** token | `[7.4541e-05, 3.1061e-05, 1.1563e-05, ...]` |
| 4 | Take the natural **[[log-probability\|log]]** | `[-9.5042, -10.3796, -11.3677, -11.4798, -9.7764, -12.2561]` |
| 5 | Average them | $-10.7940$ |
| 6 | Flip the sign | $\mathbf{10.7940}$ ← the loss |

Check of step 5: $(-9.5042-10.3796-11.3677-11.4798-9.7764-12.2561)/6=-64.7638/6=-10.7940$.

### The formula
$$\mathcal{L}(\theta)=-\frac{1}{N}\sum_{n=1}^{N}\log p_\theta\big(y_n \mid \text{context}_n\big)$$

Reading it symbol by symbol:
- $\mathcal{L}$ = "the loss", one number for the whole batch.
- $\theta$ = all the model's trainable weights ([[model-parameters]]).
- $N$ = how many predictions we are scoring (here $2 \text{ texts} \times 3 \text{ tokens} = 6$).
- $\sum_{n=1}^{N}$ = "add up, for $n = 1, 2, \dots, N$".
- $y_n$ = the correct (target) token ID at position $n$.
- $\text{context}_n$ = the tokens the model saw before position $n$.
- $p_\theta(y_n \mid \text{context}_n)$ = "the probability the model with weights $\theta$ gives to the correct token, **given** the context". The bar $\mid$ reads "given".
- $\log$ = natural logarithm (base $e$).
- The minus sign in front turns "maximize the log-probability" into "minimize the loss".

Where does $p_\theta$ come from? From the softmax of the logit vector $\mathbf{z}$:
$$p_\theta(y\mid\text{context})=\operatorname{softmax}(\mathbf{z})_y=\frac{e^{z_y}}{\sum_{j=1}^{V}e^{z_j}}$$
- $\mathbf{z}$ = the logit vector for this position (one raw score per vocabulary token).
- $z_y$ = the score of the correct token $y$.
- $V$ = vocabulary size (50,257 for GPT-2).
- The fraction = "my score, exponentiated, divided by everybody's exponentiated scores", so all probabilities are positive and add to 1.

Plug that in and you get the form PyTorch actually computes:
$$\ell(\mathbf z,y)=-z_y+\log\sum_{j=1}^{V}e^{z_j}$$
("minus the right answer's score, plus the log of the sum of all exponentiated scores").

### In PyTorch: one call does all six steps (book pp. 138–139)
```python
logits_flat  = logits.flatten(0, 1)   # [2, 3, 50257] -> [6, 50257]
targets_flat = targets.flatten()      # [2, 3]        -> [6]
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)                           # tensor(10.7940)
```
- `cross_entropy` wants **raw logits**, not probabilities. It does the softmax
  inside, for "numerical efficiency and stability" (book pp. 269, 276; the logits are "unscaled model outputs", p. 139). So **never**
  apply softmax yourself before calling it.
- The flatten merges the batch and token dimensions, so every token position
  becomes one "row" to score.

The book wraps this in `calc_loss_batch` (one batch) and `calc_loss_loader`
(average over `num_batches` batches) (book pp. 144–145).

### What the numbers mean
- **Random guessing** over $V$ tokens gives every token $1/V$, so the loss is
  $\log V=\ln 50257\approx 10.82$. The untrained GPT-2 124M model gives
  **10.988** (train) and **10.981** (validation) on "The Verdict" (book p. 145),
  slightly *worse* than uniform.
- After 10 epochs of pretraining: train **0.391**, validation **6.452** (book p. 149).
  The big gap means [[overfitting]].
- OpenAI's pretrained GPT-2 124M on the same text: train **3.755**, validation
  **3.560** (Exercise 5.5 solution, appendix C, book p. 305). The validation loss being slightly lower is attributed to noise on a small dataset; "The Verdict" was not in GPT-2's pretraining data.
- $e^{\mathcal L}$ is the **[[perplexity]]**: $e^{10.7940}\approx 48{,}726$.

### Same loss, three jobs
| Use | What the targets are | Logit shape fed to the loss | Book |
|---|---|---|---|
| Pretraining | next token at **every** position | $[B\cdot T,\ 50257]$ | §5.1, p. 144 |
| [[classification-fine-tuning\|Classification]] (spam) | class label 0/1, **last token only** | $[B,\ 2]$ | §6.6, p. 193 |
| [[instruction-fine-tuning\|Instruction tuning]] | next token, padding masked with $-100$ | $[B\cdot T,\ 50257]$ | §7.3, pp. 221–222 |

For classification the only code change is `model(input_batch)[:, -1, :]`,
which keeps the [[last-token-pooling|last token]]'s 2 logits (book p. 193).
Accuracy itself is not differentiable, so cross-entropy acts as a
[[surrogate-loss|differentiable stand-in]] for it (book p. 193). The untrained
spam head starts at losses 2.453 / 2.583 / 2.322 (train / val / test, book p. 194),
above the 2-class uniform value $\ln 2\approx 0.693$, because its random logits are
confidently wrong.

### `ignore_index=-100`: leaving positions out (book pp. 221–222)
PyTorch's `cross_entropy` skips any target equal to $-100$. The book's toy demo:
```python
logits_2  = torch.tensor([[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]])
targets_2 = torch.tensor([0, 1, 1])     # loss 0.7936
targets_3 = torch.tensor([0, 1, -100])  # loss 1.1269, same as using only 2 rows
```
Per-row losses are $2.1269$, $0.1269$, $0.1269$. With the third row masked, the
mean is $(2.1269+0.1269)/2=1.1269$. The masked formula is
$$\mathcal{L}=-\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\log\operatorname{softmax}(\mathbf z_i)_{y_i},\qquad \mathcal I=\{i: y_i\neq -100\}$$
- $\mathcal I$ = the set of positions we keep (target is not $-100$).
- $|\mathcal I|$ = how many positions we keep. We average only over those.

The book uses this to ignore extra padding tokens, keeping one `<|endoftext|>`
so the model learns to stop. See [[ignore-index-loss-masking]] and [[instruction-masking]].

## 🔴 Theory (Expert)
- **Information-theory definition.** For a true distribution $q$ and model
  distribution $p$ over the same $V$ outcomes,
  $H(q,p)=-\sum_{i=1}^{V}q_i\log p_i$. When $q$ is **one-hot** on the correct token
  $y$ (all mass on $y$), every term but one vanishes and $H(q,p)=-\log p_y$.
  The book states that cross-entropy "measures the difference between two
  probability distributions" and that the "negative average log probability" is
  what is used in practice; the two names are often used interchangeably
  (book p. 138).
- **Cross-entropy = entropy + KL.** $H(q,p)=H(q)+D_{\mathrm{KL}}(q\,\|\,p)$. With a
  fixed data distribution $q$, $H(q)$ is a constant, so minimizing cross-entropy
  is the same as minimizing the KL divergence from data to model.
- **It is maximum likelihood.** For a corpus $x_1,\dots,x_T$ the autoregressive
  likelihood is $\prod_t p_\theta(x_{t+1}\mid x_{\le t})$. Its log is a sum, and
  dividing by $-T$ gives exactly $\mathcal L$. So minimizing cross-entropy is
  [[maximum-likelihood-estimation]] of a categorical next-token model, and the
  averaged training loss is an [[empirical-risk-minimization|empirical risk]].
  The book says the training goal is "to maximize the likelihood of the correct
  token" (book p. 136).
- **Gradient with respect to the logits.** With $\mathbf p=\operatorname{softmax}(\mathbf z)$
  and one-hot $\mathbf e_y$, $\nabla_{\mathbf z}\,\ell=\mathbf p-\mathbf e_y$. The
  push is "lower every token by its probability, raise the right one by 1". It is
  bounded in $[-1,1]$ per entry, which is one reason the softmax + log pair
  trains well with [[backpropagation]].
- **Why fuse softmax and log.** Computing $\log\operatorname{softmax}$ in two steps
  can underflow (probabilities like $10^{-5}$ and smaller) or overflow ($e^{z}$
  for large $z$). The fused log-sum-exp form
  $\log\sum_j e^{z_j}=m+\log\sum_j e^{z_j-m}$ with $m=\max_j z_j$ is stable. This is
  the "numerical efficiency and stability" reason the book gives (book p. 269).
  See [[numerical-stability]].
- **Lower bound and calibration.** $\mathcal L\ge 0$, with equality only if the
  model puts probability 1 on every target. On real text the achievable minimum
  is the data's own entropy rate, not 0. A training loss near 0 (0.035 in
  appendix D, book p. 321) on a tiny corpus means memorization, not perfect
  language modeling.
- **Binary case.** With 2 classes, softmax cross-entropy equals
  [[binary-cross-entropy]] on the logit difference $z_1-z_0$ passed through the
  [[sigmoid-function]].

## Connections
- Builds on: [[softmax]], [[logits]], [[log-probability]], [[probability-distribution]], [[loss-function]], [[one-hot-encoding]]
- Leads to: [[perplexity]], [[training-and-validation-loss]], [[llm-training-loop]], [[ignore-index-loss-masking]]
- Used by: [[pretraining]], [[next-token-prediction]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[logistic-regression]]
- Related: [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[binary-cross-entropy]], [[surrogate-loss]], [[numerical-stability]], [[backpropagation]], [[gradient-descent]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.2, book pp. 132–140 (six-step recipe Fig. 5.7, p. 137; cross-entropy sidebar, p. 138; `cross_entropy` call, p. 139)
- Same, §5.1.3, book pp. 144–145 (`calc_loss_batch`, `calc_loss_loader`, untrained losses 10.988 / 10.981)
- Same, §5.2, book pp. 146–151 (training losses 9.781 → 0.391, val 6.452); §5.5 summary, book pp. 167–168
- Same, §6.6, book pp. 193–194 (last-token classification loss, proxy for accuracy); §6.7, book pp. 195–197; §6.8 summary, book p. 203
- Same, §7.3, book pp. 221–223 (`ignore_index=-100` demo, instruction masking); §7.6, book p. 230 (initial losses 3.826 / 3.762)
- Same, appendix A §A.5, book p. 269; §A.7, book pp. 275–276; §A.9, book p. 281 (logits go straight into `F.cross_entropy`)
- Same, appendix D §D.4, book pp. 319–321 (losses 10.934 → 0.035, val 6.938)
- Same, appendix C, Exercise 5.5, book p. 305 (pretrained GPT-2: 3.755 / 3.560)
