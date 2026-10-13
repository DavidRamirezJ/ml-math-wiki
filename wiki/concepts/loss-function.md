---
title: Loss Function
type: concept
tags: [machine-learning, optimization, supervised-learning]
difficulty: beginner
sources: [mml-ch08-when-models-meet-data, llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-06-04
updated: 2026-10-13
---

# Loss Function

> **One-liner:** A loss function is the rule that turns "how wrong was this one prediction?" into a single non-negative number you can try to make small.

## 🟢 Intuition (Dummy)
You guessed a friend's height as 170 cm; they're actually 180 cm. How bad is that guess? A loss
function answers with one number — say "10" if you score by the raw gap, or "100" if you score by
the gap *squared* (which punishes big misses much harder). Zero loss means a perfect guess. The
choice of how you score the gap quietly shapes what the learner cares about: square it and a few
huge errors dominate; use the absolute gap and every error counts proportionally.

**In LLMs.** A language model does not guess a height, it guesses the *next word*. It hands out
"confidence points" (probabilities) to all 50,257 possible tokens. The loss asks one question: *how
many points did you give to the word that actually came next?* Lots of points → small loss. Almost
none → big loss. Raschka calls the loss "a progress and success indicator" (book p. 132): it is the
one number that tells training which way to turn every knob ([[model-parameters]]). Before
training, the model spreads its points almost evenly, about $1/50{,}257\approx0.00002$ per token,
so the loss starts high, near 11 (book pp. 135, 145).

## 🟡 Mechanics (Practitioner)
A **loss function** $\ell(y_n,\hat y_n)\ge 0$ takes the ground-truth label $y_n$ and the prediction
$\hat y_n$ and returns the cost of that single prediction (MML §8.2.2). It is the inner ingredient of
the empirical risk averaged over the data ([[empirical-risk-minimization]], Eq. 8.6):
$$ R_{\text{emp}}=\frac1N\sum_{n=1}^N \ell(y_n,\hat y_n). $$

**The squared loss (Example 8.2):**
$$ \ell(y_n,\hat y_n)=(y_n-\hat y_n)^2 . $$
With a linear predictor $\hat y_n=\boldsymbol\theta^\top\mathbf x_n$ it produces the
**least-squares problem** $\min_{\boldsymbol\theta}\frac1N\lVert\mathbf y-X\boldsymbol\theta\rVert^2$.

**Worked example.** Labels $\mathbf y=(2,4,6)$, predictions $\hat{\mathbf y}=(1,4,8)$. Per-example
squared losses are $(2-1)^2=1$, $(4-4)^2=0$, $(6-8)^2=4$. Empirical risk $=\frac{1+0+4}{3}=\tfrac53\approx1.67$.

**ML usage.** Common losses by task:
| Task | Typical loss | Note |
|---|---|---|
| Regression | squared $(y-\hat y)^2$, absolute $\lvert y-\hat y\rvert$ | squared ⇒ least squares / [[maximum-likelihood-estimation|Gaussian MLE]] |
| Classification | cross-entropy, hinge | hinge ⇒ [[support-vector-machine]] margin |
| Counts | (negative log) Poisson likelihood | a generalized linear model |

### In LLMs: the cross-entropy loss, step by step (Raschka §5.1.2)
The book builds the next-token loss by hand in six steps (Figure 5.7, book p. 137). Two training
examples, `"every effort moves"` → `" effort moves you"` and `"I really like"` → `" really like
chocolate"`; the targets are the inputs shifted one token to the right (book p. 133):
```python
inputs  = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345],  [1107, 588, 11311]])
```
1. **Logits.** `logits = model(inputs)` has shape `[2, 3, 50257]` (batch, tokens, vocabulary).
2. **Probabilities.** `probas = torch.softmax(logits, dim=-1)`, where
   $\text{softmax}(\mathbf z)_i = e^{z_i}/\sum_{j=1}^{V}e^{z_j}$ ($\mathbf z$ = the logit vector at one
   position, $z_i$ = the score of token $i$, $V$ = 50,257).
3. **Target probabilities.** Pick the probability of the *correct* token at each position:
   `[7.4541e-05, 3.1061e-05, 1.1563e-05]` and `[1.0337e-05, 5.6776e-05, 4.7559e-06]` (book p. 136).
4. **Log.** `torch.log(...)` → `[-9.5042, -10.3796, -11.3677, -11.4798, -9.7764, -12.2561]`.
5. **Average.** $(-64.7638)/6 = -10.7940$. The goal is to push this toward $0=\log 1$ (book p. 138).
6. **Negate.** $10.7940$. Deep-learning convention is to *minimize*, so we flip the sign. This
   "negative average log probability" is the **[[cross-entropy-loss]]** (book p. 138).

As one formula:
$$\mathcal L(\theta) = -\frac{1}{N}\sum_{n=1}^{N}\log p_\theta\big(y_n \mid \text{context}_n\big).$$
Symbol by symbol: $\mathcal L$ = the loss (one number); $\theta$ = the [[model-parameters]];
$N$ = number of predicted positions (here $2\times3=6$); $\sum_{n=1}^{N}$ = "add up over every
position"; $y_n$ = the true next token at position $n$; $p_\theta(y_n\mid\text{context}_n)$ = the
softmax probability the model gives that token; $\log$ = natural log (negative for numbers below 1);
the leading minus makes the result positive.

PyTorch does all six steps in one call, after flattening batch and token dimensions (book p. 139):
```python
logits_flat  = logits.flatten(0, 1)    # [6, 50257]
targets_flat = targets.flatten()       # [6]
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)   # tensor(10.7940)
```
Note that `cross_entropy` takes raw **[[logits]]**, not probabilities: it applies (log-)softmax
inside. That is why PyTorch models return the last layer's outputs with no final softmax; combining
softmax and negative log-likelihood in one function is more efficient and numerically stable
(book pp. 269, 276).

**[[perplexity|Perplexity]]** is the loss made human-readable (book pp. 139–140):
$$\text{PPL} = e^{\mathcal L},\qquad e^{10.7940}\approx 48{,}725.8 .$$
Read it as "the model is unsure about which among 48,725 tokens in the vocabulary to generate as the
next token" (book p. 140; out of 50,257). Lower is better.

**One loss, three tasks.** The book reuses the same cross-entropy everywhere:
| Where | Classes per prediction | Which positions count | Book numbers |
|---|---|---|---|
| Pretraining (Ch. 5) | 50,257 tokens | all $B\times T$ positions | untrained 10.988 train / 10.981 val (p. 145); after 10 epochs 0.391 / 6.452 (p. 149) |
| Spam classification (Ch. 6) | 2 (spam / not spam) | **last token only**, `model(x)[:, -1, :]` | initial 2.453 / 2.583 / 2.322 train/val/test (p. 194) |
| Instruction tuning (Ch. 7) | 50,257 tokens | all except targets set to `-100` | initial 3.826 / 3.762 (p. 230); 0.300 / 0.657 after 2 epochs (p. 232) |

**Ignoring positions with `-100`** (§7.3, book pp. 221–222). `cross_entropy` has
`ignore_index=-100` by default, so padding targets marked `-100` drop out of the average:
$$\mathcal L = -\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\log\frac{e^{z_{i,y_i}}}{\sum_{k}e^{z_{i,k}}},\qquad
\mathcal I=\{i: y_i\neq-100\}.$$
Symbol by symbol: $\mathcal I$ = the set of positions *not* ignored; $|\mathcal I|$ = how many there
are; $z_{i,k}$ = logit of class $k$ at position $i$; the fraction is the softmax probability of the
correct class. Toy check from the book: logits `[[-1.0, 1.0], [-0.5, 1.5]]`, targets `[0, 1]` →
per-token losses $\log(1+e^{2})=2.1269$ and $\log(1+e^{-2})=0.1269$ (our arithmetic), mean **1.1269** (book p. 221). Adding a
third position `[-0.5, 1.5]` with target 1 gives **0.7936**; with target `-100` it gives
**1.1269** again (`loss_1 == loss_3: tensor(True)`).

**Other losses in the book.**
- **[[mean-squared-error|Mean squared error]]** (§4.4): `nn.MSELoss()` with target 0 is used only to
  generate gradients and show how [[shortcut-connections]] stop them from vanishing (book pp. 111–112):
  $\mathcal L=\frac1n\sum_{i=1}^{n}(\hat y_i-y_i)^2$ ($\hat y_i$ = output, $y_i$ = target, $n$ = count).
- **[[binary-cross-entropy|Binary cross-entropy]]** (§A.3–A.4): for [[logistic-regression]] with
  output $a=\sigma(z)\in(0,1)$ and label $y\in\{0,1\}$,
  $$L(a,y) = -\big[\,y\ln a + (1-y)\ln(1-a)\,\big].$$
  Book example: $x_1=1.1$, $w_1=2.2$, $b=0$, $y=1$ gives $z=2.42$, $a=\sigma(2.42)\approx0.9183$,
  loss $=-\ln 0.9183\approx0.0852$ (values from Listing A.2, arithmetic ours; book p. 262).

**Loss in the training loop.** Every loop in the book has the same five lines (§5.2, §A.7):
```python
optimizer.zero_grad()                     # reset old gradients
loss = calc_loss_batch(x, y, model, device)
loss.backward()                           # gradients of the loss w.r.t. every parameter
optimizer.step()                          # move parameters to lower the loss
```
**Batching and loss spikes.** The data loader uses `drop_last=True`, which drops the last batch
if it is shorter than the batch size, "to prevent loss spikes during training" (§2.6, book p. 38).

In the Appendix A toy network the loss falls 0.75 → 0.65 → 0.44 → 0.13 → 0.03 → 0.00 over three
epochs, "a sign that the model converged on the training set" (book p. 275).
With [[low-rank-adaptation|LoRA]] (Appendix E) the spam classifier's training loss drops from 3.820
to 0.019 in five epochs, while validation loss bottoms out near 0.045 and drifts back up to 0.252
(book pp. 334–335): the same cross-entropy, only fewer trainable parameters.

Ideally the loss equals the task's **performance measure**; in practice they differ for ease of
optimization (a smooth surrogate replaces a non-differentiable metric like accuracy).

## 🔴 Theory (Expert)
- Every loss has a **probabilistic alter ego**: minimizing average loss is equivalent to maximizing
  a likelihood, with $\ell(y,\hat y)=-\log p(y\mid\hat y)$ up to constants. Squared loss
  $\Leftrightarrow$ Gaussian likelihood (see [[maximum-likelihood-estimation]], Example 8.5); hinge
  has no exact likelihood (it bounds 0–1 loss). This is the **loss $\leftrightarrow$ negative
  log-likelihood** correspondence the chapter builds the whole §8.2–§8.3 parallel on.
- The **0–1 loss** (1 if wrong, 0 if right) is the "honest" classification objective but is
  non-convex and non-differentiable; convex **surrogate losses** (hinge, logistic, exponential) are
  minimized instead, with consistency guarantees under conditions (classification calibration).
- **LLM cross-entropy = categorical negative log-likelihood.** With a one-hot target $q$, the general
  cross-entropy $H(q,p)=-\sum_i q_i\log p_i$ collapses to $-\log p_{y}$, so the LLM loss is exactly
  the NLL of a categorical model, and minimizing it is [[maximum-likelihood-estimation]] (book p. 136:
  "maximize the likelihood of the correct token"). This is the loss ↔ NLL correspondence above, with
  a categorical instead of a Gaussian likelihood.
- **Reference points.** A uniform guess over $V$ tokens gives $\mathcal L=\ln V$; for $V=50{,}257$,
  $\ln V\approx10.825$ and PPL $=V$. The book's untrained losses (10.988, 10.981) sit slightly
  *above* this, so random initialization is a bit worse than uniform. A 2-class uniform guess gives
  $\ln 2\approx0.693$ (these reference values are our derivation, not printed in the book).
- **Why logs.** Log probabilities are "more manageable in mathematical optimization" (book p. 138):
  they turn products of many tiny probabilities ($\sim10^{-5}$) into sums and avoid numerical
  underflow. Fusing log and softmax (log-sum-exp) is what makes `cross_entropy` stable on raw logits.
- **Surrogate for a non-differentiable metric.** Accuracy is "not a differentiable function", so the
  book uses cross-entropy "as a proxy" to maximize it (book p. 193); see [[surrogate-loss]]. The A.4
  framing is the same: the loss "serves as a proxy for measuring the model's performance"
  (book pp. 263–264). And even a good loss is not the end goal: in Ch. 7 "the most crucial aspect"
  is response quality, judged separately (book p. 233).
- **Loss geometry and the learning rate.** Decaying the [[learning-rate]] lowers the risk of
  "overshooting the loss minima" late in training (Appendix D, book p. 316). [[low-rank-adaptation|LoRA]]
  still minimizes the same loss; it only restricts the update $\Delta W$ to low rank (book p. 322).
- **Instruction masking is a loss design choice.** Masking instruction tokens with `-100` makes the
  loss count only response tokens (book p. 222). Shi et al. (2024) found *not* masking helps, and the
  book leaves masking as an exercise (book p. 223).
- Choosing the loss is choosing the noise model: squared loss assumes symmetric Gaussian noise, so
  it is sensitive to outliers; heavier-tailed assumptions (Laplace ⇒ absolute loss) are more robust.

## Connections
- Builds on: [[vector]]
- Leads to: [[empirical-risk-minimization]], [[regularization]]
- Used by: [[linear-regression]], [[support-vector-machine]], [[logistic-regression]], [[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[llm-training-loop]]
- Variants (LLM book): [[cross-entropy-loss]], [[binary-cross-entropy]], [[mean-squared-error]], [[perplexity]], [[ignore-index-loss-masking]], [[surrogate-loss]]
- Related (LLM book): [[softmax]], [[logits]], [[log-probability]], [[backpropagation]], [[gradient-descent]], [[training-and-validation-loss]]
- Parallels: likelihood in [[maximum-likelihood-estimation]]
- People: [[carl-friedrich-gauss]] (least squares), [[zhengyan-shi]] (loss over instructions)

## Sources
- `raw/mml-book.pdf` §8.2.2 (Eq. 8.6) & Example 8.2 (Eqs. 8.7–8.9), printed pp. 260–261 (PDF pp. 266–267).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1 (error minimization), book p. 3 (PDF p. 25); §1.3, book pp. 6–7 (PDF pp. 28–29).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6 (`drop_last` prevents loss spikes), book p. 38 (PDF p. 60).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1–4.2 (parameters minimize a loss), book pp. 93, 99 (PDF pp. 115, 121); §4.4 (MSE, `loss.backward()`), book pp. 111–112 (PDF pp. 133–134).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.2–5.1.3 (six-step cross-entropy, perplexity, Listing 5.2), book pp. 132–145 (PDF pp. 154–167); §5.2 (training loop and curves), book pp. 146–151 (PDF pp. 168–173).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6 (classification loss, proxy for accuracy), book pp. 193–194 (PDF pp. 215–216); §6.8, book p. 203 (PDF p. 225).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3 (`ignore_index=-100`, instruction masking), book pp. 218–223 (PDF pp. 240–245); §7.6 (instruction-tuning losses), book pp. 229–232 (PDF pp. 251–254); §7.7, book p. 233 (PDF p. 255).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.1 (loss functions in PyTorch), book p. 252 (PDF p. 274); §A.3–A.4 (binary cross-entropy, loss as proxy), book pp. 261–264 (PDF pp. 283–286); §A.5 (logits, fused softmax+NLL), book p. 269 (PDF p. 291); §A.7 (training loop), book pp. 274–276 (PDF pp. 296–298).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §D.2 (overshooting loss minima), book p. 316 (PDF p. 338); §E.1 (LoRA minimizes the loss), book p. 322 (PDF p. 344); §E.4 (LoRA loss log), book pp. 334–335 (PDF pp. 356–357).
