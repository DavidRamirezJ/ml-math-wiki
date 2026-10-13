---
title: Regularization
type: concept
tags: [machine-learning, optimization, generalization, deep-learning, llm]
difficulty: intermediate
sources: [mml-ch08-when-models-meet-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appD-training-loop-bells-and-whistles]
created: 2026-06-04
updated: 2026-10-13
---

# Regularization

> **One-liner:** Regularization adds a "keep it simple" penalty to training so the model fits the data without contorting itself into something that won't work on new data.

## 🟢 Intuition (Dummy)
Give a student a practice exam and unlimited time and they might *memorize* every answer — perfect
on practice, useless on the real test. Regularization is the teacher saying "your explanation has to
be simple, not just correct on these examples." It charges a fee for complicated, extreme solutions,
nudging the learner toward an explanation that's modest enough to also work on questions it has
never seen.

**In LLMs** there are two everyday versions of that teacher. **Weight decay** is the "keep your
numbers small" fee: every training step, each weight shrinks a tiny bit unless the data keeps pushing
it up. **[[dropout|Dropout]]** is like a team practice where, each round, a random half of the players
sit on the bench: nobody can rely on one star player, so the whole team learns to cover for each other.
The LLM book uses both, and it also shows the problem they fight: a small GPT trained for ten epochs on
one short story starts copying the story word for word (Raschka §5.2, pp. 150–151).

## 🟡 Mechanics (Practitioner)
Plain [[empirical-risk-minimization]] can drive training loss to zero by memorizing the data —
classic [[overfitting-and-underfitting|overfitting]]. Regularization **biases the search** by adding
a penalty on the parameters (MML §8.2.3). For least squares:
$$ \underbrace{\min_{\boldsymbol\theta}\ \tfrac1N\lVert \mathbf y - X\boldsymbol\theta\rVert^2}_{\text{fit the data}}
\;\;\longrightarrow\;\;
\min_{\boldsymbol\theta}\ \tfrac1N\lVert \mathbf y - X\boldsymbol\theta\rVert^2 \;+\; \lambda\,\lVert\boldsymbol\theta\rVert^2 . $$
- $\lVert\boldsymbol\theta\rVert^2$ is the **regularizer / penalty term**; it pulls $\boldsymbol\theta$ toward the origin (small weights).
- $\lambda\ge 0$ is the **regularization parameter**: it trades off data fit against parameter size.
  $\lambda=0$ recovers ordinary least squares; large $\lambda$ forces $\boldsymbol\theta\to\mathbf 0$.

$\lambda$ is a **hyperparameter** — not learned by the optimizer, but chosen by
[[cross-validation]].

**Worked intuition.** Overfit models tend to have **large** weights (a wiggly curve needs big
coefficients). Penalizing $\lVert\boldsymbol\theta\rVert^2$ makes the optimizer "pay" for those big
weights, so it keeps them small unless the data fit really demands otherwise.

**ML usage.** This $\ell_2$ penalty is **ridge regression** (a.k.a. weight decay in neural nets,
Tikhonov regularization classically). Swapping in $\lVert\boldsymbol\theta\rVert_1$ gives the
**lasso**, which drives weights exactly to zero (feature selection).

### In LLMs (Raschka, *Build a Large Language Model (From Scratch)*)
The LLM book never writes an explicit penalty term. Instead it uses two regularizers built into the
training code, plus some training-time habits.

**1. Weight decay via AdamW** (sidebar, §5.2, p. 148). The book: weight decay "aims to minimize model
complexity and prevent overfitting by penalizing larger weights", and AdamW "improves the weight decay
approach", which gives more effective regularization and better generalization — "hence AdamW is
frequently used to train LLMs". The update (standard form, not printed in the book):
$$\boldsymbol\theta_t = \boldsymbol\theta_{t-1} - \eta\left(\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\epsilon} + \lambda\,\boldsymbol\theta_{t-1}\right).$$
Symbol by symbol:
- $\boldsymbol\theta_t$ = all the model's weights after step $t$; $\boldsymbol\theta_{t-1}$ = before it.
- $\eta$ ("eta") = the [[learning-rate]]; $\lambda$ ("lambda") = the weight-decay strength.
- $\hat{\mathbf m}_t$, $\hat{\mathbf v}_t$ = Adam's running averages of the [[gradient]] and of the squared gradient; $\epsilon$ = a tiny number to avoid dividing by zero.
- The first term inside the brackets is the usual Adam step. The second term, $\lambda\boldsymbol\theta_{t-1}$, pulls every weight straight toward 0.

The book uses `weight_decay=0.1` everywhere:

| Where | Call | Page |
|---|---|---|
| Pretraining GPT-124M on *The Verdict* | `AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)` | §5.2, p. 149 |
| Resuming from a checkpoint | `AdamW(..., lr=5e-4, weight_decay=0.1)` | §5.4, p. 160 |
| Spam-classification fine-tuning | `AdamW(..., lr=5e-5, weight_decay=0.1)` | §6.7, p. 197 |
| Instruction fine-tuning | `AdamW(..., lr=0.00005, weight_decay=0.1)` | §7.6, p. 231 |
| Training loop with warmup/cosine/clipping | `AdamW(model.parameters(), weight_decay=0.1)` (lr set by hand each step) | App. D.1, p. 315 |

Worked number (ours): with $\eta = 4\times10^{-4}$ and $\lambda = 0.1$, the decay part multiplies each weight by
$1 - \eta\lambda = 1 - 0.00004 = 0.99996$ per step — tiny, but it adds up over thousands of steps. See [[weight-decay]], [[adamw-optimizer]].

**2. Dropout** (§3.5.2, p. 78; §4.1, p. 95; §4.5, p. 115). During training, randomly chosen hidden units
are ignored ("dropped out"), so the model can't rely too much on any specific set of units. It is used
**only during training**. Survivors are scaled up by $1/(1-p)$ (p. 79):
$$\tilde a = \frac{m}{1-p}\,a,\qquad m\sim\operatorname{Bernoulli}(1-p).$$
- $a$ = an activation or attention weight; $p$ = the dropout rate; $m$ = a random 0/1 "keep" switch that is 1 with probability $1-p$; $\tilde a$ = the value after dropout.

Where GPT uses it: on the [[attention-weights]] (§3.5, the book picks this placement because it is "the
more common variant in practice", p. 78), on the embeddings (`drop_emb`, p. 96), and after each sub-layer of the
[[transformer-block]] before the shortcut add (`drop_shortcut`, p. 115):
$$\mathbf h = \mathbf x + \operatorname{Dropout}\big(\operatorname{MHA}(\operatorname{LN}_1(\mathbf x))\big),\qquad
\mathbf y = \mathbf h + \operatorname{Dropout}\big(\operatorname{FFN}(\operatorname{LN}_2(\mathbf h))\big).$$
Rates: 0.5 in the demo; "a lower dropout rate, such as 0.1 or 0.2" for real GPT training (p. 78);
`drop_rate: 0.1` ("a 10% drop of random hidden units") in `GPT_CONFIG_124M` (p. 95) and in Appendix D (p. 313).
`model.eval()` switches dropout off for evaluation and generation; `model.train()` turns it back on
(§5.2 p. 148, §5.3 p. 151, §5.4 p. 160). See [[train-vs-eval-mode]].

```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
print(dropout(torch.ones(6, 6)))   # about half the entries become 0, survivors become 2.
```

**3. Seeing overfitting, and the non-penalty fixes.**
- §5.2 (pp. 149–151): over 10 epochs the training loss falls from 9.781 to 0.391 while the validation loss stays high (6.452 after epoch 10); the two curves diverge past epoch 2, and the model reproduces *The Verdict* verbatim. Cause: a "very, very small training dataset" and multiple epochs. The usual practice is a much larger dataset for only **one epoch** (e.g. 60,000 Project Gutenberg books, Appendix B) — more data is the strongest regularizer. See [[training-and-validation-loss]].
- §6.7 (p. 199): "five epochs is usually a good starting point"; if the model overfits after the first few epochs, **reduce** the number of epochs (this is the idea of early stopping; the book does not use the name).
- §6.7 (p. 200): accuracies 97.21% train / 97.32% validation / 95.67% test. Validation accuracy is usually a bit higher than test accuracy because hyperparameters are tuned on the validation set; to shrink that validation–test gap, the book suggests "increasing the dropout rate (`drop_rate`) or the `weight_decay` parameter" — both are regularization methods.
- §7.6 (p. 232): two epochs of instruction fine-tuning; a third "is not essential and may even be counterproductive" because it could increase overfitting.

> ⚠️ Contradiction: the MML section above calls the $\ell_2$ penalty "a.k.a. weight decay in neural nets". The LLM book says AdamW "improves the weight decay approach" (p. 148), which only makes sense because, for adaptive optimizers like Adam, **weight decay and an $\ell_2$ penalty are not the same thing**. They coincide for plain SGD (see Theory). So read MML's "a.k.a." as "for SGD"; for AdamW, the decay is *decoupled* from the gradient. (The "decoupled" explanation is standard background from Loshchilov & Hutter; the LLM book does not spell it out.)

## 🔴 Theory (Expert)
- **Regularizer $\equiv$ prior.** Adding $\lambda\lVert\boldsymbol\theta\rVert^2$ is exactly
  [[maximum-a-posteriori-estimation|MAP estimation]] with a zero-mean Gaussian prior
  $p(\boldsymbol\theta)=\mathcal N(\mathbf 0,\Sigma)$: $-\log p(\boldsymbol\theta)$ contributes a
  quadratic term in $\boldsymbol\theta$. The $\ell_1$ lasso corresponds to a Laplace prior. The
  chapter makes this the **regularization $\leftrightarrow$ prior** half of the ERM/MLE parallel
  (§8.2.3, §8.3.2).
- This is **Tikhonov regularization**, born of ill-posed inverse problems (Neumaier 1998); the
  constrained twin is **Ivanov regularization**. It is intimately tied to the bias–variance
  trade-off: more regularization raises bias, lowers variance.
- In [[support-vector-machine]]s the regularizer is equivalent to a **large margin** (Ch. 12) — the
  geometry of generalization made explicit.

- **$\ell_2$ penalty = weight decay, but only for SGD.** Gradient descent on $L(\boldsymbol\theta)+\tfrac{\lambda}{2}\lVert\boldsymbol\theta\rVert^2$ gives
  $\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta(\nabla L + \lambda\boldsymbol\theta) = (1-\eta\lambda)\boldsymbol\theta - \eta\nabla L$: a multiplicative shrink, i.e. weight decay. In Adam, the penalty gradient $\lambda\boldsymbol\theta$ would be divided by $\sqrt{\hat{\mathbf v}_t}$ like everything else, so weights with large past gradients get *less* decay. AdamW applies $-\eta\lambda\boldsymbol\theta$ outside the adaptive rescaling, restoring a uniform shrink (Loshchilov & Hutter, "Decoupled Weight Decay Regularization" — background, not cited by the book). This decoupled form no longer corresponds to minimizing a fixed penalized objective, so the clean "regularizer = Gaussian prior" link above holds only approximately.
- **Inverted dropout is unbiased.** With $m\sim\operatorname{Bernoulli}(1-p)$, $\mathbb E[\tilde a] = \frac{1-p}{1-p}a = a$. That is the book's statement that the $1/(1-p)$ scaling keeps "the average influence of the attention mechanism" the same in training and inference (pp. 79–80); see [[expected-value-and-mean]]. Dropout can also be read as training an exponentially large ensemble of thinned networks with shared weights and averaging them at test time (Srivastava et al. 2014, cited in Appendix B p. 292; the ensemble reading is from that paper, not the book).
- **Noise as regularization.** Dropout injects multiplicative noise; for a linear model with squared loss its expected objective equals the clean loss plus a data-dependent $\ell_2$-type penalty (a classical result, outside the book). The book's Appendix B also describes GELU as combining nonlinearity with "stochastic regularization" ideas (Hendrycks & Gimpel 2016, p. 293).
- **Large-scale LLMs regularize mostly by data.** One pass over a huge corpus means each example is seen about once, so classic overfitting is rare; the book's small-data runs (a single short story, *The Verdict*) show the opposite regime (§5.2, p. 151).

## Connections
- Builds on: [[empirical-risk-minimization]], [[loss-function]], [[gradient-descent]], [[norm]]
- Leads to: [[overfitting-and-underfitting]], [[maximum-a-posteriori-estimation]], [[weight-decay]], [[dropout]]
- Used by: [[linear-regression]] (ridge), [[regularized-least-squares]], [[support-vector-machine]], [[adamw-optimizer]], [[transformer-block]], [[large-language-model]] training ([[pretraining]], [[classification-fine-tuning]], [[instruction-fine-tuning]])
- Related: [[overfitting]], [[training-and-validation-loss]], [[train-vs-eval-mode]], [[cross-validation]], [[epoch]], [[hyperparameter]]
- People: [[nitish-srivastava]], [[geoffrey-hinton]], [[sebastian-raschka]]

## Sources
- `raw/mml-book.pdf` §8.2.3 (Eqs. 8.11–8.12, Example 8.3), printed pp. 262–263 (PDF pp. 268–269); §8.2.5 (Tikhonov/Ivanov), p. 265.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5.2 (dropout definition, rates, $1/(1-p)$ scaling), book pp. 78–80; §4.1 (`drop_rate` 0.1), book p. 95; §4.5 (dropout in the transformer block), book p. 115.
- Same PDF §5.2 (AdamW / weight-decay sidebar; overfitting on *The Verdict*), book pp. 148–151; §5.3 (`model.eval()` disables dropout), book p. 151; §5.4 (dropout and eval mode; AdamW on resume), book p. 160.
- Same PDF §6.7 (choosing epochs; raising `drop_rate`/`weight_decay` to shrink the gap), book pp. 197–200; §7.6 (more epochs may increase overfitting), book pp. 231–232; Appendix D.1 (AdamW `weight_decay=0.1`, `drop_rate` 0.1), book pp. 313–315.
- Same PDF Appendix B (Srivastava et al. 2014 on dropout; Hendrycks & Gimpel 2016 on GELU), book pp. 292–293.
