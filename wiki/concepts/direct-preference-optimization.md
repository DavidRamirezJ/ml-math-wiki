---
title: Direct Preference Optimization (DPO)
type: concept
tags: [llm, fine-tuning, alignment, optimization, probability]
difficulty: advanced
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Direct Preference Optimization (DPO)

> **One-liner:** DPO teaches a model to prefer the "better" of two answers by nudging up the better answer's probability and nudging down the worse one's, while staying close to the original model, with no separate reward model needed.

## 🟢 Intuition (Dummy)
Say you have two answers to the same question, and people liked answer A more than answer B. The simplest wish is: "model, make A a bit more likely and B a bit less likely." DPO is that wish, written as a clean loss you can train with normal [[gradient-descent]].

There is one safety rope. You keep a frozen copy of the model you started from (the *reference*). DPO does not ask "is A likely?" It asks "has A become *more* likely than it was in the reference, compared to how B changed?" That relative view stops the model from wandering off into strange text just to win comparisons.

The older way ([[preference-fine-tuning|RLHF]]) needed three moving parts: a reward model, a reinforcement-learning loop, and a lot of tuning. DPO gets to the same goal with one loss and one training loop, much like the [[llm-training-loop]] the book already built. That is why the book's GitHub repository uses DPO for its preference-tuning bonus material (folder `04_preference-tuning-with-dpo`, book p. 247).

## 🟡 Mechanics (Practitioner)

> **Book coverage:** the book's text only names DPO, through the repository folder `04_preference-tuning-with-dpo` (https://mng.bz/dZwD) in §7.9.1, book p. 247. The formulas, example and code below are **standard background** (Rafailov et al. 2023), not from the book's pages.

### The data
Each example is a triple $(x, y_w, y_l)$: a prompt $x$, a preferred ("chosen") answer $y_w$, and a rejected answer $y_l$. A natural source is the book's chapter 7 [[instruction-dataset]]: take each instruction, keep one good answer and add a worse one (for example, a less polite or less correct version).

### The DPO loss

$$\mathcal{L}_{\text{DPO}}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} \;-\; \beta \log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

Symbol by symbol:
- $\mathcal{L}_{\text{DPO}}(\theta)$ = the loss we want to make small; $\theta$ = the [[model-parameters]] being trained.
- $\mathbb{E}_{(x,y_w,y_l)}[\cdot]$ = the average over all preference triples in the dataset (in practice, over a [[mini-batch]]).
- $\pi_\theta(y\mid x)$ = the probability that the model being trained gives to the whole answer $y$ after prompt $x$.
- $\pi_{\text{ref}}(y\mid x)$ = the same probability under the **frozen reference model** (the instruction-fine-tuned model you started from). It is never updated.
- $\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ = "how many times more likely is this answer now than it was at the start?"
- $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ = the same idea on a log scale: positive means "more likely than before," negative means "less likely."
- $\beta$ = a [[hyperparameter]] (typical values 0.1 to 0.5). It controls how much a change in log-ratio counts, and so how tightly the model is held near the reference.
- The big difference inside $\sigma(\cdot)$ = "how much more the chosen answer went up than the rejected answer." Call it the **margin** $m$.
- $\sigma(m) = \frac{1}{1+e^{-m}}$ = the [[sigmoid-function]]; it turns the margin into a probability that "the chosen answer wins."
- $-\log\sigma(m)$ = the [[binary-cross-entropy]] for the label "chosen wins." It is large when $m$ is negative and close to 0 when $m$ is large and positive.

### Computing $\log\pi(y\mid x)$ with the book's model
An answer's log-probability is the sum of its token [[log-probability|log-probabilities]], read from the GPT model's [[logits]] after [[softmax]]:

$$\log\pi_\theta(y\mid x) = \sum_{t=1}^{T}\log\pi_\theta(y_t\mid x, y_{<t})$$

- $y_t$ = token $t$ of the answer; $y_{<t}$ = the tokens before it; $T$ = number of answer tokens.

Only the *answer* tokens count. Prompt tokens and padding are masked out, the same trick the book uses with `ignore_index=-100` in its [[collate-function]] and in [[instruction-masking]] (§7.3, book pp. 221–223).

### Worked example (illustrative numbers)
Take $\beta = 0.1$ and one triple with these summed log-probabilities:

| | trained model $\log\pi_\theta$ | reference $\log\pi_{\text{ref}}$ | log-ratio |
|---|---|---|---|
| chosen $y_w$ | $-12.0$ | $-13.0$ | $+1.0$ |
| rejected $y_l$ | $-15.0$ | $-14.0$ | $-1.0$ |

1. Margin: $m = 0.1\times(+1.0) - 0.1\times(-1.0) = 0.2$.
2. Sigmoid: $\sigma(0.2) = 1/(1+e^{-0.2}) = 1/1.8187 \approx 0.5498$.
3. Loss: $-\log 0.5498 \approx 0.598$.

At the very start of training, $\pi_\theta = \pi_{\text{ref}}$, so both log-ratios are 0, $m = 0$, $\sigma(0) = 0.5$, and the loss is $\log 2 \approx 0.693$. Training pushes the margin up, so the loss falls below $0.693$.

### A short PyTorch sketch (in the style of the book's repository, not printed in the book)
```python
import torch.nn.functional as F

def compute_dpo_loss(model_chosen_logprobs, model_rejected_logprobs,
                     ref_chosen_logprobs, ref_rejected_logprobs, beta=0.1):
    # each argument: tensor of shape (batch_size,), summed answer log-probs
    chosen_logratios = model_chosen_logprobs - ref_chosen_logprobs
    rejected_logratios = model_rejected_logprobs - ref_rejected_logprobs
    logits = beta * (chosen_logratios - rejected_logratios)   # the margin m
    losses = -F.logsigmoid(logits)                            # -log sigma(m)
    return losses.mean()
```
`F.logsigmoid` computes $\log\sigma(m)$ in a numerically stable way (see [[numerical-stability]]). The reference log-probs are computed under `torch.no_grad()` because the reference model is frozen.

## 🔴 Theory (Expert)
**Derivation sketch (Rafailov et al. 2023).**
1. Start from the KL-regularized objective of [[preference-fine-tuning]]: $\max_\pi \mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})$.
2. Its optimum is $\pi^*(y\mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$, where $Z(x) = \sum_y \pi_{\text{ref}}(y\mid x)\exp(r(x,y)/\beta)$ sums over every possible answer, so it is intractable.
3. Solve for the reward: $r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x)$.
4. Plug into the Bradley–Terry likelihood $P(y_w\succ y_l\mid x) = \sigma(r(x,y_w) - r(x,y_l))$. The $\beta\log Z(x)$ terms **cancel** because they depend only on $x$.
5. Replace $\pi^*$ by the trainable $\pi_\theta$ and maximize the log-likelihood of the observed preferences. That gives $\mathcal{L}_{\text{DPO}}$.

So DPO is [[maximum-likelihood-estimation]] of a Bradley–Terry model whose reward is *implicitly* parameterized by the policy: $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$. Hence the paper's subtitle: "your language model is secretly a reward model."

**Gradient.** With $m$ the margin,
$$\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\sigma(-m)\,\big(\nabla_\theta\log\pi_\theta(y_w\mid x) - \nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big].$$
Read it as: raise the chosen answer's log-likelihood, lower the rejected one's, and weight each example by $\sigma(-m)$, which is large when the implicit reward ranks the pair *wrongly*. Correctly ranked pairs contribute little. The gradient flows through the usual [[backpropagation]] of the summed token log-probs.

**Properties and caveats.**
- No sampling during training, no reward model, no value network: one [[forward-pass]] each for $\pi_\theta$ and $\pi_{\text{ref}}$ per answer, then ordinary [[adamw-optimizer|AdamW]] steps. Memory cost is about two models; [[low-rank-adaptation|LoRA]] (appendix E) is a common way to cut it.
- DPO is *offline*: it learns only from the fixed comparison set. It can lower the probability of *both* answers as long as the margin grows, and it can overfit when preferences are near-deterministic (the implicit reward can grow without bound). Variants such as IPO, KTO and ORPO change the loss to address these points.
- $\beta$ plays the role of the KL weight; small $\beta$ allows larger moves away from $\pi_{\text{ref}}$.

## Connections
- Builds on: [[preference-fine-tuning]], [[instruction-fine-tuning]], [[log-probability]], [[sigmoid-function]], [[binary-cross-entropy]], [[cross-entropy-loss]], [[maximum-likelihood-estimation]], [[softmax]]
- Leads to: aligned chat assistants (the "personal assistant" branch of [[llm-building-roadmap]])
- Uses: [[instruction-masking]], [[collate-function]], [[llm-training-loop]], [[adamw-optimizer]], [[numerical-stability]]
- Related: [[logistic-regression]], [[low-rank-adaptation]], [[llm-as-a-judge]]
- People: [[long-ouyang]] (RLHF, the method DPO simplifies), [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9.1, book p. 247 (DPO named only via the `04_preference-tuning-with-dpo` folder, https://mng.bz/dZwD).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 7 references, book p. 298 (Raschka's articles on RLHF and its alternatives, and on reward models).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 221–223 (the `-100` ignore-index masking that DPO reuses for answer log-probs).
- The DPO loss, derivation, gradient, worked numbers and code sketch are standard background from Rafailov et al. (2023), "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model," arXiv 2305.18290. **Not** printed in the book.
