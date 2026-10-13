---
title: Preference Fine-Tuning
type: concept
tags: [llm, fine-tuning, alignment, deep-learning]
difficulty: advanced
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Preference Fine-Tuning

> **One-liner:** Preference fine-tuning teaches a model that already follows instructions to give the *kind* of answers people like best, by showing it pairs of answers where one is marked "better."

## 🟢 Intuition (Dummy)
Picture a new assistant who has learned to do what you ask. Ask for a summary, you get a summary. That is what [[instruction-fine-tuning]] gives you. But some summaries are nicer than others: shorter, kinder, more accurate, better formatted. So you sit with the assistant and say, "for this request, answer A is better than answer B." After many such comparisons, the assistant learns your taste.

That is preference fine-tuning. It does not teach the model *what* to do (it already knows that). It teaches the model *how you like it done*. The book calls it an **optional** last step that comes **after** instruction fine-tuning, and says it is "particularly useful for customizing a model to better align with specific user preferences" (book p. 247). Appendix B repeats the idea: it is "an optional step after instruction fine-tuning to align the LLM more closely with human preferences" (book p. 298).

Why the word "preference"? Because people find it much easier to say "I like this one more" than to write the perfect answer from scratch. So the training data is comparisons, not model answers.

## 🟡 Mechanics (Practitioner)

### Where it sits in the book's pipeline
The book's roadmap (Figure 7.21, book p. 248) is: build the LLM → [[pretraining]] → [[foundation-model]] → fine-tune, either for a classifier ([[classification-fine-tuning]]) or a personal assistant ([[instruction-fine-tuning]]). Section 7.9.1 then adds one more optional step:

$$\text{pretrained model} \;\xrightarrow{\text{instruction fine-tuning}}\; \text{instruction model} \;\xrightarrow{\text{preference fine-tuning}}\; \text{aligned model}$$

The book does not code this step. It points to the `04_preference-tuning-with-dpo` folder in its GitHub repository (https://mng.bz/dZwD) and to two of the author's articles: "LLM Training: RLHF and Its Alternatives" (https://mng.bz/ZVPm) and "Tips for LLM Pretraining and Evaluating Reward Models" (https://mng.bz/RNXj) (book pp. 247, 298). The repository README also lists more bonus material in its "Bonus Material" section (https://mng.bz/r12g, book p. 248).

> Everything below this line in the Mechanics and Theory tiers is **standard background, not from the book's text**. It is here so the step makes sense mathematically.

### What the data looks like
Each training example is a triple $(x, y_w, y_l)$:
- $x$ = the prompt (for example, the book's Alpaca-style instruction plus input, see [[prompt-template]]),
- $y_w$ = the "winning" answer, the one people preferred,
- $y_l$ = the "losing" answer, the one people liked less.

Compare this with chapter 7's [[instruction-dataset]], where each example has only *one* answer.

### How preferences become a number: the Bradley–Terry model
The standard way to turn "A beats B" into maths is to give each answer a hidden score $r(x, y)$ and say:

$$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big), \qquad \sigma(z) = \frac{1}{1+e^{-z}}$$

Symbol by symbol:
- $P(\cdot)$ = a probability, a number between 0 and 1.
- $y_w \succ y_l$ = "$y_w$ is preferred over $y_l$." The curly symbol $\succ$ means "beats."
- $\mid x$ = "given the prompt $x$."
- $r(x, y)$ = the reward: a single number that says how good answer $y$ is for prompt $x$.
- $r(x, y_w) - r(x, y_l)$ = the score gap between the two answers.
- $\sigma$ = the [[sigmoid-function]], which squashes any number into $(0, 1)$. A big positive gap gives a probability near 1; a gap of 0 gives exactly 0.5.

So "the better answer has a much higher score" means "people will almost surely prefer it."

### Two families of methods
1. **RLHF (reinforcement learning from human feedback).** First train a separate *reward model* $r_\phi$ on the comparisons, using the loss $-\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$. Then use reinforcement learning (often PPO) to push the LLM toward answers with high reward, while a penalty keeps it close to the instruction model. This is the InstructGPT recipe (see [[long-ouyang]]).
2. **Direct methods, such as [[direct-preference-optimization]] (DPO).** Skip the reward model and the RL loop. Train the LLM directly on the comparisons with a simple classification-style loss. This is the method the book's repository uses.

### The common goal, written as one formula
Both families try to solve this problem:

$$\max_{\pi_\theta}\; \mathbb{E}_{x,\; y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big] \;-\; \beta\, \mathrm{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$

Symbol by symbol:
- $\max_{\pi_\theta}$ = "find the model that makes the rest as large as possible."
- $\pi_\theta$ = the LLM being tuned, seen as a *policy*: a [[probability-distribution]] over answers. $\theta$ = its [[model-parameters]].
- $y \sim \pi_\theta(\cdot\mid x)$ = "an answer $y$ sampled from the model for prompt $x$."
- $\mathbb{E}[\cdot]$ = the average (see [[expected-value-and-mean]]).
- $r(x,y)$ = the reward from above. First part: "give answers people like."
- $\pi_{\text{ref}}$ = the frozen **reference model**, usually the instruction-fine-tuned model you started from.
- $\mathrm{KL}(\cdot\,\|\,\cdot)$ = the Kullback–Leibler divergence, a measure of how different two distributions are. It is 0 when they are the same.
- $\beta$ = a [[hyperparameter]] that sets how strongly the model is held near the reference. Second part: "but do not drift too far from what you already were."

The KL penalty matters. Without it, the model can find odd answers that score high but read badly ("reward hacking"), or forget what it learned during [[pretraining]] and instruction tuning.

### Link to what the book already built
The model output is the same as in chapters 4–7: [[logits]] → [[softmax]] → next-token probabilities. The probability of a whole answer is the product of its token probabilities, so its [[log-probability]] is a sum:

$$\log \pi_\theta(y\mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})$$

- $y_t$ = the $t$-th token of the answer; $y_{<t}$ = all tokens before it; $T$ = the answer length.

This is exactly the quantity whose negative average is the [[cross-entropy-loss]] in chapter 5 and chapter 7. Preference methods reuse it, but they compare *two* answers instead of fitting *one*.

## 🔴 Theory (Expert)
- **From maximum likelihood to pairwise likelihood.** Instruction fine-tuning is [[maximum-likelihood-estimation]] on demonstrations: maximize $\log\pi_\theta(y\mid x)$. Preference learning is maximum likelihood on *comparisons* under the Bradley–Terry model: maximize $\log\sigma(\Delta r)$. The reward-model loss is therefore a [[binary-cross-entropy]] / [[logistic-regression]] loss on the score difference, with label "1 = $y_w$ wins."
- **Shift invariance.** Bradley–Terry depends only on reward *differences*, so $r$ is identified only up to an additive function $c(x)$ of the prompt. This fact is what lets DPO cancel the intractable partition function (see [[direct-preference-optimization]]).
- **Closed-form optimum of the KL-regularized objective.** For a fixed reward, the objective above has the solution
  $\pi^*(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\,\exp\!\big(r(x,y)/\beta\big)$,
  a Gibbs / Boltzmann reweighting of the reference model. $Z(x)$ sums over all possible answers and cannot be computed, which is why RLHF needs sampling-based RL and DPO needs a reparameterization.
- **Why do it at all?** Next-token likelihood treats every demonstration token as "correct" and gives no signal about *which* plausible answer is better. Comparisons carry relative-quality information that demonstrations cannot. The book's appendix B also cites Gekhman et al. (2024): models "primarily acquire factual knowledge during pretraining," fine-tuning "mainly enhances their efficiency in using this knowledge," and new facts introduced during fine-tuning are learned more slowly and increase the tendency to generate incorrect information (book pp. 297–298). Preference tuning is a style and alignment step, not a knowledge step.
- **Failure modes.** Reward over-optimization (Goodhart's law), length bias (longer answers often win comparisons), and sensitivity to $\beta$. Evaluation again needs care; the book's [[llm-as-a-judge]] approach (Ollama + Llama 3 8B, §7.8) is one way to score the result.

## Connections
- Builds on: [[instruction-fine-tuning]], [[fine-tuning]], [[cross-entropy-loss]], [[log-probability]], [[sigmoid-function]], [[softmax]], [[maximum-likelihood-estimation]]
- Leads to: [[direct-preference-optimization]]
- Related: [[binary-cross-entropy]], [[logistic-regression]], [[llm-as-a-judge]], [[low-rank-adaptation]] (often used to make preference tuning cheaper), [[llm-building-roadmap]]
- People: [[long-ouyang]] (InstructGPT / RLHF), [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9.1, book p. 247 (preference fine-tuning as an optional step after instruction fine-tuning; `04_preference-tuning-with-dpo` folder, https://mng.bz/dZwD).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9, Figure 7.21, book p. 248 (the three stages of coding an LLM).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 7 references, book pp. 297–298 (Gekhman et al. 2024; Raschka, "LLM Training: RLHF and Its Alternatives"; "Tips for LLM Pretraining and Evaluating Reward Models").
- The Bradley–Terry model, the RLHF recipe, the KL-regularized objective and its closed-form optimum are standard background (Ouyang et al. 2022; Rafailov et al. 2023), **not** stated in the book.
