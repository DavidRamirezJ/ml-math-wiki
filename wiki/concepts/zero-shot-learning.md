---
title: Zero-Shot Learning
type: concept
tags: [llm, generalization, prompting, nlp]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Zero-Shot Learning

> **One-liner:** Zero-shot learning is when a model does a task it was never shown an example of — you just ask, and it figures it out.

## 🟢 Intuition (Dummy)
Picture a friend who has read millions of books but never took a German class. You say:
"Translate English to German: breakfast =>" and they answer "Frühstück". Nobody gave them a
practice example of *this* exact task. They just used everything they already soaked up.

That's zero-shot. "Shot" here means "example". **Zero** shots = zero examples. Compare:
- **Zero-shot:** "Do this task." (no examples)
- **[[few-shot-learning|Few-shot]]:** "Here are 2 examples of the task. Now do one more."

The cool part: the model's brain (its weights) doesn't change at all. You don't retrain it. You
only change what you type into it.

## 🟡 Mechanics (Practitioner)
**The book's definition** (Raschka §1.4, p. 9): zero-shot learning is "the ability to generalize to
completely unseen tasks without any prior specific examples." GPT models, trained mainly for
[[text-completion]], turn out to be good at both zero-shot and few-shot tasks.

**The book's three modes of use (Fig. 1.6, p. 10)** — none needs retraining or architecture changes:

| Mode | Input (prompt) | Output |
|---|---|---|
| [[text-completion\|Text completion]] | "Breakfast is the" | "most important meal of the day." |
| **Zero-shot** | "Translate English to German: breakfast =>" | "Frühstück" |
| [[few-shot-learning\|Few-shot]] | "gaot => goat, sheo => shoe, pohne =>" | "phone" |

**How it works, in math.** An LLM is a [[next-token-prediction|next-token predictor]]: given a
sequence of [[token|tokens]], it outputs a [[probability-distribution]] over the next one. A zero-shot
task just *phrases the task as text* and lets the model continue it:

$$\hat{y} = \arg\max_{y}\; p_\theta\big(y \mid \underbrace{\text{instruction}}_{\text{the prompt}}\big)$$

Reading it symbol by symbol:
- $p_\theta(\,\cdot \mid \cdot\,)$ = "the probability the model assigns", where $\theta$ (theta) = all the model's learned weights ([[model-parameters]]). The bar $\mid$ means "given".
- "instruction" = the prompt text, e.g. "Translate English to German: breakfast =>".
- $y$ = a candidate answer (one token or a string of tokens).
- $\arg\max_y$ = "pick the $y$ that makes the probability as large as possible" ([[argmax]], i.e. [[greedy-decoding]]).
- $\hat{y}$ ("y-hat") = the model's answer, here "Frühstück".

The key point hiding in the formula: $\theta$ is **fixed**. Nothing gets trained. The only thing that
changes from task to task is the conditioning text. The book does not use the name, but this is
usually called *in-context learning*.

**Zero-shot vs. fine-tuning.** The other way to get a task done is [[fine-tuning]], which *does*
change $\theta$ using labeled examples (Ch. 6 [[classification-fine-tuning]], Ch. 7
[[instruction-fine-tuning]]). Zero-shot is free and instant. Fine-tuning usually gives better
accuracy on a narrow task.

## 🔴 Theory (Expert)
- **Why it is possible at all.** [[pretraining]] minimizes next-token [[cross-entropy-loss]] on a huge,
  diverse corpus (GPT-3: 300B tokens sampled from 499B, §1.5). Web text naturally contains
  patterns like "X in German is Y", so "translate" is already one of the conditional distributions
  the model learned. Zero-shot ability is a by-product of the [[maximum-likelihood-estimation]]
  objective, not a separately trained skill. This is one form of [[emergent-behavior]].
- **Generalization, not memorization.** In the classic ML sense ([[overfitting-and-underfitting]]),
  generalization means doing well on new *inputs* from the same task. Zero-shot asks for more:
  doing well on a new *task*. The "task" is encoded inside the input string, so for the LLM a new
  task is just a new region of input space.
- **Meaning shift across fields.** In older computer-vision literature, "zero-shot learning" means
  recognizing *classes never seen in training* using side information (e.g. attribute vectors).
  The LLM sense used in this book ("no task-specific examples in the prompt") is related but not the
  same. When you read papers, check which one is meant.
- **Sensitivity to wording.** Because the answer is $\arg\max_y p_\theta(y\mid\text{prompt})$, small
  changes in prompt wording change the conditioning and can change the answer. This is why
  [[prompt-template|prompt templates]] matter so much later in the book (Ch. 7).
- **Base model vs. instruction model.** The book notes (§1.3, p. 7) that the GPT-3 *base* model
  already has limited few-shot ability. Instruction fine-tuning (Ch. 7) makes zero-shot use much more
  reliable, since the model is trained directly to answer instructions.

### From llm-ch06-fine-tuning-for-classification
Ch. 6 shows a failed zero-shot attempt. The book gives the pretrained GPT-2 (124M) the prompt "Is the following text 'spam'? Answer with 'yes' or 'no': 'You are a winner …'". The model does not answer. It only repeats part of the prompt ("The following text 'spam'? Answer with 'yes' or 'no': …"). The book concludes that the model "has only undergone pretraining and lacks instruction fine-tuning" (§6.4, book p. 183). So a small base model has almost no zero-shot classification ability. The fix in Ch. 6 is to change $\theta$ with labeled data ([[classification-fine-tuning]]).

## Connections
- Builds on: [[large-language-model]], [[next-token-prediction]], [[pretraining]], [[probability-distribution]], [[gpt-architecture]]
- Leads to: [[few-shot-learning]], [[instruction-fine-tuning]], [[prompt-template]]
- Contrasts with: [[fine-tuning]], [[classification-fine-tuning]]
- Related: [[text-completion]], [[emergent-behavior]], [[foundation-model]], [[gpt-3]], [[overfitting-and-underfitting]]
- People: [[tom-brown]] (GPT-3 paper, "Language Models are Few-Shot Learners")

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 9–10 (definition p. 9; Fig. 1.6 p. 10, PDF pp. 31–32).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 7 (GPT-3 base model's limited few-shot ability) — context.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4, book p. 183 (GPT-2 small fails the zero-shot spam prompt) — see [[llm-ch06-fine-tuning-for-classification]].
