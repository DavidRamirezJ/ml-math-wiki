---
title: Few-Shot (In-Context) Learning
type: concept
tags: [llm, prompting, in-context-learning, generalization, gpt]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models]
created: 2026-10-13
updated: 2026-10-13
---

# Few-Shot (In-Context) Learning

> **One-liner:** Few-shot learning means showing an LLM a couple of solved examples inside the prompt, and it picks up the task without any retraining.

## 🟢 Intuition (Dummy)
Imagine handing a friend a sticky note:

> gaot → goat
> sheo → shoe
> pohne → ?

You never said "fix the spelling." Your friend still writes **phone**, because two examples were
enough to show the pattern. That's **few-shot learning**: the "teaching" is just a few examples
placed in the input.

What makes this surprising for a machine is that **nothing inside the model changes**. No
training and no weight updates happen. The model reads the examples, notices the pattern, and
continues it. That's why people also call it **in-context learning**: the learning lives in the
context (the prompt), not in the weights.

Its sibling, [[zero-shot-learning]], is even bolder. You give *no* examples, just a description
("Translate English to German: breakfast =>"), and the model still does it ("Frühstück").

## 🟡 Mechanics (Practitioner)

### The book's definitions (§1.4, book pp. 9–10)
- **Zero-shot learning:** the model generalizes to completely unseen tasks *without any prior
  specific examples* (book p. 9).
- **Few-shot learning:** the model learns from *a minimal number of examples the user provides as
  input* (book p. 9).
- GPT models can do both (book p. 9). A pretrained base model such as [[gpt-3]] has only *limited*
  few-shot ability (§1.3, book p. 7). It is one of the two basic skills it gets from
  [[pretraining]], next to [[text-completion]] (Fig. 1.3, book p. 6).

### Figure 1.6: three ways to use the same model (book p. 10)
None of these needs retraining, [[fine-tuning]], or any change to the architecture.

| Mode | Input (prompt) | Output |
|---|---|---|
| [[text-completion]] | "Breakfast is the" | "most important meal of the day." |
| Zero-shot | "Translate English to German: breakfast =>" | "Frühstück" |
| Few-shot | "gaot => goat", "sheo => shoe", "pohne =>" | "phone" |

The caption sums it up: putting examples of the target task in the input is the **few-shot
setting**, and doing the task with no example is the **zero-shot setting** (book p. 10).

### Writing it as math (our formalization; the book uses words only)
A $k$-shot prompt is a list of $k$ solved examples followed by one new question:
$$P_k = (x_1, y_1,\; x_2, y_2,\; \dots,\; x_k, y_k,\; x_{\text{new}})$$
Symbol by symbol:
- $x_j$ = the input of example $j$ (e.g. "gaot"), and $y_j$ = its answer (e.g. "goat");
- $k$ = how many examples ("shots") you give. In Fig. 1.6, $k = 2$. Zero-shot means $k = 0$;
- $x_{\text{new}}$ = the new question ("pohne").

The model then writes the answer one token at a time with the same next-token distribution it
learned in pretraining ([[next-token-prediction]], [[autoregressive-text-generation]]):
$$\hat y = \text{generate}\big(p_\theta(\,\cdot \mid P_k)\big)$$
- $p_\theta(\cdot \mid P_k)$ = the model's [[probability-distribution]] over the next token, given
  the whole prompt as context;
- $\theta$ = the model's weights. They are **frozen**; nothing about $\theta$ depends on $k$;
- "generate" = pick tokens from that distribution until done, e.g. by [[greedy-decoding]].

Compare that with fine-tuning, which *does* change the weights:
$\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}$ (a [[gradient-descent]] step with
learning rate $\eta$). Few-shot learning never runs this line.

### Practical notes
- The examples use up [[context-length]]. Every shot costs tokens, so $k$ is limited by the
  context window (1,024 tokens for GPT-2).
- The name of the GPT-3 paper is literally "Language Models are **Few-Shot** Learners" (Brown et al.
  2020, appendix B, book pp. 289–290).
- If you need a task done reliably and often, the book's answer is fine-tuning instead:
  [[instruction-fine-tuning]] (general assistant) or [[classification-fine-tuning]] (one fixed
  task). The base GPT-2 in §6.4 fails a zero-shot spam question and just repeats the prompt
  (book p. 183).

## 🔴 Theory (Expert)
- **Learning without gradients.** In the classical sense, "learning" means changing parameters to
  reduce a loss ([[empirical-risk-minimization]]). In-context learning moves the adaptation into
  the *input*. The function $x_{\text{new}} \mapsto p_\theta(\cdot\mid P_k)$ changes with the
  demonstrations while $\theta$ stays fixed. The book doesn't use the term "in-context learning"
  in §1.4; that is standard terminology added here.
- **Where the ability comes from.** Nobody trains GPT to do few-shot tasks. It is an
  [[emergent-behavior]] of large-scale next-word prediction on diverse data (book p. 14, §1.6).
  Web text naturally contains lists, Q&A pairs and worked examples, so continuing a pattern of
  examples is just good next-token prediction.
- **A Bayesian reading** (outside the book). One popular explanation treats the prompt as evidence
  about a hidden "task" variable $\tau$:
  $$p(y\mid P_k) = \sum_{\tau} p(y\mid x_{\text{new}},\tau)\,p(\tau\mid P_k)$$
  More examples sharpen the posterior $p(\tau\mid P_k)$, the way data sharpens a posterior in
  [[bayesian-inference]]. It is an interpretation, not a theorem about transformers. The book
  doesn't discuss it.
- **Limits.** The book stresses the base model's few-shot ability is *limited* (book p. 7).
  Results depend on the wording and order of the examples. Few-shot prompting also can't teach
  the model new facts or new output formats as reliably as fine-tuning on labeled data. That's
  the motivation for Ch. 6–7.
- **Contrast with "few-shot" in classic ML.** In classic machine learning, "few-shot" usually means
  training (updating weights) on very few labeled examples. In the LLM sense used by the book,
  the examples are only *read*, never trained on.

## Connections
- Builds on: [[large-language-model]], [[pretraining]], [[next-token-prediction]], [[foundation-model]], [[probability-distribution]], [[autoregressive-text-generation]], [[context-length]]
- Siblings: [[zero-shot-learning]], [[text-completion]]
- Leads to: [[instruction-fine-tuning]], [[classification-fine-tuning]], [[prompt-template]]
- Related: [[emergent-behavior]], [[gpt-3]], [[gpt-architecture]], [[bayesian-inference]], [[overfitting-and-underfitting]] (generalization)
- People: [[tom-brown]] (GPT-3, "Language Models are Few-Shot Learners")

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 6–7 (base model's limited few-shot capabilities, Fig. 1.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 9–10 (zero-shot and few-shot definitions, Fig. 1.6).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book p. 14 (emergent behavior).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4, book p. 183 (base GPT-2 fails a zero-shot spam prompt).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 289–290 (Brown et al. 2020, "Language Models are Few-Shot Learners").
