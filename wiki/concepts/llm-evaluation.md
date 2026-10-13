---
title: LLM Evaluation Methods
type: concept
tags: [llm, evaluation, benchmarks, instruction-fine-tuning, metrics]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# LLM Evaluation Methods

> **One-liner:** Evaluating an LLM means checking how good its answers really are — with quizzes, with human votes, or by asking another (bigger) LLM to grade them.

## 🟢 Intuition (Dummy)
Imagine you trained a student for a year. How do you find out if they learned anything?
You have three classic options:

1. **Give them a multiple-choice exam.** Easy to grade automatically — right or wrong. This is
   what benchmarks like **MMLU** do.
2. **Let people vote.** Show two students' answers side by side to real people and ask "which
   one is better?" That's what **Chatbot Arena** does with chatbots.
3. **Ask a senior teacher to grade the essays.** Instead of a human, you ask a *stronger* LLM
   to score the answers. That's the "LLM-as-a-judge" idea behind **AlpacaEval** — and it's
   exactly what the book does at the end of chapter 7.

Here's the catch that makes LLMs tricky: a classifier answers "spam" or "not spam", so you just
count how many it got right. But a chatbot writes *free text*. "Jane Austen." and "The author
of *Pride and Prejudice* is Jane Austen." are both correct — but they don't match word for word.
So you need smarter ways to judge quality than simple string matching.

And one more thing: a low training loss is *not* the same as good answers. The loss tells you
the model is learning *something*; only evaluation tells you whether the answers are actually
right and helpful.

## 🟡 Mechanics (Practitioner)

### Where evaluation sits in the big picture
In the book's roadmap (Figure 1.9), **step 6 "Model evaluation"** belongs to stage 2 (building
the [[foundation-model]]), described as "the fundamentals of evaluating LLMs, which is essential
for developing capable NLP systems" (book p. 15). In chapter 7, evaluation returns as all of
**stage 3** of the instruction-fine-tuning pipeline: (7) extracting responses, (8) qualitative
evaluation, (9) scoring the responses (Figures 7.18–7.19).

### Why it's harder than classification accuracy
For the chapter 6 spam classifier, the metric was simply

$$\text{accuracy} = \frac{\#\,\text{correct predictions}}{\#\,\text{total predictions}}$$

Reading it symbol by symbol:
- $\#\,\text{correct predictions}$ = how many messages got the right label,
- $\#\,\text{total predictions}$ = how many messages we tested,
- the fraction = the share we got right (see [[classification-accuracy]]).

For free-form instruction responses this simple formula doesn't work, because there is no single
"correct string" to compare against. The book says evaluating instruction-tuned models "is not as
straightforward" as classification (book p. 236).

### The three families of LLM evaluation (book p. 236)
| Family | How it works | Example |
|---|---|---|
| Short-answer / multiple-choice benchmarks | Automatic grading of fixed answers; tests **general knowledge** | **MMLU** — *Measuring Massive Multitask Language Understanding* (arXiv:2009.03300) |
| Human preference comparison | People compare responses from different LLMs | **LMSYS Chatbot Arena** (lmsys.org) |
| Automated conversational benchmarks | Another LLM (e.g. GPT-4) judges the responses | **AlpacaEval** (tatsu-lab.github.io/alpaca_eval) |

The book suggests that in practice it can be useful to consider all three. But since chapter 7
cares about **conversational performance** rather than multiple-choice skill, human evaluation
and automated (LLM-judged) metrics are the more relevant ones.

**Conversational performance** (sidebar, book p. 236) = the model's ability to have human-like
conversations by understanding **context, nuance, and intent**: giving relevant and coherent
answers, staying consistent, and adapting to different topics and styles.

### The book's actual recipe
1. **Inspect the loss** (step 6). Before fine-tuning, the loss on 5 batches is about **3.83**
   (training) and **3.76** (validation) (book p. 230). After 2 epochs on the **935-example training
   split** (`lr=0.00005`, `weight_decay=0.1`, AdamW), the last logged values are training **0.300** and
   validation **0.657** (book pp. 231–232). Good sign — but "the most crucial aspect is its
   performance in terms of response quality and correctness" (book p. 233).
2. **Extract responses** on the 110-entry held-out test set (see [[response-extraction]]) and
   save them to `instruction-data-with-response.json` (book p. 237).
3. **Qualitative check** — read a few side by side (book pp. 235–236):

| Instruction | Correct | Model (GPT-2 medium, fine-tuned) | Verdict |
|---|---|---|---|
| Rewrite "The car is very fast." using a simile | …as fast as lightning. | …as fast as a bullet. | correct |
| Cloud type of thunderstorms? | cumulonimbus | a cumulus cloud | close, not accurate |
| Author of *Pride and Prejudice*? | Jane Austen. | The author … is Jane Austen. | correct |

4. **Score automatically** with a larger LLM as judge (Llama 3 8B via Ollama; see
   [[llm-as-a-judge]] and [[local-llm-inference-ollama]]). Each response gets an integer
   $s_i \in \{0,\dots,100\}$ and the summary is the mean

$$\bar{s} = \frac{1}{N}\sum_{i=1}^{N} s_i$$

- $s_i$ = the judge's score for test response number $i$,
- $N$ = how many responses got a valid score (here all 110),
- $\sum_{i=1}^{N}$ = "add them all up", and $\frac{1}{N}$ = "divide by how many" → the
  [[expected-value-and-mean|average]].

The book's fine-tuned GPT-2 medium (355M) gets **$\bar s = 50.32$**; for reference, Llama 3 8B
*base* scores **58.51** and Llama 3 8B *instruct* scores **82.6** with the same method (book pp. 246–247).

### Why human rating doesn't scale
Human evaluation is laborious: "reading and assigning ratings to all 1,100 responses would require
a significant amount of effort" (book p. 237). (The test set actually has 110 entries; the book
prints 1,100 here — almost certainly a slip.) So the book goes the AlpacaEval way: an automated
LLM judge on a **custom test set**, which gives a more targeted assessment of the use cases the
instruction dataset represents than a generic public benchmark would.

## 🔴 Theory (Expert)
- **Loss vs. task metric.** Training minimizes next-token [[cross-entropy-loss]] (a surrogate,
  see [[surrogate-loss]] and [[empirical-risk-minimization]]); evaluation measures the thing we
  actually care about (correctness, helpfulness). These can diverge: in §7.6 the validation loss
  hovers around 0.66–0.69 during epoch 2 while training loss keeps falling to 0.300 — a gap that
  can hint at [[overfitting]] (our reading; the book only warns that a third epoch "could lead to
  increased overfitting", book p. 232), and the loss says nothing direct about factual errors like
  "cumulus".
- **Evaluation as estimation.** $\bar s$ is a sample mean estimating $\mathbb{E}[S]$, the expected
  judge score over the task distribution. Since the Ollama judge is not fully deterministic across
  operating systems, $S$ is a [[random-variable]] even for fixed responses; the book recommends
  repeating the scoring and averaging (book p. 247), i.e. $\bar{\bar s}=\frac1R\sum_{r=1}^R \bar s^{(r)}$,
  whose variance shrinks like $1/R$ for independent runs (see [[covariance-and-variance]]).
- **Reference-based vs. reference-free.** The book's judge prompt includes the *correct output*,
  so the judge grades against a reference. Arena-style human preference is reference-free and
  pairwise; MMLU is exact-match on fixed options. Each trades off cost, scalability and validity
  differently — hence "consider all three".
- **Judge reliability is itself an open problem.** The book's judge writes "I'd rate *my own*
  response…" (book p. 245) — a reminder that the grader can err. Appendix B points to dedicated
  open judge models (Prometheus, Prometheus 2, PHUDGE) that were not usable via Ollama at the time
  (book p. 297).
- **What scores suggest.** 50.32 (355M, instruction-tuned) < 58.51 (8B base) < 82.6 (8B instruct)
  suggests both [[model-capacity]] and instruction tuning matter (our reading of the numbers; the
  book does not draw this comparison explicitly). Improvement levers the book lists: tune
  [[hyperparameter]]s (learning rate, batch size, epochs), enlarge/diversify the data, try
  different [[prompt-template]]s, or use a larger pretrained model (book p. 247).

## Connections
- Builds on: [[classification-accuracy]], [[expected-value-and-mean]], [[train-validation-test-split]], [[training-and-validation-loss]], [[cross-entropy-loss]]
- Leads to: [[llm-as-a-judge]], [[baseline-evaluation]], [[response-extraction]], [[preference-fine-tuning]]
- Used by: [[instruction-fine-tuning]], [[llm-building-roadmap]], [[local-llm-inference-ollama]]
- Related: [[perplexity]], [[overfitting]], [[random-variable]], [[large-language-model]]
- People: [[dan-hendrycks]] (MMLU — authors not named in the book), [[sebastian-raschka]]

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** §5.1 "Evaluating generative text models" is the book's first evaluation method. It is a *numerical* measure, not a human judgement. The model gives a probability to each correct next token. The [[cross-entropy-loss]] is the negative mean of the log of these probabilities, and [[perplexity]] is $e^{\text{loss}}$. The untrained GPT-2 124M scores loss 10.7940 and perplexity ≈ 48,726 on the 6-token toy example (book pp. 137–140). On *The Verdict* it scores train 10.9876 and validation 10.9811 (book p. 145). §5.2 tracks both losses during training, so the gap between them shows overfitting ([[training-and-validation-loss]]). Ch. 5 also judges the printed text samples by eye after each epoch (book p. 149).

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7, book pp. 14–15 (Figure 1.9, step 6 "Model evaluation").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book pp. 230–233 (losses, Figure 7.17).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7, book pp. 233–238 (three evaluation families, MMLU, Chatbot Arena, AlpacaEval, conversational-performance sidebar, Figure 7.18).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, book pp. 238–247 (judge scores 85/40/95, average 50.32, reference 58.51/82.6).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B (Ch. 7 refs), book p. 297 (Prometheus, PHUDGE).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1–5.1.3, book pp. 129–146 (loss and perplexity as evaluation); §5.2, book pp. 146–151 (loss curves and text samples during training).
