---
title: LLM-as-a-Judge
type: concept
tags: [llm, evaluation, benchmarks, llama, gpt-4]
difficulty: intermediate
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# LLM-as-a-Judge

> **One-liner:** Instead of a human grading every answer, you ask a bigger, smarter LLM to grade them, and average the grades.

## 🟢 Intuition (Dummy)
Grading a spam classifier is easy: count how many labels it got right. Grading a chatbot is hard. "The car is as fast as a bullet" isn't word-for-word the expected "as fast as lightning", but it's still a good answer. You need judgment.

A human could read every answer, but that's slow and tiring. The book notes that human evaluation is "laborious and time-consuming": reading and rating "all 1,100 responses would require a significant amount of effort" (book p. 237). So the book hires a teaching assistant: a much bigger model, **Llama 3 with 8 billion parameters**, running on your own laptop through [[local-llm-inference-ollama|Ollama]]. For each test question, we show the judge the question, the correct answer, and our model's answer, and ask for a score from 0 to 100. Then we average the scores into one number.

The book doesn't actually use the phrase "LLM-as-a-judge", but that's the standard name for what it does.

## 🟡 Mechanics (Practitioner)

### Three ways people evaluate LLMs (book p. 236)
1. **Short-answer and multiple-choice benchmarks**, like **MMLU** (Measuring Massive Multitask Language Understanding, arXiv 2009.03300). They test general knowledge.
2. **Human preference comparisons** against other LLMs, like the **LMSYS Chatbot Arena**.
3. **Automated conversational benchmarks**, where another LLM such as **GPT-4** judges the answers, like **AlpacaEval**.

We care about *conversational* ability, so the book picks option 3, "inspired by AlpacaEval", but on its own test set (book p. 237). The book stresses that this is *not* as straightforward as [[classification-accuracy]] in Chapter 6, where you just count the correct spam/not-spam labels (book p. 236). The book also suggests that, in practice, all three families are useful together; human evaluation and automated conversational metrics are the most relevant for conversational performance (book p. 236).

### The judge
- **Default:** instruction-fine-tuned **Llama 3 8B** from Meta AI, run locally with Ollama (book p. 238).
- **Stronger option:** **GPT-4** via the OpenAI API. The book says the 8B Llama is capable but "not as capable as large proprietary LLMs such as GPT-4" and gives an optional notebook for it (book p. 239).
- **Open judge models** listed in Appendix B: Prometheus, Prometheus 2, and PHUDGE (Phi-3 as judge). They weren't used because Ollama didn't support them at the time (book p. 297).

### The scoring prompt (book pp. 244, 246)
```python
prompt = (
    f"Given the input `{format_input(entry)}` "
    f"and correct output `{entry['output']}`, "
    f"score the model response `{entry[json_key]}`"
    f" on a scale from 0 to 100, where 100 is the best score. "
    f"Respond with the integer number only."
)
```
The judge sees the **reference answer**, so this is reference-guided grading. Without the last sentence, the judge writes a long explanation. With it, it returns just an integer.

### Scoring every test item (Listing 7.11, book p. 246)
```python
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = ...                       # as above
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:                 # judge replied with non-integer text
            print(f"Could not convert score: {score}")
            continue
    return scores
```
`json_key` picks which stored answer to grade (here `"model_response"`), so you can score other models stored under other keys.

### The average score
$$
\bar{s} = \frac{1}{N}\sum_{i=1}^{N} s_i, \qquad s_i \in \{0, 1, \dots, 100\}
$$

Symbol by symbol:
- $\bar{s}$ ("s-bar") = the average score, the one number we report.
- $N$ = how many scores parsed successfully (here 110 of 110).
- $i$ = which test item (1 to $N$).
- $s_i$ = the integer the judge gave to item $i$.
- $\sum_{i=1}^{N}$ = add them all up; $\frac{1}{N}$ = divide by how many there are.
- $\{0,1,\dots,100\}$ = the allowed score values.

### The book's numbers
**First three items, with explanations (book pp. 244–245):**
| Item | Model answer | Judge score |
|---|---|---|
| Simile for "The car is very fast." | as fast as a bullet | **85** (good simile, less vivid than "lightning") |
| Thunderstorm cloud | cumulus (correct: cumulonimbus) | **40** ("a significant error", but partial credit) |
| Author of *Pride and Prejudice* | Jane Austen (repeats the question) | **95** |

In item 3 the judge oddly says "I'd rate *my own* response", a reminder that judges make mistakes too (book p. 245).

**Whole test set (book pp. 246–247):**
| Model | Average $\bar{s}$ |
|---|---|
| Our GPT-2 medium 355M, instruction-fine-tuned | **50.32** (110 of 110 scored, ~1 min on an M3 MacBook Air) |
| Llama 3 8B base, no fine-tuning | **58.51** |
| Llama 3 8B instruct | **82.6** |

**Variants from the exercises (Appendix C):** Phi-3 prompt template ≈ 50 (and ~17% faster training, book p. 308); [[instruction-masking]] ≈ 4 points worse (book p. 309); [[low-rank-adaptation|LoRA]] ≈ 50 with ~28% less training time (book p. 310).

### Improving the score (book p. 247)
Tune [[hyperparameter|hyperparameters]] (learning rate, batch size, epochs), use a larger or more diverse dataset, try other prompt formats, or use a bigger pretrained model ([[model-capacity]]).

## 🔴 Theory (Expert)
- **The score is a random variable.** Each $s_i$ depends on the judge's decoding, which isn't fully deterministic even with `seed=123, temperature=0` (book pp. 241, 247). So $\bar s$ estimates the expected score $\mathbb{E}[s]$ over the test distribution and the judge's randomness ([[random-variable]], [[expected-value-and-mean]], [[monte-carlo-estimation]]).
- **How precise is 50.32?** With $N$ items and per-item standard deviation $\sigma$, the standard error is $\sigma/\sqrt{N}$. If $\sigma \approx 30$ (a guess for 0–100 scores), then $\mathrm{SE} \approx 30/\sqrt{110} \approx 2.9$. A "4 points worse" difference is therefore on the edge of noise unless it's measured as a paired difference or over repeated runs. The book's advice to "repeat the evaluation multiple times and average" (book p. 247) reduces the judge-noise part: averaging $R$ runs, $\bar{\bar s} = \frac1R\sum_r \bar s^{(r)}$, divides that variance by $R$. (Numbers here are my illustration.) See [[covariance-and-variance]].
- **Known judge biases** (from the wider literature, not the book): preference for longer answers (verbosity bias), for answers in a certain position in pairwise setups, and for text in its own style (self-preference). Reference-guided prompts, like the book's, and clear rubrics reduce but don't remove these.
- **Validity.** A judge score is a *proxy* for human judgment. It's useful mainly for **relative** comparisons under a fixed protocol (same judge, prompt, and test set), which is how the book uses it: 50.32 vs. 58.51 vs. 82.6. Absolute values from different judges aren't comparable.
- **Loss vs. task metric.** Training minimizes [[cross-entropy-loss]], but quality is measured by the judge. The two can disagree: a model can have low validation loss and still answer wrongly. That's why the book says response quality is "the most crucial aspect" (book p. 233), much like the gap between [[surrogate-loss|surrogate loss]] and true risk in [[empirical-risk-minimization]].
- **Next step: preferences.** Judge-style or human pairwise preferences are also the data for [[preference-fine-tuning]] such as [[direct-preference-optimization]], which the book points to as an optional step after instruction tuning (book p. 247).

## Connections
- Builds on: [[response-extraction]], [[local-llm-inference-ollama]], [[prompt-template]], [[expected-value-and-mean]], [[baseline-evaluation]]
- Leads to: [[llm-evaluation]], [[preference-fine-tuning]], [[direct-preference-optimization]]
- Related: [[classification-accuracy]], [[instruction-fine-tuning]], [[instruction-masking]], [[model-capacity]], [[low-rank-adaptation]], [[random-variable]], [[covariance-and-variance]], [[gpt-3]]
- People: [[sebastian-raschka]], [[dan-hendrycks]] (MMLU; authorship from outside the book)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7 (evaluation families: MMLU, Chatbot Arena, AlpacaEval/GPT-4; why an LLM judge), book pp. 233–238.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8 (Llama 3 judge, GPT-4 sidebar, scoring prompts, Listing 7.11, scores 85/40/95, 50.32, 58.51, 82.6, improvement strategies), book pp. 238–247.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9 + Chapter 7 summary (scoring with another LLM; preference fine-tuning), book pp. 247–249.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B (Prometheus, Prometheus 2, PHUDGE), book p. 297; Appendix C (Exercises 7.1, 7.2, 7.4 scores), book pp. 307–310.
