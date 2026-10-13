---
title: Baseline Evaluation
type: concept
tags: [llm, evaluation, fine-tuning, experimental-design]
difficulty: beginner
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Baseline Evaluation

> **One-liner:** Before you train a model, test it once so you know what "before" looks like. Then you can tell whether training actually helped.

## 🟢 Intuition (Dummy)
Say you start a workout plan. If you never weighed yourself or timed your run on day one, how would you know the plan worked? A **baseline** is that day-one measurement.

In Chapter 7, the book does exactly this. Right after loading the pretrained GPT-2 medium model, and *before* any instruction fine-tuning, it asks the model to do one task: "Convert the active sentence to passive: 'The chef cooks the meal every day.'" The model doesn't do it. It just repeats the sentence and starts writing a new "### Instruction:" (book pp. 228–229). That's our "before" picture. It tells us the pretrained model can't follow instructions yet, and it gives us something to compare the fine-tuned model against later.

## 🟡 Mechanics (Practitioner)

### The book's baseline check (§7.5, book pp. 228–229)
1. Fix the seed so the run is reproducible ([[random-seed-reproducibility]]): `torch.manual_seed(123)`.
2. Take the **first validation example**, `val_data[0]`, and format it with the Alpaca-style [[prompt-template]] (`format_input`).
3. Generate up to 35 new tokens with the Chapter 5 `generate` function, stopping at `<|endoftext|>` (ID 50256).
4. Cut the prompt off the output ([[response-extraction]]) and read what's left.

```python
torch.manual_seed(123)
input_text = format_input(val_data[0])
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],   # 1024
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()
```

Output (book p. 229):

```
### Response:

The chef cooks the meal every day.

### Instruction:

Convert the active sentence to passive: 'The chef cooks the
```

Verdict: no passive voice. After fine-tuning (§7.6, book p. 232) the model answers "The meal is cooked every day by the chef."

### Baselines as numbers
A single example is a *qualitative* baseline. Section 7.8 adds *quantitative* reference points, all scored the same way by the Llama 3 judge ([[llm-as-a-judge]]) on the same test set (book p. 247):

| System | Average score |
|---|---|
| Our fine-tuned GPT-2 medium | 50.32 |
| Llama 3 8B **base** (no fine-tuning) | 58.51 |
| Llama 3 8B **instruct** | 82.6 |

The book calls a score above 50 "a useful benchmark" for comparing other models or training setups (book pp. 246–247).

### The comparison formula
The thing we really care about is the improvement over the baseline:

$$
\Delta = \bar{s}_{\text{after}} - \bar{s}_{\text{before}}
$$

- $\Delta$ (delta) = the change caused by our intervention (fine-tuning, masking, LoRA…).
- $\bar{s}$ (s-bar) = the average score on the test set; the bar means "average".
- "after" / "before" = the model with and without the change, evaluated **the same way** (same data, same prompt, same judge, same seed).

Example from Appendix C: instruction masking gives about $\Delta \approx -4$ points versus the unmasked model (book p. 309; see [[instruction-masking]]).

## 🔴 Theory (Expert)
- **Controlled comparison.** A baseline is the control arm of an experiment. A difference $\Delta$ means something only if *everything else* stays fixed: the data split ([[train-validation-test-split]]), the decoding settings, the prompt, and the judge. Changing two things at once confounds the result.
- **Paired design.** Because both models answer the *same* $N$ test items, the right noise estimate uses per-item differences $d_i = s_i^{\text{after}} - s_i^{\text{before}}$. The standard error is $\mathrm{SE}(\bar d)=\mathrm{sd}(d)/\sqrt N$. That is usually much smaller than comparing two independent averages, because item difficulty cancels out. (Standard statistics, not from the book.) See [[covariance-and-variance]].
- **Noise floor.** The book notes Ollama is not fully deterministic, so scores can shift between runs and operating systems; it recommends repeating and averaging (book p. 247). A difference smaller than this run-to-run spread is not evidence of improvement.
- **Kinds of baselines.** (a) the untrained or pretrained starting point (the book's §7.5 check), (b) a strong external reference (Llama 3 8B instruct), (c) a trivial baseline (e.g., always copying the input, the failure mode §7.5 shows). Good reports include at least (a) and (b).
- **Validation vs. test.** The §7.5 sanity check uses a *validation* example; the §7.8 scores use the held-out *test* set. Keeping the test set untouched until the end prevents you from tuning on it.

## Connections
- Builds on: [[train-validation-test-split]], [[pretrained-weights]], [[autoregressive-text-generation]], [[random-seed-reproducibility]]
- Leads to: [[llm-as-a-judge]], [[llm-evaluation]], [[instruction-fine-tuning]]
- Related: [[response-extraction]], [[model-capacity]], [[expected-value-and-mean]], [[covariance-and-variance]], [[text-completion]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5 (baseline test on `val_data[0]`, generation code, output), book pp. 228–229.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6 (fine-tuned answer to the same example), book p. 232.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8 (reference scores 58.51 / 82.6; 50.32 benchmark; repeat-and-average advice), book pp. 246–247.
