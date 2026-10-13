---
title: Prompt Templates (Alpaca and Phi-3 Styles)
type: concept
tags: [llm, instruction-fine-tuning, prompting, data-formatting]
difficulty: beginner
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Prompt Templates (Alpaca and Phi-3 Styles)

> **One-liner:** A prompt template is a fixed "form" that every instruction example is poured into, so the model always sees requests and answers laid out the same way.

## 🟢 Intuition (Dummy)
Think of a restaurant order slip. Every slip has the same boxes: *Table*, *Dish*, *Notes*. The
cook never has to guess what's what — the layout tells them.

A prompt template does the same for an LLM. Each instruction-dataset entry (a dict with
`instruction`, `input`, `output`) gets written into the same fixed layout, with little signposts
like `### Instruction:` and `### Response:`. After seeing hundreds of these, the model learns:
"when I see `### Response:`, it's my turn to answer — and when I'm done, I stop."

There's no single "correct" template. Different model families use different styles. The book
shows two: the wordy, sectioned **Alpaca** style and the compact **Phi-3** style with special
tags. What matters most is that you **use the same template at training time and at question time.**

## 🟡 Mechanics (Practitioner)

### The two styles side by side (Figure 7.4, book p. 208)
Same entry: `{"instruction": "Identify the correct spelling of the following word.", "input": "Ocassion", "output": "The correct spelling is 'Occasion.'"}`

**Alpaca style** — explicit sections:
```
Below is an instruction that describes a task. Write a response that
appropriately completes the request.

### Instruction:
Identify the correct spelling of the following word.

### Input:
Ocassion

### Response:
The correct spelling is 'Occasion'.
```

**Phi-3 style** (Microsoft) — designated special tokens `<|user|>` and `<|assistant|>`, with the
input merged into the user text:
```
<|user|>
Identify the correct spelling of the following word: 'Ocassion'

<|assistant|>
The correct spelling is 'Occasion'.
```

The book uses **Alpaca** for the rest of chapter 7, because it was one of the early LLMs to
publicly detail its instruction fine-tuning process, is among the most popular styles, and helped
define the original approach (book p. 209).

### The Alpaca formatter in code (Listing 7.2, book p. 209)
```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text
```
- If `entry['input']` is empty, the `### Input:` section is **skipped entirely** (e.g. `data[999]`,
  "What is an antonym of 'complicated'?", book p. 210).
- The response part is built separately:
  `desired_response = f"\n\n### Response:\n{entry['output']}"`.

As a formula, the full training text is a string concatenation:

$$\text{text} = \mathtt{format\_input}(e)\ \Vert\ h\ \Vert\ e[\text{output}]$$

- $e$ = one dataset entry (a dict),
- $\mathtt{format\_input}(e)$ = the preamble + instruction (+ input) — the **model input**,
- $h$ = the fixed response header string `"\n\n### Response:\n"`,
- $\Vert$ = "glue these strings together end to end",
- $e[\text{output}]$ = the **desired response** the model must learn to write.

### The template is reused everywhere
- **Training:** `InstructionDataset` applies `format_input` before tokenizing (step 2.1 in Figure 7.6,
  book p. 212). Every Alpaca prompt begins with the same 9 token IDs
  `21106, 318, 281, 12064, 326, 8477, 257, 4876, 13` — which presumably encode the fixed preamble
  "Below is an instruction that describes a task." (our inference; the book doesn't decode them).
- **Baseline check (§7.5):** the *pretrained* GPT-2 medium gets `format_input(val_data[0])` —
  "Convert the active sentence to passive: 'The chef cooks the meal every day.'" — and produces a
  `### Response:` header but just repeats the input sentence (book p. 229). It knows the shape, not
  the task. See [[baseline-evaluation]].
- **During fine-tuning (§7.6):** `start_context=format_input(val_data[0])` is printed after each
  epoch: "The meal is prepared every day by the chef." (epoch 1) → "The meal is cooked every day by
  the chef." (epoch 2) (book p. 232).
- **Response extraction (§7.7):** because the template marks where the answer begins, we cut off
  the prompt and remove `"### Response:"`:
  `generated_text[len(input_text):].replace("### Response:", "").strip()` (book p. 235). See
  [[response-extraction]].
- **Judging (§7.8):** the scoring prompt is itself a template: "Given the input `{format_input(entry)}`
  and correct output `{entry['output']}`, score the model response `{entry['model_response']}` on a
  scale from 0 to 100, where 100 is the best score." plus "Respond with the integer number only."
  for numeric scores (book pp. 244–246). See [[llm-as-a-judge]].

### Exercise 7.1 — switching to Phi-3 (book pp. 209, 307–308)
```python
def format_input(entry):
    instruction_text = f"<|user|>\n{entry['instruction']}"
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
# extraction: generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
```
Result: fine-tuning is about **17% faster** (shorter inputs), and the judge score stays close to
**50**, in the same range as Alpaca style (book p. 308).

> Book quirks (appendix C): the printed Phi-3 template on p. 307 shows `<user>`/`<assistant>`
> without pipes while the code uses `<|user|>`; and this `format_input` never appends
> `<|assistant|>`, yet extraction strips `"<|assistant|>:"` — the repo has the complete version.

## 🔴 Theory (Expert)
- **Templates define the conditional distribution you train.** The model learns
  $p_\theta(\text{response}\mid\text{template}(x))$. At inference, a different template is a
  distribution shift in the conditioning context, which is why train/test template consistency
  matters more than which template you choose — consistent with Exercise 7.1's near-identical scores.
- **Cost scales with template length.** Every training token costs compute in
  [[self-attention]] (quadratic in sequence length) and in the [[feed-forward-network]] (linear).
  Phi-3's shorter wrapper is the reason for the ~17% speedup.
- **Delimiters as learned control signals.** `### Response:` (or `<|assistant|>`) and the trailing
  [[end-of-text-token]] act as boundaries: the first marks where generation should begin, the second
  where it should stop (the [[collate-function]] keeps exactly one `50256` in each target for this).
  Phi-3's markers are [[special-tokens]]; Alpaca's are ordinary text tokenized by [[byte-pair-encoding]].
- **Template vs. masking.** With [[instruction-masking]], the template tokens would be excluded from
  the loss; the book does *not* mask by default, following Shi et al. (2024)
  (see [[ignore-index-loss-masking]]).
- **Prompt choice as a hyperparameter.** The book lists "try different prompts or instruction
  formats" among strategies to improve the judge score (book p. 247).

## Connections
- Builds on: [[instruction-dataset]], [[tokenization]], [[special-tokens]], [[end-of-text-token]]
- Leads to: [[collate-function]], [[response-extraction]], [[llm-as-a-judge]], [[instruction-masking]]
- Used by: [[instruction-fine-tuning]], [[baseline-evaluation]], [[llm-evaluation]]
- Related: [[autoregressive-text-generation]], [[next-token-prediction]], [[byte-pair-encoding]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2, book pp. 208–210 (Figure 7.4, Listing 7.2, Exercise 7.1 statement).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 212–213 (step 2.1, Figure 7.7 token IDs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book pp. 228–229 (baseline with `format_input`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book pp. 231–232 (start_context samples).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7, book p. 235 (response extraction).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, book pp. 242–247 (judge prompt, "try different prompts").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book p. 297 (Phi-3 technical report, arXiv:2404.14219).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book pp. 307–308 (Exercise 7.1 solution: Phi-3 code, ~17% faster, score ~50).
