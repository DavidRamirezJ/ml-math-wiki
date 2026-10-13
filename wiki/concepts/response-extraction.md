---
title: Response Extraction
type: concept
tags: [llm, text-generation, evaluation, instruction-tuning]
difficulty: beginner
sources: [llm-ch07-fine-tuning-to-follow-instructions, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Response Extraction

> **One-liner:** The model hands back your question *and* its answer glued together, so you cut off the question to keep just the answer.

## 🟢 Intuition (Dummy)
Imagine you write a question at the top of a page and hand it to a friend. They write the answer underneath and give the whole page back. To grade the answer, you'd fold the page so only the answer shows.

That's response extraction. The book's `generate` function returns the prompt plus the new text as one string. That made sense in pretraining, because a pretrained LLM is basically a [[text-completion]] machine: it just continues the text. But when we evaluate an instruction-following model, we want *only* the response (book p. 228). So we chop off the prompt, remove the `### Response:` header, and trim the extra spaces.

## 🟡 Mechanics (Practitioner)

### The three steps (§7.7, book pp. 234–235)
```python
input_text = format_input(entry)
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer).to(device),
    max_new_tokens=256,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = (
    generated_text[len(input_text):]   # 1. cut off the prompt
    .replace("### Response:", "")      # 2. drop the section header
    .strip()                           # 3. trim whitespace
)
```

The same idea written as a formula:

$$
r = g\big[\,|x| :\,\big], \qquad r' = \operatorname{strip}\!\big(\operatorname{replace}(r,\ \texttt{"\#\#\# Response:"},\ \texttt{""})\big)
$$

Symbol by symbol:
- $x$ = the prompt string that went in (from `format_input`).
- $|x|$ = the length of the prompt in **characters** (Python `len`).
- $g$ = the full generated string (prompt + continuation).
- $g[\,|x|:\,]$ = "every character of $g$ from position $|x|$ to the end", i.e., everything after the prompt.
- $r$ = the raw response; $r'$ = the cleaned response we store.

### Why generation stops where it does
- `eos_id=50256` makes `generate` stop early when the model emits `<|endoftext|>` ([[end-of-text-token]]). The model learned to do this because the [[collate-function]] kept one `50256` in every target (§7.3).
- `max_new_tokens=256` is a hard cap in case it never stops.
- No `temperature` or `top_k` is passed, so `generate` uses its Chapter 5 defaults, which is [[greedy-decoding]]: $x_{t+1}=\arg\max_v \text{logits}_t[v]$. (Inferred from Ch. 5; §7.7 doesn't say it.)

### Collecting all test responses (Listing 7.9, book p. 237)
```python
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    ...                                    # same three steps as above
    test_data[i]["model_response"] = response_text

with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)   # indent for pretty-printing
```
On the 110 test items this takes about **1 minute on an A100** and **6 minutes on an M3 MacBook Air**; the progress bar reads `110/110 [01:05<00:00, 1.68it/s]` (book p. 238). Each entry now has four keys, e.g. (book p. 238):

```
{'instruction': 'Rewrite the sentence using a simile.',
 'input': 'The car is very fast.',
 'output': 'The car is as fast as lightning.',
 'model_response': 'The car is as fast as a bullet.'}
```

Saving to JSON lets you score the responses later in a new Python session ([[llm-as-a-judge]]). The book also saves the fine-tuned weights as `gpt2-medium355M-sft.pth` ([[model-checkpointing]]).

### First three side-by-side results (book pp. 235–236)
| Instruction | Correct | Model | Book's verdict |
|---|---|---|---|
| Rewrite "The car is very fast." using a simile | …as fast as lightning. | …as fast as a bullet. | correct |
| Cloud type of thunderstorms? | cumulonimbus | a cumulus cloud | close but not entirely accurate (cumulus clouds can develop into cumulonimbus) |
| Author of *Pride and Prejudice*? | Jane Austen. | The author … is Jane Austen. | correct |

### If you change the prompt format
The marker you strip depends on the [[prompt-template]]. With the Phi-3 style template of Exercise 7.1, the cleanup becomes `.replace("<|assistant|>:", "")` instead of `"### Response:"` (Appendix C, book pp. 307–308).

## 🔴 Theory (Expert)
- **The prefix property.** Slicing by `len(input_text)` works only if `decode(encode(x) + new_ids)` starts with exactly the characters of $x$. GPT-2's byte-level BPE is lossless ($\text{decode}(\text{encode}(x)) = x$) and decoding concatenated token lists concatenates their strings, so the prefix is preserved. A tokenizer that normalizes text (lowercasing, Unicode normalization, stripped spaces) would break this; slicing at the **token** level, `token_ids[:, len(prompt_ids):]`, is the more robust alternative. (My analysis; the book slices characters.) See [[tokenizer-encode-decode]].
- **Generation as conditioning.** The model samples from $p_\theta(y \mid x)=\prod_t p_\theta(y_t \mid x, y_{<t})$; the prompt $x$ is the condition and only $y$ is the "answer". Extraction simply recovers $y$ from the concatenation $x \Vert y$. See [[autoregressive-text-generation]].
- **Failure modes.** A model that hasn't learned the format may never emit EOS and run on into a fake new "### Instruction:" block. That is exactly what the pretrained baseline does in §7.5 (book p. 229; see [[baseline-evaluation]]). Real systems add **stop sequences** (e.g., stop on `"### Instruction"`) on top of EOS. The `.replace` call also removes *any* later occurrence of the header, not only the first.
- **Deterministic decoding for evaluation.** Using greedy decoding (and a fixed seed) makes the extracted responses reproducible, so later score differences come from the model, not from sampling noise ([[random-seed-reproducibility]], [[decoding-strategies]]).

## Connections
- Builds on: [[autoregressive-text-generation]], [[prompt-template]], [[end-of-text-token]], [[tokenizer-encode-decode]], [[greedy-decoding]]
- Leads to: [[llm-as-a-judge]], [[baseline-evaluation]], [[llm-evaluation]]
- Related: [[text-completion]], [[instruction-fine-tuning]], [[model-checkpointing]], [[train-validation-test-split]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5 (why `generate` returns prompt + response; first slicing), book pp. 228–229.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7 (full extraction code, Listing 7.9, JSON saving, timings, example outputs), book pp. 233–238.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 7.1 (Phi-3 marker), book pp. 307–308.
