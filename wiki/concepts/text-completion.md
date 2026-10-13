---
title: Text Completion (vs Instruction Following)
type: concept
tags: [llm, generation, pretraining, instruction-tuning, gpt]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Text Completion (vs Instruction Following)

> **One-liner:** Text completion is what a pretrained LLM naturally does: it keeps writing whatever text you started, which is not the same as answering what you asked.

## 🟢 Intuition (Dummy)
Think of the phone keyboard that suggests your next word, then imagine it is enormously well
read. Type "Breakfast is the" and it finishes: "most important meal of the day." That's
**text completion**. A pretrained LLM is trained on only one game, "guess the next word," so
continuing your text is the thing it does best (book pp. 6–7, 10).

Now type a *request*: "Convert the active sentence to passive: 'The chef cooks the meal every
day.'" A pure completion model doesn't see an order. It sees the start of some document and
asks itself "what text usually comes next?" Maybe it repeats the sentence, or maybe it writes
another instruction, as if it were copying a worksheet. That's exactly what the book's
pretrained GPT-2 does (book p. 229).

**Instruction following** is a learned habit on top: when text looks like a request, the next
words should be a helpful *answer*, and then stop. You get that habit through
[[instruction-fine-tuning]]. Same engine, different manners.

## 🟡 Mechanics (Practitioner)

### What the book says
- After [[pretraining]], the base/[[foundation-model]] has basic skills: **text completion** (finishing a
  half-written sentence) and limited **few-shot** ability (§1.3, book pp. 6–7; Fig. 1.3).
- GPT models are "primarily designed and trained to perform text completion tasks," yet show "remarkable versatility" (§1.4, book p. 9).
  In Fig. 1.5 it learns from *incomplete* texts ("This is an example of how concise I can") to
  produce the next word ("be") (book p. 9).
- GPT-3-style models "are competent text completion models and can carry out other tasks such as
  spelling correction, classification, or language translation" (§1.6, book p. 12).
- Pretrained LLMs "often struggle with specific instructions" such as "Fix the grammar in this
  text" or "Convert this text into passive voice" (§7.1, book p. 205).

### How completion is computed
Completion is [[autoregressive-text-generation]]: predict one token, append it, repeat. With
[[greedy-decoding]] (what `generate_text_simple` and the book's `generate` do when no temperature
or top-k is set), each step is
$$x_{t+1} = \arg\max_{v\in\{0,\dots,V-1\}} \; \mathrm{softmax}(\mathbf z_t)_v$$
Symbol by symbol:
- $x_1,\dots,x_t$ = the token IDs so far (your prompt plus anything already generated);
- $\mathbf z_t\in\mathbb R^{V}$ = the [[logits]] vector the model outputs at the last position;
- $V$ = vocabulary size, 50,257 for GPT-2;
- $\mathrm{softmax}(\mathbf z_t)_v$ = the probability that the next token is $v$ ([[softmax]]);
- $\arg\max_v$ = "pick the token $v$ with the highest probability" ([[argmax]]);
- $x_{t+1}$ = the new token, which is appended before the next step.

Behind it is the [[next-token-prediction]] factorization learned in pretraining:
$$p_\theta(x_1,\dots,x_T)=\prod_{t=1}^{T}p_\theta(x_t\mid x_1,\dots,x_{t-1})$$
"The probability of a whole text = probability of word 1, times probability of word 2 given word
1, times …". $\theta$ = the model's weights, $T$ = the text length.

### The output contains your prompt
Because the model just *continues* text, the book's `generate` returns **prompt + continuation**
as one sequence. This "was previously convenient since pretrained LLMs are primarily designed as
text-completion models" (§7.5, book p. 228). For an assistant you want only the answer, so the book cuts off
the prompt ([[response-extraction]], book pp. 228–229):

```python
token_ids = generate(model=model, idx=text_to_token_ids(input_text, tokenizer),
                     max_new_tokens=35, context_size=BASE_CONFIG["context_length"],
                     eos_id=50256)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()
```

### Side by side: the same prompt, before and after instruction fine-tuning
Prompt (Alpaca style, `val_data[0]`): "Convert the active sentence to passive: 'The chef cooks the meal every day.'"

| Model | What it writes (book pp. 229, 232) |
|---|---|
| Pretrained GPT-2 medium (completion only) | `### Response:` The chef cooks the meal every day. `### Instruction:` Convert the active sentence to passive: 'The chef cooks the … |
| After 1 epoch of instruction fine-tuning | The meal is prepared every day by the chef. |
| After 2 epochs | The meal is cooked every day by the chef. ✔ |

Another completion example, from Ch. 6: GPT-2 small turns "Every effort moves you" into
"…forward. The first step is to understand the importance of your work". That's fluent, so the
weights loaded fine. But asked *"Is the following text 'spam'? Answer with 'yes' or 'no': …"* it
just repeats the question (§6.4, book pp. 182–183).

### Completion vs. instruction following at a glance

| | Text completion (base model) | Instruction following (fine-tuned) |
|---|---|---|
| Trained on | raw, unlabeled text | instruction → response pairs |
| Treats your input as | the beginning of a document | a request to answer |
| Output | continues in the same style | a direct answer, then end-of-text |
| How you steer it | phrase the start cleverly, or [[few-shot-learning]] examples | just ask |
| Book chapters | Ch. 4–5 | Ch. 7 |

## 🔴 Theory (Expert)
- **Same distribution family, different data.** Both behaviors come from the same object, a
  conditional next-token distribution $p_\theta(x_{t+1}\mid x_{\le t})$. Pretraining fits it to web
  text, where a line like "Convert this to passive voice:" is often followed by *more exercises*.
  [[instruction-fine-tuning]] refits it ([[maximum-likelihood-estimation]] with the same
  [[cross-entropy-loss]]) on text where that line is followed by the *answer*. The objective
  doesn't change, only the empirical distribution does (book pp. 229–230). That this is "just a
  data shift" is our framing, consistent with the book's statement that the training loop is
  reused unchanged.
- **Stopping is learned too.** A completion model has no reason to stop. An instruction-tuned
  model is trained with one `<|endoftext|>` (ID 50256) target after each answer, so it learns to
  end (book p. 218). Generation then stops at `eos_id=50256` ([[end-of-text-token]]). Even after
  fine-tuning, the epoch samples printed during training keep going past `<|endoftext|>` into a
  new invented instruction, because that printing doesn't stop at EOS (book p. 232).
- **Completion is still the engine of other skills.** [[few-shot-learning]] and
  [[zero-shot-learning]] are text completion with a well-designed prefix. Fig. 1.6 puts all three
  modes side by side, and none of them needs retraining (book p. 10). Translation also emerges from
  completion on multilingual data ([[emergent-behavior]], book p. 14).
- **Decoding choice matters.** Greedy decoding, as above, is deterministic and tends to repeat
  itself. [[temperature-scaling]] and [[top-k-sampling]] (Ch. 5) trade determinism for variety;
  see [[decoding-strategies]].
- **Terminology note.** In §7.7 the book once calls the Ch. 6 classifier "completion fine-tuning"
  (book p. 236). That is a slip: Ch. 6 is [[classification-fine-tuning]], which doesn't produce
  text at all.

## Connections
- Builds on: [[next-token-prediction]], [[autoregressive-text-generation]], [[pretraining]], [[softmax]], [[argmax]], [[logits]], [[probability-distribution]], [[gpt-architecture]], [[decoder-only-transformer]]
- Contrast: [[instruction-fine-tuning]], [[classification-fine-tuning]]
- Related: [[few-shot-learning]], [[zero-shot-learning]], [[foundation-model]], [[gpt-3]], [[gpt-2]], [[greedy-decoding]], [[decoding-strategies]], [[response-extraction]], [[end-of-text-token]], [[emergent-behavior]]
- Leads to: [[instruction-fine-tuning]], [[prompt-template]]
- People: [[alec-radford]] (GPT lineage), [[tom-brown]] (GPT-3)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 6–7 (base model can do text completion; Fig. 1.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 9–10 (GPT trained mainly for text completion; Figs. 1.5, 1.6).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book pp. 12, 14 (good text-completion models; emergent translation).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4, book pp. 182–183 (completion sanity check; spam prompt fails).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.1, book p. 205 (pretrained LLMs struggle with instructions).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book p. 218 (one end-of-text target teaches stopping).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book pp. 228–229 (generate returns input + output; baseline response).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book p. 232 (responses after epochs 1 and 2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7, book p. 236 ("completion fine-tuning" wording).
