---
title: Instruction Masking
type: concept
tags: [llm, fine-tuning, instruction-tuning, loss-function, pytorch]
difficulty: intermediate
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Instruction Masking

> **One-liner:** Instruction masking means the model is only graded on the *answer* part of a training example, not on the question part it was given.

## 🟢 Intuition (Dummy)
Picture a student practicing with flashcards. Each card has a question on the front and the answer on the back. When you check the student's work, you only care whether they got the *answer* right. You don't take points off because they couldn't recite the question word for word. After all, the question was handed to them!

Instruction masking does the same thing for an LLM. During [[instruction-fine-tuning]], every training example is one long piece of text: a fixed preamble ("Below is an instruction…"), the instruction, an optional input, and then the response. By default the model gets trained to predict *every* next token in that text, including the instruction tokens. With instruction masking, we tell the loss function: "ignore the instruction part, and only score the response."

The idea is that the model should spend its effort learning to *answer well*, not memorizing instructions. Here's the twist, though: the book reports that, in practice, this doesn't always help. When the book's author tried it on the chapter's model, the score actually got about 4 points *worse*. So it's an optional knob, not a rule.

## 🟡 Mechanics (Practitioner)

### Where it fits
In the book's batching pipeline (§7.3), the [[collate-function]] already builds targets by shifting the inputs one position to the left. It also replaces all but the first padding token with `-100` ([[ignore-index-loss-masking]]). Instruction masking uses the **same `-100` trick** one more time, this time on the positions that belong to the instruction (book p. 222, Figure 7.13).

Figure 7.13 shows it with the example "Rewrite the following sentence using passive voice." / input "The team achieved great results." / response "Great results were achieved by the team." The input token IDs start `[21106, 318, 281, 12064, 326, …]`. In the target list, the instruction positions become `[-100, -100, -100, -100, -100, …, 13, 50256]`, so only the response tokens and the final `<|endoftext|>` are left to score (book p. 222).

### The formula
Ordinary next-token [[cross-entropy-loss]] with PyTorch's `ignore_index` averages only over the positions that are *not* `-100`:

$$
\mathcal{L} = -\frac{1}{|\mathcal{R}|}\sum_{t\in\mathcal{R}} \log p_\theta\!\left(y_t \mid x_{\le t}\right),
\qquad \mathcal{R} = \{\,t : y_t \neq -100\,\}
$$

Reading it symbol by symbol:
- $\mathcal{L}$ = the loss, one number for the whole sequence (or batch).
- $t$ = a position in the sequence (0, 1, 2, …).
- $y_t$ = the target token ID at position $t$ (the token that *should* come next).
- $x_{\le t}$ = all input tokens up to and including position $t$ (what the model can "see").
- $p_\theta(y_t \mid x_{\le t})$ = the probability the model, with parameters $\theta$, gives to the correct next token. It comes from a [[softmax]] over the [[logits]].
- $\log$ = natural logarithm; the minus sign turns "high probability" into "low loss".
- $\mathcal{R}$ = the set of positions we actually score. With padding masking only, $\mathcal{R}$ is "everything except extra padding". With instruction masking, $\mathcal{R}$ also excludes the instruction positions, so it is essentially just the response.
- $|\mathcal{R}|$ = how many positions are in that set. Dividing by it makes the loss an *average* over the scored positions.

### A tiny worked number
The book's own cross-entropy demo (§7.3, book pp. 221–222) shows the mechanism. Two token predictions with logits `[[-1.0, 1.0], [-0.5, 1.5]]` and targets `[0, 1]` give `loss = 1.1269`. Per position, that's $\log(1+e^{2}) \approx 2.1269$ and $\log(1+e^{-2}) \approx 0.1269$, and the mean is $1.1269$.

Now pretend position 0 is an "instruction" token and mask it: targets `[-100, 1]`. Only position 1 is left, so the loss becomes $0.1269$. (This masked number is my own arithmetic from the book's example. The book itself demonstrates the same effect with `[0, 1, -100]`, which gives `1.1269` again, exactly as if the third token did not exist.)

### The code (Exercise 7.2 solution, Appendix C, book pp. 308–309)
Two small changes. First, the dataset remembers how many tokens the instruction-plus-input part has:

```python
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.instruction_lengths = []          # separate list for instruction lengths
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    def __getitem__(self, index):
        return self.instruction_lengths[index], self.encoded_texts[index]
```

Second, the collate function receives `(instruction_length, item)` tuples and adds **one new line** after the padding mask:

```python
targets[:instruction_length-1] = -100   # masks all input and instruction tokens
```

Why `instruction_length - 1`? Because targets are the inputs shifted by one. If the instruction occupies input positions $0,\dots,n-1$, then target positions $0,\dots,n-2$ are the ones whose correct answers are instruction tokens $1,\dots,n-1$. So masking the first $n-1$ targets removes exactly the "predict the instruction" jobs. (This index reasoning is my explanation of the book's `-1`.) Note that the `"\n\n### Response:\n"` header is *not* in `instruction_plus_input`, so it stays scored.

### What the book found
- §7.3 says masking is common and "can help reduce overfitting", but researchers are divided. The book does **not** use masking in the main chapter; it leaves it as **Exercise 7.2** (book pp. 222–223).
- The Exercise 7.2 solution reports the masked model scores **about 4 points worse** with the Ollama Llama 3 judge ([[llm-as-a-judge]]) (Appendix C, book p. 309).
- That matches Shi et al. (2024), "Instruction Tuning With Loss Over Instructions" (arXiv 2405.14394). Appendix B summarizes it: *not* masking can improve performance, especially when instructions are long and outputs are short, or when there are few training examples (book p. 297).

## 🔴 Theory (Expert)
- **Masked maximum likelihood.** Unmasked instruction tuning maximizes the joint likelihood $\log p_\theta(\text{prompt}, \text{response}) = \log p_\theta(\text{prompt}) + \log p_\theta(\text{response}\mid\text{prompt})$ (by the autoregressive chain rule). Instruction masking drops the first term and maximizes only the **conditional** likelihood $\log p_\theta(\text{response}\mid\text{prompt})$. That is the quantity we actually use at inference time, which is the argument *for* masking. See [[maximum-likelihood-estimation]].
- **Why not masking can still win.** The prompt term acts like extra language-modeling signal, a kind of auxiliary loss that keeps the model close to its pretraining distribution. When responses are short, masking throws away most of the tokens in each example, so the gradient comes from very few positions. Shi et al.'s finding (long instructions + short outputs, or small data) fits this view: the dropped term was doing useful [[regularization]] work. (This interpretation is mine, consistent with the book's summary on p. 297.)
- **Normalization subtlety.** PyTorch's mean reduction divides by $|\mathcal{R}|$ summed over the **whole batch**, not per example. So with masking, examples with long responses carry more weight in a batch than examples with short responses. Per-sequence averaging is an alternative design choice. (Outside knowledge about `cross_entropy` behavior, not discussed in the book.)
- **Tokenization boundary.** `len(tokenizer.encode(prefix))` equals the number of prefix tokens inside `tokenizer.encode(prefix + suffix)` only if BPE does not merge across the boundary. Here the suffix begins with `"\n\n"`, so the split is very likely clean, but in general it is an assumption worth checking.
- **End-of-text is kept.** Both padding masking and instruction masking keep exactly one `50256` target, so the model still learns when to stop ([[end-of-text-token]]).

## Connections
- Builds on: [[cross-entropy-loss]], [[ignore-index-loss-masking]], [[collate-function]], [[softmax]], [[instruction-dataset]], [[prompt-template]], [[next-token-prediction]]
- Leads to: [[instruction-fine-tuning]], [[llm-as-a-judge]] (how its effect is measured)
- Related: [[overfitting]], [[regularization]], [[maximum-likelihood-estimation]], [[dynamic-padding]], [[end-of-text-token]]
- People: [[zhengyan-shi]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3 (instruction masking, Figure 7.13, cross-entropy `-100` demo, Exercise 7.2), book pp. 221–223.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, Chapter 7 references (Shi 2024, arXiv 2405.14394), book p. 297.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 7.2 solution (code, ~4-point drop), book pp. 308–309.
