---
title: Instruction Dataset
type: concept
tags: [llm, instruction-fine-tuning, datasets, json, alpaca]
difficulty: beginner
sources: [llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Instruction Dataset

> **One-liner:** An instruction dataset is a big list of "here's a request → here's the ideal answer" examples that teaches a pretrained LLM to follow instructions.

## 🟢 Intuition (Dummy)
A freshly pretrained LLM is like a very well-read parrot: give it the start of a sentence and it
happily continues it. But ask it "Fix the grammar in this text" and it may just ramble on, because
it was only ever trained to *continue* text, not to *obey* requests.

An instruction dataset is the fix. It's a workbook of examples, each with a request and a model
answer:

| Instruction | Desired response |
|---|---|
| Convert 45 kilometers to meters. | 45 kilometers is 45000 meters. |
| Provide a synonym for "bright." | A synonym for "bright" is "radiant." |

Train the model on enough of these and it learns the *pattern*: "when someone asks me something,
I answer it." That's the difference between a text-completer and a personal assistant.

## 🟡 Mechanics (Practitioner)

### Structure of one entry
The book's dataset is a **JSON** file (JavaScript Object Notation — a plain-text format whose
structure mirrors Python dictionaries, readable by both humans and machines). It holds
**1,100** instruction–response pairs and is only **204 KB** (book p. 207). Each entry is a dict
with three keys:

```python
{'instruction': 'Identify the correct spelling of the following word.',
 'input': 'Ocassion',
 'output': "The correct spelling is 'Occasion.'"}          # data[50], book p. 208
```

- `instruction` = the task,
- `input` = optional extra material for the task — it can be the **empty string** `''`:

```python
{'instruction': "What is an antonym of 'complicated'?",
 'input': '',
 'output': "An antonym of 'complicated' is 'simple'."}     # data[999], book p. 208
```

- `output` = the desired response, i.e. the *target* the model must learn to produce.

Because the input–output pairs are given explicitly, this is called **supervised instruction
fine-tuning** (book p. 208) — it's [[supervised-learning]] where the "label" is a whole sentence.

### Loading it (Listing 7.1, book p. 207 — abridged: the book's `else` branch that re-reads an already-downloaded file is omitted)
```python
import json, os, urllib.request

def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):                 # download only once
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)                         # -> list of 1,100 dicts
    return data
```

### Splitting it: train / validation / test (Listing 7.3, book p. 210)
With $N$ = number of entries:

$$n_{\text{train}} = \lfloor 0.85\,N \rfloor,\qquad n_{\text{test}} = \lfloor 0.10\,N \rfloor,\qquad n_{\text{val}} = N - n_{\text{train}} - n_{\text{test}}$$

Symbol by symbol:
- $N$ = `len(data)` = 1100,
- $\lfloor x \rfloor$ = "floor": round down to a whole number (Python's `int()` on a positive number),
- $n_{\text{train}}$, $n_{\text{test}}$, $n_{\text{val}}$ = how many examples go to each set.

Plugging in: $n_{\text{train}} = 935$, $n_{\text{test}} = 110$, $n_{\text{val}} = 1100-935-110 = 55$.

```python
train_portion = int(len(data) * 0.85)   # 935
test_portion  = int(len(data) * 0.1)    # 110
val_portion   = len(data) - train_portion - test_portion   # 55
train_data = data[:train_portion]
test_data  = data[train_portion:train_portion + test_portion]
val_data   = data[train_portion + test_portion:]
```

Note the slice order: train first, then **test**, then validation at the very end. See
[[train-validation-test-split]].

### From entry to training text
An entry is not fed to the model as a dict. It is first turned into one string with a
[[prompt-template]] (the book uses the Alpaca style), then tokenized and batched by a
[[collate-function]]. So the path is: JSON entry → prompt text → [[token-id]]s → padded batch.

### Public instruction datasets
- **Stanford Alpaca** — one of the earliest and most popular openly shared instruction datasets.
  The book gives **52,002** entries in Exercise 7.3 (book p. 233) and "52,000 instruction–response
  pairs" in appendix B (book p. 296) — the round number is just an approximation. That's about
  **50×** the chapter dataset, with mostly longer entries. Exercise 7.3 swaps the URL to
  `https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json`; if you
  run out of memory, drop `batch_size` from 8 → 4, 2, or 1 and/or `allowed_max_length` from
  1,024 → 512 or 256 (book pp. 233, 309–310).
- Appendix B (book pp. 296–297) also lists **LIMA**, **UltraChat** (805,000 pairs),
  **Alpaca GPT4** (52,000 pairs generated with GPT-4 instead of GPT-3.5), and **Magpie**
  (300,000 synthetic pairs from instruction-tuned Llama 3; a Llama 3 base model tuned on them
  performs comparably to the official instruct model).

## 🔴 Theory (Expert)
- **ERM on a new distribution.** Each entry is a pair $(x, y)$ with $x$ = formatted
  instruction(+input) and $y$ = response. The dataset is a finite sample from a task distribution
  $\mathcal{D}$, and fine-tuning minimizes the empirical next-token loss over it
  ([[empirical-risk-minimization]], [[maximum-likelihood-estimation]]): starting from pretrained
  $\theta_0$ rather than random weights — a form of [[transfer-learning]].
- **Contrast with classification fine-tuning.** Chapter 6 uses a *dataset with class labels*
  (spam / not spam) and trains only on the last token's prediction; chapter 7 uses an
  *instruction dataset* and trains on whole sequences (Figure 1.9 / 7.1).
- **Size vs. quality.** 1,100 hand-curated pairs already teach GPT-2 medium the instruction
  format (judge score 50.32, §7.8). Appendix B's Gekhman et al. (2024) reference argues LLMs mainly
  acquire factual knowledge in pretraining; fine-tuning mostly improves how they use it, and new
  facts introduced in fine-tuning are learned slowly and increase hallucination (book pp. 297–298).
  So an instruction dataset teaches *behavior* more than *knowledge*.
- **Split hygiene.** The book slices contiguously without shuffling; this is fine only if the file
  order carries no systematic pattern. A held-out test set is essential, since response quality,
  not loss, is the real target (see [[llm-evaluation]]).

## Connections
- Builds on: [[supervised-learning]], [[train-validation-test-split]], [[pretraining]], [[fine-tuning]]
- Leads to: [[prompt-template]], [[collate-function]], [[pytorch-dataset-and-dataloader]], [[instruction-fine-tuning]]
- Used by: [[instruction-fine-tuning]], [[llm-evaluation]], [[llm-as-a-judge]]
- Related: [[classification-fine-tuning]], [[empirical-risk-minimization]], [[transfer-learning]]
- People: [[sebastian-raschka]] (created the 1,100-entry dataset for the book)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.1, book pp. 205–206 (instruction dataset, Figure 7.2 examples).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2, book pp. 207–211 (JSON format, 1,100 entries, 204 KB, Listings 7.1 and 7.3, 935/55/110 split).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 212–213 (`InstructionDataset` pre-tokenizes entries).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book p. 233 (Exercise 7.3, Alpaca 52,002 entries).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, book pp. 296–298 (public datasets, Gekhman et al.).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book pp. 309–310 (Exercise 7.3 solution).
