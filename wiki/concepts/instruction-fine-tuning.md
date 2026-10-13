---
title: Instruction Fine-Tuning
type: concept
tags: [llm, fine-tuning, instruction-tuning, supervised-learning, cross-entropy, nlp]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Instruction Fine-Tuning

> **One-liner:** Instruction fine-tuning takes a model that can only finish sentences and keeps training it on "instruction → good answer" examples, so it learns to actually do what you ask.

## 🟢 Intuition (Dummy)
Picture someone who has read half the internet but has only ever played one game: "guess the
next word." That's a pretrained LLM (a [[foundation-model]]). Ask it *"Convert this sentence to
passive voice: 'The chef cooks the meal every day.'"* and it doesn't answer you. It just keeps
writing in the same style, maybe repeating your sentence, maybe inventing a new instruction. It
is a [[text-completion]] machine, not an assistant.

Instruction fine-tuning is like a short apprenticeship. You show the model a pile of worked
examples: *here is a request, here is the answer we want*. "Convert 45 kilometers to meters" →
"45 kilometers is 45000 meters." "Give a synonym for 'bright'" → "A synonym for 'bright' is
'radiant'." After seeing enough of these, the model gets the idea: when text looks like a
request, the next words should be the *answer*, and then it should stop.

The neat part is that nothing new is invented. It is the same "guess the next word" game, the
same loss, the same training loop as [[pretraining]]. Only the *data* changes. That's why the
book says most of the real work is preparing the dataset. This is the main trick behind
chatbots and personal assistants such as ChatGPT (book pp. 12, 204).

## 🟡 Mechanics (Practitioner)

### Where it sits in the big picture
The book's roadmap ([[llm-building-roadmap]]) ends in **stage 3**, where the pretrained model
branches two ways (Fig. 1.9, book p. 14; Fig. 7.1, book p. 205):
- step 8: fine-tune on a *dataset with class labels* → a **classifier** ([[classification-fine-tuning]], Ch. 6);
- step 9: fine-tune on an *instruction dataset* → a **personal assistant / chat model** (this page, Ch. 7).

The book also calls this **supervised instruction fine-tuning** (book p. 205), or SFT for short
(the saved file is `gpt2-medium355M-sft.pth`, book p. 238). It is "supervised" because every
example comes with the target answer spelled out.

**Instruction vs. classification fine-tuning** (sidebar "Choosing the right approach", book p. 171):

| | Instruction fine-tuning | Classification fine-tuning |
|---|---|---|
| Teaches | understand instructions and write responses | pick one class from a fixed set |
| Output | free text, any task | only the trained labels ("spam"/"not spam") |
| Data needed | larger datasets | less data |
| Compute | more | less |
| Kind of model | generalist | specialist |

### The pipeline: 3 stages, 9 steps (Fig. 7.3, book p. 206)
1. **Preparing the dataset:** (1) download and format, (2) batch, (3) data loaders.
2. **Fine-tuning the LLM:** (4) load a pretrained LLM, (5) instruction fine-tune, (6) inspect the loss.
3. **Evaluating the LLM:** (7) extract responses, (8) qualitative evaluation, (9) score the responses.

### Step 1: the instruction dataset (§7.2, book pp. 207–211)
The book's [[instruction-dataset]] is a 204 KB JSON file with **1,100** entries. Each entry is a
dict with three keys, `instruction`, `input` (optional, may be `''`), and `output` (book pp. 207–208):

```python
data[50]  = {'instruction': 'Identify the correct spelling of the following word.',
             'input': 'Ocassion', 'output': "The correct spelling is 'Occasion.'"}
data[999] = {'instruction': "What is an antonym of 'complicated'?",
             'input': '', 'output': "An antonym of 'complicated' is 'simple'."}
```

Each entry is turned into one string with a [[prompt-template]]. The book uses the **Alpaca**
style (Listing 7.2, book p. 209). The `### Input:` block is skipped when the input is empty:

```python
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text
# the desired response is appended as: f"\n\n### Response:\n{entry['output']}"
```

Another style is **Phi-3** (Microsoft), which uses the special tokens `<|user|>` and
`<|assistant|>` (Fig. 7.4, book p. 208). Exercise 7.1 swaps it in: training is about 17% faster
(shorter prompts) and the judge score stays near 50 (appendix C, book pp. 307–308).

**Split** (Listing 7.3, book p. 210), with $N = 1100$:
$$n_{\text{train}} = \lfloor 0.85\,N\rfloor = 935,\qquad n_{\text{test}} = \lfloor 0.10\,N\rfloor = 110,\qquad n_{\text{val}} = N - n_{\text{train}} - n_{\text{test}} = 55$$
Reading it: $N$ = number of examples; $\lfloor\cdot\rfloor$ = "round down" (Python's `int()`);
the training set gets 85%, the test set 10%, validation gets whatever is left (5%). See
[[train-validation-test-split]].

### Step 2: batching with a custom collate function (§7.3, book pp. 211–223)
Instruction examples have different lengths, so the book writes its own
[[collate-function]] (the function a `DataLoader` uses to glue samples into one batch). It does
five sub-steps (Fig. 7.6, book p. 212):

1. **2.1** format with the prompt template;
2. **2.2** tokenize (every prompt starts with the same IDs `21106, 318, 281, 12064, 326, 8477, 257, 4876, 13`, book p. 213);
3. **2.3** pad with the [[end-of-text-token]] `<|endoftext|>` = ID **50256**, but only up to the longest
   example *in this batch* ([[dynamic-padding]], book p. 214);
4. **2.4** make targets = inputs shifted one place to the left, the usual [[next-token-prediction]] setup;
5. **2.5** replace every padding token in the targets **except the first one** with **−100**.

The shift in step 2.4, for a padded sequence $p = (p_0, p_1, \dots, p_L)$:
$$x_t = p_t,\qquad y_t = p_{t+1},\qquad t = 0,\dots,L-1$$
Symbol by symbol: $p_t$ = the token ID at position $t$ of the padded list; $x_t$ = the input the
model sees at position $t$; $y_t$ = the target it must predict there, which is simply the *next*
token; $L$ = the padded length. In code: `inputs = padded[:-1]`, `targets = padded[1:]`.

**The book's toy batch** (book pp. 215–221). Three token lists `[0,1,2,3,4]`, `[5,6]`, `[7,8,9]` become:

```
inputs:  [[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]]
targets: [[1, 2, 3, 4, 50256], [6, 50256, -100, -100, -100], [8, 9, 50256, -100, -100]]
```

Why keep exactly one 50256 in the targets? So the model learns to *say* end-of-text when the
answer is finished. That token is the "I'm done" signal used at generation time (book p. 218).

The final version (Listing 7.5, book p. 220):

```python
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100,
                      allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index     # all pads but the first -> -100
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs); targets_lst.append(targets)
    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)
```

### The loss: cross-entropy that skips the −100 positions
Instruction fine-tuning uses the very same [[cross-entropy-loss]] as pretraining. PyTorch's
`cross_entropy` has `ignore_index=-100` by default, so those positions are simply left out of the
average ([[ignore-index-loss-masking]], book pp. 221–222):
$$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} \log \frac{e^{z_{i,y_i}}}{\sum_{k=1}^{V} e^{z_{i,k}}},\qquad \mathcal{I} = \{\, i : y_i \neq -100 \,\}$$
Symbol by symbol:
- $\theta$ = all the model's weights;
- $i$ = one token position in the batch; $y_i$ = the correct next-token ID there;
- $z_{i,k}$ = the [[logits]] (raw score) the model gives token $k$ at position $i$;
- $V$ = vocabulary size (50,257 for GPT-2);
- the fraction is the [[softmax]] probability of the correct token, and $-\log$ of it is large when that probability is small;
- $\mathcal{I}$ = the set of positions whose target is **not** −100; $|\mathcal{I}|$ = how many there are, so we average only over real tokens.

**Worked example** (book pp. 221–222). Two positions, two "vocabulary" classes:
`logits_1 = [[-1.0, 1.0], [-0.5, 1.5]]`, `targets_1 = [0, 1]`.
- position 1: $-\log\frac{e^{-1}}{e^{-1}+e^{1}} = \log(1+e^{2}) \approx 2.1269$
- position 2: $-\log\frac{e^{1.5}}{e^{-0.5}+e^{1.5}} = \log(1+e^{-2}) \approx 0.1269$
- mean $= 1.1269$ ✔ (PyTorch prints `tensor(1.1269)`).

Add a third prediction `[-0.5, 1.5]` with target `1`: the loss becomes $(2.1269+0.1269+0.1269)/3 = 0.7936$.
Set that third target to `-100` instead and the loss is back to **1.1269**: `loss_1 == loss_3` is `True`.

**Optional instruction masking.** You can also set the targets of the whole instruction part
to −100, so only response tokens count ([[instruction-masking]], Fig. 7.13, book p. 222).
Researchers disagree on whether this helps. Shi et al. (2024), "Instruction Tuning With Loss
Over Instructions", found that **not** masking can be better (book p. 223; appendix B p. 297).
The book doesn't mask. In Exercise 7.2 masking scored about **4 points worse** with the judge
(appendix C, book pp. 308–309). The one-line change is `targets[:instruction_length-1] = -100`.

### Step 3: data loaders (§7.4, book pp. 223–226)
`functools.partial` pre-fills the device and `allowed_max_length=1024` (GPT-2's
[[context-length]]). The loaders use `batch_size=8`, `torch.manual_seed(123)`, and
`shuffle=True, drop_last=True` for training only. Because padding is per batch, batch shapes
change: `[8, 61]`, `[8, 76]`, `[8, 73]`, …, `[8, 69]` (book pp. 225–226). See
[[pytorch-dataset-and-dataloader]].

### Steps 4–6: load GPT-2 medium and train (§7.5–7.6, book pp. 226–233)
The book switches from GPT-2 small (124M) to **GPT-2 medium, 355 M parameters**, because the
small model lacks the [[model-capacity]] to learn good instruction following (book pp. 226–227).
Config ([[gpt-model-configuration]]): `vocab_size=50257`, `context_length=1024`, `drop_rate=0.0`,
`qkv_bias=True`, `emb_dim=1024`, `n_layers=24`, `n_heads=16`. The weights file is 1.42 GB.

**Baseline first** ([[baseline-evaluation]], book pp. 228–229). On `val_data[0]` ("Convert the
active sentence to passive: 'The chef cooks the meal every day.'") the pretrained model just
echoes the sentence and starts a new `### Instruction:`. It cannot follow instructions yet.

**Training** reuses `train_model_simple` from Ch. 5 ([[llm-training-loop]], Listing 7.8, book p. 231):

```python
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 2
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer)
```

| What | Value (book pp. 230–232) |
|---|---|
| Loss before training (5 batches) | train 3.826, val 3.762 |
| Step 0 | train 2.637, val 2.626 |
| End of epoch 1 (step 115) | train 0.520, val 0.665 |
| End of epoch 2 (step 230) | train 0.300, val 0.657 |
| Run time, 2 epochs | book's run prints 0.87 min; table (p. 231): 0.86 min A100, 1.83 min L4, 15.78 min M3 MacBook Air CPU |
| Sample after epoch 1 | "The meal is **prepared** every day by the chef." |
| Sample after epoch 2 | "The meal is **cooked** every day by the chef." ✔ |

The optimizer is [[adamw-optimizer]] with learning rate $\eta = 5\times10^{-5}$ and
[[weight-decay]] $\lambda = 0.1$. More epochs "may even be counterproductive" because of
[[overfitting]] (book p. 232). You can already see it: validation loss flattens near 0.66 while
training loss keeps falling. Rough meaning of the starting number (our own arithmetic, not the
book's): a loss of 3.8 is a [[perplexity]] of $e^{3.8}\approx 45$. The model is about as unsure
as if it were picking among 45 equally likely tokens.

### Steps 7–9: extract responses and score them (§7.7–7.8, book pp. 233–247)
The `generate` function returns prompt + answer together, so the answer is cut out by
[[response-extraction]] (book p. 235):

```python
response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
```

Generation uses `max_new_tokens=256` and `eos_id=50256`, so it stops at end-of-text. Three
test answers (book pp. 235–236): simile ✔ ("as fast as a bullet"), clouds ✘ ("cumulus" instead
of "cumulonimbus"), Jane Austen ✔.

There is no simple "accuracy" for free text, so the book uses [[llm-as-a-judge]]. Llama 3 8B
(instruction-tuned, run locally with Ollama, [[local-llm-inference-ollama]]) scores every test answer
from 0 to 100 against the reference answer:
$$\bar s = \frac{1}{N}\sum_{i=1}^{N} s_i,\qquad s_i\in\{0,1,\dots,100\}$$
Here $s_i$ = the judge's integer score for test answer $i$, $N$ = the number of answers that
parsed (110), and $\bar s$ = their plain average. The fine-tuned GPT-2 medium gets
**$\bar s = 50.32$** (book p. 246). For comparison, with the same method: Llama 3 8B *base*
gets 58.51 and Llama 3 8B *instruct* gets 82.6 (book p. 247).

## 🔴 Theory (Expert)
- **It is conditional maximum likelihood.** The loss above is the negative log-likelihood of the
  target tokens, so minimizing it is [[maximum-likelihood-estimation]] on a new data
  distribution (instruction–response pairs). The objective is the same as pretraining; only the
  empirical distribution changes ([[empirical-risk-minimization]]). This is why the book can reuse
  `calc_loss_loader` and `train_model_simple` unchanged (book pp. 229–230).
- **What masking really changes** (our derivation, not in the book). By the chain rule of
  probability, $\log p(\text{instr}, \text{resp}) = \log p(\text{instr}) + \log p(\text{resp}\mid\text{instr})$.
  The unmasked loss fits the *joint* sequence. Instruction masking drops the first term and fits
  only the *conditional* $p(\text{resp}\mid\text{instr})$. Shi et al. (2024) report the extra
  instruction term helps, especially with long instructions and short outputs, or few training
  examples (appendix B p. 297). This acts like a regularizer against [[overfitting]] on short targets.
- **Why −100 is exact, not approximate.** Masked positions are removed from both the sum and the
  count $|\mathcal{I}|$. So adding padding never changes the loss value or its gradient. That's
  the `loss_1 == loss_3` check. A plain zero-weighting that still divided by the full count would
  bias the loss toward shorter sequences.
- **Why keep one end-of-text target.** Generation stops when the model emits `<|endoftext|>`
  (`eos_id=50256`). If every pad were masked, the model would never be rewarded for emitting
  EOS. It would ramble on, and the epoch samples show exactly that: the model writes a new fake
  instruction after `<|endoftext|>` (book p. 232).
- **Capacity matters.** 124M parameters were judged too small (book pp. 226–227). The improvement
  list on book p. 247 includes a larger pretrained model, more or more diverse data, other prompt
  formats, and tuning the learning rate, batch size and epochs. The reference numbers (GPT-2
  medium 50.32 < Llama 3 8B base 58.51 < Llama 3 8B instruct 82.6) suggest both scale and
  instruction tuning help. That comparison is our reading; the book does not draw it.
- **Fine-tuning teaches format, not facts.** Appendix B cites Gekhman et al. (2024): LLMs pick up
  factual knowledge mainly in pretraining, and fine-tuning mostly improves how they use it. New
  facts in fine-tuning are learned slowly and *increase* hallucination (book pp. 297–298).
- **Loss is not quality.** Training loss tells you the model fits the data, but "the most crucial
  aspect is its performance in terms of response quality and correctness" (book p. 233). LLM
  evaluation falls into three families: multiple-choice benchmarks (MMLU), human preference
  (LMSYS Chatbot Arena), and LLM judges (AlpacaEval) (book p. 236). The judge is a noisy
  [[random-variable]]. Ollama isn't fully deterministic, so the book suggests averaging several
  runs (book p. 247). If $R$ runs are independent, the variance of the mean drops by a factor of $R$
  ([[covariance-and-variance]]).
- **Bigger data.** Exercise 7.3 swaps in the original Stanford Alpaca dataset: 52,000 entries
  (50x more), with longer entries, so a GPU is recommended. On out-of-memory errors, lower the
  batch size from 8 to 4, 2 or 1, or `allowed_max_length` from 1024 to 512 or 256 (appendix C,
  book pp. 309–310).
- **Parameter-efficient variant.** With [[low-rank-adaptation]] ($W + AB$, rank 16, alpha 16) only
  the small $A,B$ matrices train. Exercise 7.4: 1.30 min vs. 1.80 min on an L4 GPU (about 28%
  faster), with the same judge score of about 50 (appendix C, book p. 310). See
  [[parameter-efficient-fine-tuning]].
- **What comes after.** [[preference-fine-tuning]] (e.g. [[direct-preference-optimization]], DPO) is
  an optional next step that aligns outputs with human preferences (book p. 247). The original
  ChatGPT was GPT-3 fine-tuned on a large instruction dataset using OpenAI's InstructGPT method
  (Ouyang et al. 2022, RLHF; book p. 12, appendix B p. 290).
- **Book quirks.** §7.7 says rating "all 1,100 responses" by hand would be a lot of work, but the
  test set has 110 entries (book pp. 237–238). The Figure 7.12 caption says "Step 2.4" for what
  is step 2.5 (book p. 219). Appendix C prints the Phi-3 tags as `<user>` without pipes, while
  the code uses `<|user|>` (book p. 307).

## Connections
- Builds on: [[pretraining]], [[fine-tuning]], [[next-token-prediction]], [[cross-entropy-loss]], [[softmax]], [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[supervised-learning]], [[transfer-learning]], [[gradient-descent]], [[stochastic-gradient-descent]], [[backpropagation]]
- Data pieces: [[instruction-dataset]], [[prompt-template]], [[collate-function]], [[dynamic-padding]], [[end-of-text-token]], [[ignore-index-loss-masking]], [[instruction-masking]], [[train-validation-test-split]]
- Training pieces: [[gpt-2]], [[gpt-model-configuration]], [[model-capacity]], [[adamw-optimizer]], [[weight-decay]], [[llm-training-loop]], [[training-and-validation-loss]], [[overfitting]]
- Evaluation: [[baseline-evaluation]], [[response-extraction]], [[llm-evaluation]], [[llm-as-a-judge]], [[local-llm-inference-ollama]], [[expected-value-and-mean]]
- Leads to: [[preference-fine-tuning]], [[direct-preference-optimization]], [[low-rank-adaptation]], [[parameter-efficient-fine-tuning]]
- Contrast: [[classification-fine-tuning]], [[text-completion]], [[few-shot-learning]], [[zero-shot-learning]]
- Used by: [[gpt-3]] (ChatGPT), [[large-language-model]], [[llm-applications]]
- People: [[sebastian-raschka]], [[long-ouyang]] (InstructGPT), [[zhengyan-shi]] (loss over instructions), [[edward-j-hu]] (LoRA)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 6–7 (two categories of fine-tuning; instruction–answer pairs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book p. 12 (ChatGPT = GPT-3 fine-tuned on an instruction dataset, InstructGPT).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7, book pp. 14–15 (stage 3, step 9, Fig. 1.9).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book pp. 170–171 (definition, Fig. 6.2, "Choosing the right approach"); §6.4, book p. 183 (base model fails a spam instruction).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.1, book pp. 204–206 (Figs. 7.1–7.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.2, book pp. 207–211 (dataset, Alpaca/Phi-3 styles, Listings 7.1–7.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book pp. 211–223 (collate function, −100, cross-entropy demo, Listing 7.5, instruction masking).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.4, book pp. 223–226 (data loaders, Listing 7.6).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book pp. 226–229 (GPT-2 medium, Listing 7.7, baseline).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.6, book pp. 229–233 (Listing 7.8, training log, Fig. 7.17).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.7, book pp. 233–238 (response extraction, Listing 7.9, evaluation families).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, book pp. 238–247 (Ollama judge, Listings 7.10–7.11, score 50.32, references 58.51/82.6).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.9 and Ch. 7 summary, book pp. 247–249 (preference fine-tuning, DPO).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix E §E.1, book p. 322 (LoRA also applies to instruction fine-tuning).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 290, 296–298 (InstructGPT; Alpaca, LIMA, UltraChat, Magpie; Shi et al. 2024; Gekhman et al. 2024).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 307–310 (Exercises 7.1–7.4 solutions).
