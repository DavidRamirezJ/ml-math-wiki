---
title: Pretraining
type: concept
tags: [llm, training, self-supervised-learning, cross-entropy, deep-learning]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appA-introduction-to-pytorch, llm-appD-training-loop-bells-and-whistles]
created: 2026-10-13
updated: 2026-10-13
---

# Pretraining

> **One-liner:** Pretraining is the first, huge training phase where an LLM reads mountains of ordinary text and learns to guess the next word — which turns random weights into a general-purpose "language brain."

## 🟢 Intuition (Dummy)
Imagine teaching someone a language by handing them a whole library and a single
rule: "cover the next word with your hand, guess it, then peek." Do that a few
hundred billion times and you'd absorb grammar, facts, style, even a bit of
reasoning — without anyone ever writing an answer key. The text *is* the answer key.

That's pretraining. The "pre" means it comes *first*: the model is trained on a
**large, diverse dataset** to build a broad understanding of language (book p. 6).
The result is a [[foundation-model]] that can already complete sentences, and that you
can later specialise with [[fine-tuning]].

Two honest caveats from the book:
- It is **expensive**. GPT-3's pretraining is estimated at about **\$4.6 million** in
  cloud credits (book p. 11); training a GPT-like model from scratch costs "thousands
  to millions of dollars" (book p. 15).
- So the book pretrains a small model on one short story to learn *how*, then loads
  OpenAI's already-pretrained GPT-2 weights to skip the bill (book pp. 12, 160–161).
  The book promises that "all computations are executable on consumer hardware" (book p. 12).

## 🟡 Mechanics (Practitioner)

### 1. The data: raw, unlabeled text
- **Raw text** = regular text with no labels (some filtering is fine, e.g. dropping
  formatting characters or unknown languages) (book p. 6). Figure 1.3 says "trillions
  of words" from internet texts, books, Wikipedia and research articles.
- GPT-3's mix (Table 1.1, book p. 11): CommonCrawl 410 B tokens (60% of training
  data), WebText2 19 B (22%), Books1 12 B (8%), Books2 55 B (8%), Wikipedia 3 B (3%).
  That's 499 B tokens available, but only **300 B** were used in training. See
  [[pretraining-datasets]].
- The book's educational corpus: *The Verdict* — **20,479 characters, 5,145 tokens**,
  split 90% train / 10% validation (book p. 141).

### 2. The labels come for free (self-supervision)
Each input is paired with the **same text shifted by one token** — that's the target
([[sliding-window-sampling]], book p. 36):
```
x: [290, 4920, 2241, 287]
y: [4920, 2241, 287, 257]      # " and established himself in" -> " established himself in a"
```
No human writes labels; the model "generates its own labels from the input data"
(book p. 6). See [[self-supervised-learning]] and [[next-token-prediction]].

### 3. The objective: cross-entropy on the next token
Pretraining **changes the weights to minimise the training loss** (Ch. 5 summary,
book p. 168). The loss is the average negative log-probability of the correct next
tokens:
$$\mathcal L(\theta)=-\frac{1}{N}\sum_{n=1}^{N}\log p_\theta\big(y_n\mid\text{context}_n\big)$$
Symbol by symbol:
- $\theta$ = all the model's [[model-parameters]] (weights and biases);
- $N$ = how many next-token predictions are in the batch (batch size × tokens);
- $y_n$ = the true next token for prediction $n$; "context$_n$" = the tokens before it;
- $p_\theta(y_n\mid\text{context}_n)$ = the [[softmax]] probability the model gives to the right answer;
- $\log$ = natural log (turns tiny probabilities into manageable negative numbers);
- the minus sign flips "maximise the log-probability" into "minimise a loss".

**The book's worked numbers (Figure 5.7, book pp. 137–139).** Two inputs, "every
effort moves" and "I really like", targets " effort moves you" and " really like
chocolate" → 6 target probabilities from the untrained model:
`[7.4541e-05, 3.1061e-05, 1.1563e-05, ...]`
→ log probs `[-9.5042, -10.3796, -11.3677, -11.4798, -9.7764, -12.2561]`
→ average **−10.7940** → negate → loss **10.7940**. PyTorch does all of it in one line:
```python
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
# tensor(10.7940)
```
[[perplexity]] $=e^{\mathcal L}=e^{10.7940}\approx$ **48,725** — the untrained model is
about as unsure as picking among 48,725 of the 50,257 tokens (book p. 140).

### 4. The training loop (Listing 5.3, book p. 147)
```python
for epoch in range(num_epochs):
    model.train()
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()                 # reset old gradients
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()                       # backpropagation
        optimizer.step()                      # update weights
```
Each `optimizer.step()` is a [[gradient-descent]]-style update; the plain form is
$\theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal L$, where $\eta$ is the
[[learning-rate]] and $\nabla_\theta\mathcal L$ is the [[gradient]] from
[[backpropagation]]. The book uses [[adamw-optimizer]]:
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
```
with `GPT_CONFIG_124M` (vocab 50,257, **context 256**, emb 768, 12 heads, 12 layers,
dropout 0.1) — about **5 minutes on a MacBook Air** (book pp. 130, 149).

**What happens (book pp. 149–151):** training loss **9.781 → 0.391**, validation loss
**9.933 → 6.452**. Samples go from `Every effort moves you,,,,,,,,,,,,.` (epoch 1) to a
word-for-word copy of the story (epoch 10). The widening train/validation gap is
[[overfitting]]: a tiny dataset seen for many epochs. Real pretraining uses a vastly
larger dataset for roughly **one epoch** (book p. 151).

### 5. Saving, resuming, and skipping it
- Save with `torch.save(model.state_dict(), "model.pth")`; to **continue
  pretraining**, also save the optimizer state, since AdamW keeps per-weight history
  (book pp. 159–160). See [[model-checkpointing]].
- Or load OpenAI's [[pretrained-weights]] for GPT-2 (124M/355M/774M/1558M) into the
  same `GPTModel` class (book pp. 161–166).

## 🔴 Theory (Expert)
- **Pretraining = maximum likelihood.** Minimising $\mathcal L$ is maximising
  $\sum_n\log p_\theta(y_n\mid\cdot)$, i.e. [[maximum-likelihood-estimation]] of the
  autoregressive model $p_\theta(x_{1:T})=\prod_t p_\theta(x_t\mid x_{<t})$. In
  population form, $\mathbb E_{x\sim p_{\text{data}}}[-\log p_\theta(x)]=H(p_{\text{data}})+
  D_{\mathrm{KL}}(p_{\text{data}}\Vert p_\theta)$, so the loss bottoms out at the data's
  entropy, not at 0 (background, not in the book).
- **ERM view.** It is [[empirical-risk-minimization]] with the log-loss over the
  sampled corpus. Ch. 5's train/validation gap is the classic estimation-error story;
  larger, more diverse data lowers it (§1.5, book p. 11).
- **Uniform baseline.** A model that spreads probability evenly over $V$ tokens has
  loss $\ln V=\ln 50257\approx10.825$ and perplexity $V$. The untrained model's 10.79
  (§5.1) and 10.988 (whole-dataset, book p. 145) sit right at that level — random
  init is roughly "knows nothing."
- **Data mixture.** Table 1.1's proportions are sampling weights, not shares of size.
  Derived passes per subset $e_d=p_d N_{\text{train}}/N_d$: Wikipedia
  $0.03\cdot300/3=3.0$, WebText2 ≈3.5, CommonCrawl ≈0.44 — high-quality sources are
  upweighted (wiki derivation from the book's numbers).
- **Why it transfers.** Next-token prediction forces the network to model syntax,
  semantics and context — and some general knowledge (book p. 11). Later work cited
  in appendix B (Gekhman et al. 2024, book pp. 297–298) finds that LLMs acquire
  factual knowledge mainly during pretraining; fine-tuning mostly teaches them to use
  it.
- **Same machine in fine-tuning.** Instruction fine-tuning reuses this exact loss and
  loop on (instruction, response) text (§7.6); classification fine-tuning keeps the
  cross-entropy but over 2 classes (Ch. 6 summary, book p. 203).
- **Bells and whistles.** Appendix D adds [[learning-rate-warmup]],
  [[cosine-decay]] and [[gradient-clipping]]; appendix B points to Pythia/OLMo recipes,
  continued pretraining (Ibrahim et al. 2024) and [[galore]] (book pp. 294–295).

## Connections
- Builds on: [[next-token-prediction]], [[self-supervised-learning]], [[cross-entropy-loss]], [[maximum-likelihood-estimation]], [[loss-function]], [[empirical-risk-minimization]], [[backpropagation]], [[gradient-descent]], [[stochastic-gradient-descent]], [[tokenization]], [[sliding-window-sampling]]
- Leads to: [[foundation-model]], [[fine-tuning]], [[instruction-fine-tuning]], [[classification-fine-tuning]], [[transfer-learning]]
- Uses: [[llm-training-loop]], [[adamw-optimizer]], [[training-and-validation-loss]], [[perplexity]], [[pretraining-datasets]], [[model-checkpointing]], [[pretrained-weights]]
- Pitfalls: [[overfitting]], [[overfitting-and-underfitting]]
- Related: [[gpt-3]], [[gpt-2]], [[large-language-model]], [[autoregressive-text-generation]], [[llm-building-roadmap]]
- People: [[alec-radford]], [[tom-brown]], [[sebastian-raschka]]

> **From [[llm-appD-training-loop-bells-and-whistles]]:** Appendix D upgrades the Ch. 5 pretraining loop with [[learning-rate-warmup]], [[cosine-decay]] and [[gradient-clipping]] (book pp. 313–321). The book's references (Ch. 5 group, book p. 294) point to Ibrahim et al. (2024), which shows the same warmup + cosine schedule also works for continued pretraining.

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book pp. 5–7 (definition, raw text, two-stage approach, Figure 1.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book pp. 10–12 (Table 1.1, \$4.6 M cost, open weights); §1.6, book p. 12 (next-word task); §1.7, book pp. 14–16 (stage 2, cost, summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book pp. 17–18; §2.6, book pp. 35–36 (input–target shift).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book p. 127 (untrained model ⇒ gibberish).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 128–146 (loss, perplexity, dataset); §5.2, book pp. 146–151 (Listing 5.3, AdamW, losses); §5.4, book pp. 159–160; §5.5, book pp. 160–168 (OpenAI weights, Ch. 5 summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book p. 169; §7.1, book p. 205; §7.9, book p. 248.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1, book p. 254 (labels derived from text); appendix B, book pp. 293–298.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix D, book pp. 313–321; appendix B, book p. 294 (Ibrahim et al. 2024).
