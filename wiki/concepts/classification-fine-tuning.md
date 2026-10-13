---
title: Classification Fine-Tuning
type: concept
tags: [llm, fine-tuning, text-classification, supervised-learning, cross-entropy, spam]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Classification Fine-Tuning

> **One-liner:** Classification fine-tuning swaps a pretrained LLM's "next word" output for a tiny "which class?" output (e.g. spam / not spam) and trains it on labeled examples.

## 🟢 Intuition (Dummy)
A pretrained LLM is a huge machine with one exit door. Out of that door comes a vote over
**50,257** possible next tokens. That's great for writing text, but suppose you only want to
know: *is this text message spam?*

Classification fine-tuning keeps the whole machine. Everything it learned about language stays.
You just brick up the old exit and put in a new, tiny one with **two** doors: "not spam" and
"spam." Then you show it a few hundred labeled messages and let it learn which door to open.

The result is a **specialist**. It is very good at its one job: the book's spam detector gets
about 96% right on messages it never saw. But it can *only* say "spam" or "not spam." Ask it
anything else and it has no words for it. That's the trade-off against
[[instruction-fine-tuning]], which builds a generalist that can answer anything but needs more
data and compute (book p. 171).

You know this pattern from classic machine learning: spotting plant species in photos, sorting
news into sports/politics/tech, telling benign tumors from malignant ones (book p. 171). It is
plain [[supervised-learning]], just starting from a pretrained LLM instead of from scratch
([[transfer-learning]]).

## 🟡 Mechanics (Practitioner)

### The 10-step plan (Fig. 6.4, book p. 172)
- **Stage 1, dataset:** (1) download, (2) preprocess, (3) create data loaders.
- **Stage 2, model setup:** (4) initialize the model, (5) load pretrained weights, (6) modify it for fine-tuning, (7) write evaluation utilities.
- **Stage 3, fine-tune and use:** (8) fine-tune, (9) evaluate, (10) use on new data.

In the book-wide roadmap this is **stage 3, step 8**: foundation model + *dataset with class
labels* → **classifier** (Fig. 6.1, book p. 170).

### Steps 1–2: the SMS Spam Collection (§6.2, book pp. 172–175)
The data is the UCI **SMS Spam Collection** (dataset 228), a tab-separated file of phone text
messages labeled `ham` (not spam) or `spam`. The same recipe works for email spam (book p. 172).

- Raw counts: **5,572** messages: `ham 4825`, `spam 747` (book pp. 173–174). That's a strong
  [[class-imbalance]]: only $747/5572 \approx 13.4\%$ is spam. A lazy model that always says
  "ham" would already be $4825/5572 \approx 86.6\%$ "accurate" (our arithmetic).
- **Undersampling:** randomly keep only 747 ham messages so both classes have 747 (Listing 6.2,
  `random_state=123`). The book does this for simplicity and speed (book p. 174). The balanced
  set has $2 \times 747 = 1494$ rows.
- **[[label-encoding]]:** `{"ham": 0, "spam": 1}`. It's like token IDs, but with a "vocabulary" of
  just two IDs instead of 50,000+ (book p. 174).
- **Split** 70 / 10 / 20 after shuffling (Listing 6.3, book p. 175):
$$t = \lfloor 0.7\,N\rfloor,\qquad v = t + \lfloor 0.1\,N\rfloor$$
  Reading it: $N$ = number of rows (1494), $\lfloor\cdot\rfloor$ = round down, rows $[0,t)$
  are training, $[t,v)$ validation, $[v,N)$ test. That gives **1045 / 149 / 300** examples
  (derived; it matches the batch counts below). See [[train-validation-test-split]].

### Step 3: padding and data loaders (§6.3, book pp. 175–181)
Messages have different lengths, but a batch must be a rectangle. You could **truncate** to the
shortest (cheap, but it throws away information) or **pad** to the longest. The book pads
([[padding-and-truncation]], book pp. 175–176). The pad token is `<|endoftext|>`, GPT-2 token
ID **50256** ([[end-of-text-token]]).

For a message of $\ell$ token IDs and a target length $L$:
$$\tilde{x} = (x_1,\dots,x_{\ell},\underbrace{50256,\dots,50256}_{L-\ell\ \text{times}})$$
Here $x_1,\dots,x_\ell$ = the message's token IDs, $L$ = the length every sequence is padded
to, and the brace says "repeat the pad ID $L-\ell$ times."

`SpamDataset` (Listing 6.4, book p. 177) tokenizes every message, finds the longest training
sequence (**120 tokens**), and pads everything to it. Validation and test are padded or truncated
to that same 120. The model could take up to 1,024 tokens, its [[context-length]] (book p. 178).
With `batch_size=8` (Listing 6.5, book p. 180):
- one batch = inputs `torch.Size([8, 120])` + labels `torch.Size([8])`;
- **130** training batches ($\lfloor 1045/8\rfloor$, `drop_last=True`), **19** validation, **38** test (book p. 181).

The **targets are class labels now, not next tokens** (book p. 179).

### Steps 4–6: load GPT-2, swap the head, freeze most layers (§6.4–6.5, book pp. 181–190)
Load **GPT-2 small (124M)**: `emb_dim=768`, `n_layers=12`, `n_heads=12`, `vocab_size=50257`,
`context_length=1024`, `drop_rate=0.0`, `qkv_bias=True` (Listing 6.6, book p. 182). Sanity
check: "Every effort moves you" → "…forward. The first step is to understand the importance of
your work". So the weights loaded fine. But prompt it with *"Is the following text 'spam'?
Answer with 'yes' or 'no': …"* and it just repeats the prompt. A base model can't follow
instructions (book p. 183).

**The new [[classification-head]].** The old `out_head` is a [[linear-layer]] from 768 to 50,257
with no bias. The new one maps 768 → 2 (Listing 6.7, book p. 186):
$$\mathbf{z} = \mathbf{h}\,W_{\text{cls}} + \mathbf{b},\qquad W_{\text{cls}}\in\mathbb{R}^{768\times 2},\ \mathbf{b}\in\mathbb{R}^{2}$$
Symbol by symbol:
- $\mathbf{h}\in\mathbb{R}^{768}$ = the final hidden [[vector]] of one token (after the final LayerNorm);
- $W_{\text{cls}}$ = a [[matrix]] with 768 rows and 2 columns, so it turns 768 numbers into 2;
- $\mathbf{b}$ = a [[bias-vector]] of 2 numbers, added at the end;
- $\mathbf{z} = (z_0, z_1)$ = the two class scores ([[logits]]): $z_0$ for "not spam", $z_1$ for "spam".

This is an affine [[linear-mapping]]. It has $768\cdot 2 + 2 = 1{,}538$ parameters, against
$768 \cdot 50{,}257 = 38{,}597{,}376$ in the old head (our arithmetic).

```python
for param in model.parameters():             # 1. freeze everything
    param.requires_grad = False
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(            # 2. new head (trainable by default)
    in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
for param in model.trf_blocks[-1].parameters():   # 3. unfreeze last block
    param.requires_grad = True
for param in model.final_norm.parameters():       #    and the final LayerNorm
    param.requires_grad = True
```

So only the head, the final LayerNorm and the 12th transformer block learn. The other 11
blocks and the embeddings stay frozen ([[layer-freezing]], Fig. 6.10, book p. 187). Why this
works: lower layers capture general language structure, and the last layers hold more
task-specific features (sidebar, book p. 185).

**Why two output nodes, not one?** One node would work for a yes/no task, but then the loss has
to change. Using "number of nodes = number of classes" works for any $K$, e.g. 3 nodes for
Technology/Sports/Politics (sidebar, book p. 183).

**Which output row to use: the last token.** Feed "Do you have time" (IDs `[5211, 345, 423, 640]`,
shape `[1, 4]`) and you get shape `[1, 4, 2]`, one pair of logits per token (book p. 188):

```
tensor([[[-1.5854,  0.9904],
         [-3.7235,  7.4548],
         [-2.2661,  6.6049],
         [-3.5983,  3.9902]]])      # outputs[:, -1, :] -> [[-3.5983, 3.9902]]
```

The book keeps only the **last row** ([[last-token-pooling]]). Because of the
[[causal-attention-mask]], each token sees only itself and earlier tokens. Only the last token
has seen the *whole* message (Fig. 6.12, book p. 190):

| query \ key | Do | you | have | time |
|---|---|---|---|---|
| Do | 1.0 | | | |
| you | 0.55 | 0.45 | | |
| have | 0.38 | 0.30 | 0.32 | |
| time | 0.27 | 0.24 | 0.24 | 0.25 |

### Step 7: from logits to a label, plus accuracy and loss (§6.6, book pp. 190–194)
**Prediction** ([[softmax]] then [[argmax]], Fig. 6.14):
$$p_k = \frac{e^{z_k}}{\sum_{j=0}^{1} e^{z_j}},\qquad \hat y = \arg\max_k p_k = \arg\max_k z_k$$
- $z_k$ = logit of class $k$ (0 = not spam, 1 = spam); $e^{z_k}$ makes it positive;
- dividing by the sum makes the two $p_k$ add up to 1, so they are probabilities;
- $\hat y$ ("y-hat") = the predicted class, the index with the biggest value.
- The second "=" holds because softmax is a [[monotonic-function]]: it never changes which entry
  is biggest. So you can skip softmax for prediction (book p. 192). For `[-3.5983, 3.9902]` the
  argmax is **1** ("spam").

**Accuracy** ([[classification-accuracy]], Listing 6.8):
$$\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\big[\hat y^{(i)} = y^{(i)}\big]$$
$N$ = number of examples, $\hat y^{(i)}$ = predicted label of example $i$, $y^{(i)}$ = the true
label, $\mathbb{1}[\cdot]$ = 1 if the bracket is true and 0 if not. So it's "fraction you got right."
Before training (10 batches): **train 46.25%, val 45.00%, test 48.75%**, about the 50% coin-flip
level for a balanced 2-class task (book p. 193).

**Loss.** Accuracy can't be trained directly. It jumps in steps, so its gradient is zero almost
everywhere. The book uses [[cross-entropy-loss]] as a differentiable proxy (a
[[surrogate-loss]], book p. 193):
$$\mathcal{L}(\theta) = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{e^{z^{(i)}_{y^{(i)}}}}{\sum_{c=0}^{1} e^{z^{(i)}_c}}$$
- $\theta$ = the trainable weights (head + last block + final norm);
- $B$ = batch size (8); $i$ = one example in the batch;
- $z^{(i)}_c$ = the last-token logit of class $c$ for example $i$;
- the fraction = the probability the model gives to the **correct** class $y^{(i)}$; $-\log$ of it is
  near 0 when that probability is near 1, and large when it is near 0.

The only change from pretraining's `calc_loss_batch` is `[:, -1, :]`: use the last token only.

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]      # last output token only
    return torch.nn.functional.cross_entropy(logits, target_batch)
```

Initial loss (5 batches): train 2.453, val 2.583, test 2.322 (book p. 194). A uniform guess would
give $\ln 2 \approx 0.693$. So the random new head is *confidently wrong* on many examples (our
reading).

### Steps 8–9: train and evaluate (§6.7, book pp. 195–200)
`train_classifier_simple` (Listing 6.10) is the pretraining loop with two changes: it counts
*examples* seen, not tokens, and it prints accuracy after each [[epoch]] (book pp. 195–196).

```python
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epochs = 5
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5)
```

| | Value (book pp. 197–200) |
|---|---|
| Loss at step 0 → step 600 | train 2.153 → 0.083, val 2.392 → 0.074 |
| Accuracy after epoch 1 → 5 (5-batch estimate) | train 70.00% → 100.00%, val 72.50% → 97.50% |
| Run time | 5.65 min on M3 MacBook Air (< 0.5 min on V100/A100) |
| **Final, full sets** | **train 97.21%, val 97.32%, test 95.67%** |

There is little sign of [[overfitting]]: the training and validation curves stay close together
(Figs. 6.16–6.17). Five epochs is "usually a good starting point" (sidebar, book p. 199). The
slightly lower test score is normal, because the hyperparameters were tuned on the validation
set. More `drop_rate` or `weight_decay` ([[regularization]]) could narrow the gap (book p. 200).

### Step 10: classify new messages (§6.8, book pp. 200–203)
`classify_review` (Listing 6.12, book p. 201) repeats the `SpamDataset` preprocessing, then does
one forward pass:

```python
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]          # 1024
    input_ids = input_ids[:min(max_length, supported_context_length)]  # truncate
    input_ids += [pad_token_id] * (max_length - len(input_ids))        # pad
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # shape (1, T)
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]                         # shape (1, 2)
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"
```

- "You are a winner you have been specially selected to receive \$1000 cash or a \$2000 award." → **spam** ✔
- "Hey, just wanted to check if we're still on for dinner tonight? Let me know!" → **not spam** ✔

Call it with `max_length=train_dataset.max_length` (120), so new inputs look like the training
inputs. Save it with `torch.save(model.state_dict(), "review_classifier.pth")`
([[model-checkpointing]], book p. 202). Note: `max_length=None` (the default) would crash
`min(...)`, so always pass an integer.

## 🔴 Theory (Expert)
- **Closed label set.** The model is a conditional categorical distribution
  $p_\theta(y\mid x)$ over $K$ fixed classes ([[probability-distribution]]). Cross-entropy
  training is [[maximum-likelihood-estimation]] of that distribution, i.e.
  [[empirical-risk-minimization]] with log-loss. It cannot express any answer outside
  $\{0,\dots,K-1\}$. That is the "specialist" limit (book p. 171). The chapter summary stresses
  that this is the *same* cross-entropy loss as in pretraining, just over 2 classes instead of
  the whole vocabulary (book p. 203).
- **Two-node softmax = logistic regression on the logit difference** (standard math, not in the
  book):
  $$p_1 = \frac{e^{z_1}}{e^{z_0}+e^{z_1}} = \frac{1}{1+e^{-(z_1-z_0)}} = \sigma(z_1 - z_0)$$
  So only the difference $z_1 - z_0$ matters, and the 2-node head is a [[logistic-regression]] on
  the frozen-plus-tuned features $\mathbf h$, with weight $\mathbf w_1 - \mathbf w_0$. This is
  why a single sigmoid node with [[binary-cross-entropy]] is an equivalent choice (book p. 183
  points to it; appendix B p. 295).
- **Why last-token pooling, and the experimental proof.** Under the causal mask, the output at
  position $i$ is $\mathbf c_i = \sum_{j\le i}\alpha_{ij}\mathbf v_j$. Only $i = T$ (the last
  position) sums over the whole input. Exercise 6.3 tests the first token instead: test accuracy
  falls to **75.00%** vs 95.67% (appendix C, book p. 307). Appendix B cites work that *removes*
  the causal mask for classification (Li et al. 2023, "Label Supervised LLaMA Finetuning";
  BehnamGhader et al. 2024, LLM2Vec) and finds it can help (book p. 296).
- **Padding is not free.** Exercise 6.1 pads everything to 1,024 tokens: test accuracy drops to
  **78.33%** (appendix C, book p. 306). The last real token is no longer the last position, so
  the "last token" the head reads is a pad token whose attention is spread over hundreds of
  `<|endoftext|>` copies. That explanation is our reading; the book only reports the drop.
- **How much to fine-tune.** Head only is "technically sufficient," but tuning more layers
  noticeably helps (book p. 186; appendix B p. 296). Exercise 6.2 trains the whole model: test
  accuracy **96.67%**, about 1 point better (appendix C, book p. 306). The cost is gradients and
  optimizer state for all 124 M parameters instead of one block.
- **Parameter-efficient version (appendix E).** Freeze everything, then add [[low-rank-adaptation]]
  ($W + AB$) to every linear layer with rank 16 and alpha 16 ([[lora-rank-and-alpha]]). The
  trainable count drops from **124,441,346** to **2,666,528**, about 47× fewer (book p. 332).
  With AdamW `lr=8e-4`, 5 epochs, 12.10 min on an M3 MacBook Air: **train 100.00%, val 96.64%,
  test 98.00%** (book pp. 334–336). See [[parameter-efficient-fine-tuning]].
  > ⚠️ Book inconsistency: the printed output says test 98.00%, but the next sentence says
  > "96.64% and 97.33%" for validation and test (book p. 336). Figure E.5's caption says "six
  > epochs," but the code runs 5.
- **Balancing changes the prior.** Undersampling makes $\hat p(\text{spam}) = 1/2$ instead of
  $0.134$. The classifier's probabilities are calibrated to a 50/50 world. In the real 13% world
  you would need to shift the decision threshold (Bayes' rule with the true prior,
  [[bayes-theorem]]). The book doesn't discuss this; it points to the Imbalanced-Learn guide
  (appendix B p. 296).
- **Encoder alternatives.** BERT-style encoders are natural classifiers: X uses BERT for
  toxic-content detection (book p. 9). Appendix B lists [[bert]] and RoBERTa as alternatives to
  a decoder GPT-2 (book p. 296).
- **Book quirks.** Figure 6.14 shows `[3.5983, -3.9902]` → "not spam," but the code on p. 192
  prints `[-3.5983, 3.9902]` → "spam." The signs are swapped, and the figure is only an
  illustration. The chapter summary says the old head had "50,256" outputs. The vocabulary size
  is **50,257**; 50,256 is the largest token ID (book p. 202). §7.7 calls ch. 6 "completion
  fine-tuning" (book p. 236).

## Connections
- Builds on: [[fine-tuning]], [[pretraining]], [[pretrained-weights]], [[transfer-learning]], [[supervised-learning]], [[text-classification]], [[gpt-2]], [[gpt-model-configuration]], [[linear-layer]], [[linear-mapping]], [[softmax]], [[argmax]], [[cross-entropy-loss]], [[causal-attention-mask]]
- Pieces: [[class-imbalance]], [[label-encoding]], [[train-validation-test-split]], [[padding-and-truncation]], [[end-of-text-token]], [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[classification-head]], [[layer-freezing]], [[last-token-pooling]], [[logits]], [[monotonic-function]], [[classification-accuracy]], [[surrogate-loss]], [[training-loop]], [[epoch]], [[adamw-optimizer]], [[weight-decay]], [[train-vs-eval-mode]], [[model-checkpointing]], [[random-seed-reproducibility]]
- Math: [[matrix]], [[vector]], [[bias-vector]], [[probability-distribution]], [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[logistic-regression]], [[sigmoid-function]], [[binary-cross-entropy]], [[gradient-descent]], [[stochastic-gradient-descent]], [[backpropagation]], [[expected-value-and-mean]], [[overfitting]], [[overfitting-and-underfitting]], [[regularization]]
- Leads to: [[low-rank-adaptation]], [[parameter-efficient-fine-tuning]], [[llm-evaluation]]
- Contrast: [[instruction-fine-tuning]], [[bert]], [[masked-language-modeling]]
- Used by: [[llm-applications]], [[custom-llm]]
- People: [[sebastian-raschka]], [[jacob-devlin]] (BERT), [[edward-j-hu]] (LoRA)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 7 (classification fine-tuning = texts + class labels, e.g. spam).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7, book p. 14 (stage 3, step 8, Fig. 1.9).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book pp. 169–171 (definition, Fig. 6.3, "Choosing the right approach").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.2, book pp. 172–175 (SMS Spam Collection, Listings 6.1–6.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book pp. 175–181 (padding, `SpamDataset`, Listings 6.4–6.5).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.4, book pp. 181–183 (Listing 6.6, base model fails the spam prompt).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 183–190 (classification head, freezing, Listing 6.7, last token, Fig. 6.12).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book pp. 190–194 (softmax/argmax, Listings 6.8–6.9, initial accuracy and loss).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.7, book pp. 195–200 (Listing 6.10, training log, final 97.21 / 97.32 / 95.67%).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.8 and Ch. 6 summary, book pp. 200–203 (Listing 6.12 `classify_review`, saving).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.1, book pp. 204–205; §7.9, book p. 248 (step 8 in Figs. 7.1, 7.21).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix E §§E.1–E.4, book pp. 322–336 (LoRA spam classifier: 2,666,528 trainable params; 100.00 / 96.64 / 98.00%).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 295–296 (one-node outputs, layer experiments, imbalanced data, BERT/RoBERTa, removing the causal mask).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 306–307 (Exercises 6.1–6.3: 78.33%, 96.67%, 75.00%).
