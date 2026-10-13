---
title: Last-Token Representation for Classification
type: concept
tags: [deep-learning, transformer, attention, classification, llm]
difficulty: intermediate
sources: [llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Last-Token Representation for Classification

> **One-liner:** To get one label for a whole text out of a GPT model, you read the output at the last token, because it's the only spot that has "seen" the entire text.

## 🟢 Intuition (Dummy)
A GPT model reads text strictly left to right, and it isn't allowed to peek ahead.
Imagine a line of people where each person can only hear the ones in front of them.
The first person has heard only themselves. The last person has heard *everyone*.

Say you want one quick verdict on the whole message, like "spam or not spam?". You'd
ask the person at the end of the line. That's exactly what we do with a GPT
classifier. The model gives you an output for *every* token, but only the last token's
output summarises the entire message. So that's the one you use.

The book shows it the hard way too. In Exercise 6.3 you classify from the *first*
token instead (change `[:, -1, :]` to `[:, 0, :]` everywhere), and test accuracy falls
from **95.67% to 75.00%**, "since the first token contains less information than the
last token" (book p. 307).

## 🟡 Mechanics (Practitioner)

### Shapes: one output row per token
After you swap in the 2-class [[classification-head]], the GPT model still returns one
row of [[logits]] for every input token (book p. 188):

```python
inputs = tokenizer.encode("Do you have time")       # [5211, 345, 423, 640]
inputs = torch.tensor(inputs).unsqueeze(0)          # shape [1, 4]
with torch.no_grad():
    outputs = model(inputs)                         # shape [1, 4, 2]
# tensor([[[-1.5854,  0.9904],
#          [-3.7235,  7.4548],
#          [-2.2661,  6.6049],
#          [-3.5983,  3.9902]]])
print("Last output token:", outputs[:, -1, :])      # tensor([[-3.5983, 3.9902]])
```

Here's what that slice `outputs[:, -1, :]` means:
- `:` in the first slot = keep every example in the batch.
- `-1` in the second slot = take the **last** token position.
- `:` in the third slot = keep both class logits.

So the shape goes $[1, 4, 768] \xrightarrow{\text{head}} [1, 4, 2] \xrightarrow{[:,-1,:]} [1, 2]$.
The four rows are the four tokens and the two columns are the two classes
(book pp. 188–189, Fig. 6.11). Before the head swap the same input gave $[1, 4, 50257]$,
one column per [[vocabulary]] entry.

### Why the last token? Causal attention
[[causal-attention|Causal attention]] only lets token $i$ look at tokens $1,\dots,i$
(book pp. 188–190). Its output is a weighted average:

$$\mathbf{c}_i = \sum_{j \le i} \alpha_{ij}\,\mathbf{v}_j, \qquad \alpha_{ij} = 0 \ \text{for}\ j > i$$

Reading it symbol by symbol:
- $\mathbf{c}_i$ = the [[context-vector]] (the new representation) of token $i$.
- $\sum_{j \le i}$ = "add up over every position $j$ that comes at or before $i$". Future
  tokens ($j > i$) aren't in the sum at all.
- $\alpha_{ij}$ (alpha) = the [[attention-weights|attention weight]], meaning how much token $i$
  pays attention to token $j$. Each row of weights adds up to 1.
- $\mathbf{v}_j$ = the value vector of token $j$, see [[query-key-value]].
- $\alpha_{ij} = 0$ for $j > i$ = the [[causal-attention-mask]] blanks out the future.

Only the **last** position $i = T$ (with $T$ = number of tokens) gets a sum that runs
over *all* $T$ tokens. The book's Fig. 6.12 shows this for "Do you have time" (book
p. 190). Rows are queries, columns are keys, and the empty cells are masked out:

| | Do | you | have | time |
|---|---|---|---|---|
| **Do** | 1.0 | | | |
| **you** | 0.55 | 0.45 | | |
| **have** | 0.38 | 0.30 | 0.32 | |
| **time** | 0.27 | 0.24 | 0.24 | 0.25 |

The "time" row is the only full row, which is the book's point: "the last token is the
only token with an attention score to all other tokens." (The caption says "attention
scores", but since each row sums to 1 these are really normalised attention
*weights*.) The book's conclusion: "the last token in a sequence accumulates the most
information since it is the only token with access to data from all the previous
tokens. Therefore, in our spam classification task, we focus on this last token during
the fine-tuning process" (book p. 190). The book also notes we "don't need to fine-tune
all four output rows"; only the last row enters the loss (book p. 188).

### The prediction rule
Write $\mathbf{h}_T \in \mathbb{R}^{768}$ for the last token's final hidden vector. The classifier then is

$$\mathbf{z} = \mathbf{h}_T W^\top + \mathbf{b}, \qquad \hat y = \arg\max_k z_k$$

- $\mathbf{h}_T$ = the output of the final LayerNorm at position $T$ (a 768-number vector for GPT-2 small).
- $W$ ($2 \times 768$) and $\mathbf{b}$ (2 numbers) = the head's weights and [[bias-vector|bias]].
- $\mathbf{z} = (z_0, z_1)$ = the two class logits. Index 0 is "not spam", index 1 is "spam".
- $\arg\max_k z_k$ = "the index $k$ of the biggest logit", see [[argmax]].

You use the same last-token slice everywhere in chapter 6. It shows up in
`calc_loss_batch`, `calc_accuracy_loader` and `classify_review` (book pp. 192–194, 201):
`logits = model(input_batch)[:, -1, :]`.

## 🔴 Theory (Expert)
- **Receptive field under a causal mask.** By induction over the layers, the hidden
  state at position $i$ in *any* layer depends only on tokens $x_1, \dots, x_i$. The
  attention sum skips $j > i$, and the [[layer-normalization|LayerNorm]], [[feed-forward-network]]
  and [[shortcut-connections|residual]] steps all act on one position at a time. So
  $\mathbf{h}_T = f(x_1,\dots,x_T)$ is a function of the whole sequence, while
  $\mathbf{h}_1 = g(x_1)$ depends on the **first token alone** (plus its position
  embedding). A first-token classifier is literally deciding "spam or not" from one
  token. Exercise 6.3's 75.00% (book p. 307) is about what that one-token signal is
  worth on this data.
- **It's a pooling choice.** Turning $T$ vectors into one is called *pooling*. Taking
  the last token is the natural choice for [[decoder-only-transformer|decoder-only]]
  (causal) models. Other options, which are general context and not from this book:
  mean pooling over positions, or a dedicated summary token like BERT's `[CLS]`, which
  works in [[bert]] because its attention is *bidirectional*, so every position sees
  everything.
- **The padding catch.** In chapter 6 each message is right-padded with
  `<|endoftext|>` (ID 50256) up to the longest training message, 120 tokens (book
  pp. 176–178). So for most messages the "last token" is a **padding token**, not the
  last real word. It still works, because the pad positions aren't masked out of
  attention, so the final pad token attends to all the real tokens before it. The
  model just learns to read the summary from that slot. (This is our observation. The
  book doesn't say it.) It also helps explain Exercise 6.1: padding to 1,024 tokens
  drops test accuracy to **78.33%** (book p. 306). The real content gets diluted
  among ~900 pad tokens at the readout position. If you want the last *real* token
  instead, you'd index each row at `length_i - 1`.
- **Connection to next-token prediction.** In [[pretraining]] the last position is the
  one that predicts the *next* token, so its representation is already trained to be
  a summary of the whole prefix. Classification fine-tuning just reuses that summary
  with a new 2-way readout instead of the 50,257-way one. That's why
  [[next-token-prediction|next-token]] pretraining transfers so naturally to
  last-token classification.

## Connections
- Builds on: [[causal-attention]], [[causal-attention-mask]], [[attention-weights]], [[context-vector]], [[logits]], [[tensor]]
- Leads to: [[classification-fine-tuning]], [[classification-accuracy]], [[surrogate-loss]]
- Related: [[classification-head]], [[layer-freezing]], [[padding-and-truncation]], [[end-of-text-token]], [[bert]], [[decoder-only-transformer]], [[argmax]], [[softmax]]
- Used by: [[text-classification]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 187–190 (output shape [1, 4, 2], `outputs[:, -1, :]`, Figs. 6.11–6.12, causal-mask argument)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book pp. 192–194 (last-token logits in `calc_accuracy_loader` and `calc_loss_batch`)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book pp. 176–178 (right-padding with token 50256 to length 120)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercises 6.1 and 6.3, book pp. 306–307 (78.33% with 1,024 padding; 75.00% with first token vs. 95.67%)
