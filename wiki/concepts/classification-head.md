---
title: Classification Head (Output Nodes vs Classes)
type: concept
tags: [deep-learning, classification, fine-tuning, linear-algebra, llm]
difficulty: intermediate
sources: [llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Classification Head (Output Nodes vs Classes)

> **One-liner:** A classification head is a small new output layer bolted onto a pretrained model, with one output number per class, replacing the layer that used to predict the next word.

## 🟢 Intuition (Dummy)
Think of the pretrained GPT as a very well-read brain wired to a giant scoreboard with 50,257
lights, one for every word it might say next. For spam detection we don't need 50,257 lights.
We need two: **not spam** and **spam**.

So we unplug the giant scoreboard and plug in a tiny one with two lights. The brain stays the
same; it just reports to a new board. Whichever light shines brighter is the answer.

## 🟡 Mechanics (Practitioner)

### Before and after (Fig. 6.9, book pp. 184–185)
`print(model)` ends with:
```
(final_norm): LayerNorm()
(out_head): Linear(in_features=768, out_features=50257, bias=False)
```
That's the language-model head: 768 hidden numbers in, 50,257 vocabulary scores out. We swap it
(Listing 6.7, book p. 186):

```python
torch.manual_seed(123)
num_classes = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],   # 768 for gpt2-small
    out_features=num_classes
)
```
Using `BASE_CONFIG["emb_dim"]` instead of hard-coding 768 keeps it working for the bigger GPT-2
sizes (book p. 186). The new layer starts with random weights (seed 123) and, by default, has a
bias.

### The math
For one token's final hidden vector:

$$\mathbf z=\mathbf h\,W_\text{cls}+\mathbf b$$

Symbol by symbol:
- $\mathbf h\in\mathbb R^{768}$ = the token's hidden vector after the final LayerNorm (a row of 768 numbers).
- $W_\text{cls}\in\mathbb R^{768\times2}$ = the head's weight [[matrix]]: 768 rows, 2 columns,
  one column per class.
- $\mathbf b\in\mathbb R^{2}$ = the [[bias-vector]], one offset per class.
- $\mathbf z\in\mathbb R^{2}$ = the two class [[logits]] $(z_0,z_1)$.
- $\mathbf h\,W_\text{cls}$ = a vector times a matrix: entry $k$ is the [[inner-product|dot product]] of
  $\mathbf h$ with column $k$.

(PyTorch stores the weight transposed, shape `[2, 768]`, and computes $\mathbf h W^\top+\mathbf b$ —
same thing.)

**Parameter count** (derived): new head $768\times2+2=1{,}538$ numbers; old head
$768\times50{,}257=38{,}597{,}376$.

### Running it (book pp. 187–188)
```python
inputs = torch.tensor(tokenizer.encode("Do you have time")).unsqueeze(0)
# tensor([[5211, 345, 423, 640]])   shape [1, 4]
with torch.no_grad():
    outputs = model(inputs)
# tensor([[[-1.5854,  0.9904],
#          [-3.7235,  7.4548],
#          [-2.2661,  6.6049],
#          [-3.5983,  3.9902]]])   shape [1, 4, 2]
```
The shape went from `[1, 4, 50257]` to `[1, 4, 2]`: still one row per input token, but only two
columns now (book p. 188). The head is applied to every token separately. For a single label per
message we keep only the **last** row, `outputs[:, -1, :]` → `[[-3.5983, 3.9902]]`, because under
causal attention the last token is the only one that has seen the whole message
([[last-token-pooling]], book pp. 188–190). Exercise 6.3 asks you to fine-tune on the *first*
output token instead and compare (book p. 190).

### Sidebar: why 2 output nodes, not 1? (book p. 183)
For a yes/no task you *could* use one output node, but then you'd have to change the loss
function. The book goes with the general rule:

$$\#\text{output nodes}=K=\#\text{classes}$$

- $K$ = number of classes. Spam: $K=2$. A news classifier with "Technology", "Sports",
  "Politics": $K=3$ (book p. 183).

This lines up with PyTorch's zero-based labels: classes $0,\dots,K-1$ index the $K$ outputs
([[label-encoding]]).

### What gets trained (book pp. 186–187, Fig. 6.10)
```python
for param in model.parameters():
    param.requires_grad = False                 # freeze everything
# new out_head is trainable by default
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True                  # last transformer block
for param in model.final_norm.parameters():
    param.requires_grad = True                  # final LayerNorm
```
Training only the head would technically work, but the author found that also unfreezing the
last block "noticeably" improves results (book p. 186). See [[layer-freezing]]. Result: the
output layer, final LayerNorm and the last of the 12 transformer blocks train; the other 11
blocks and the embedding layers stay frozen (Fig. 6.10, book p. 187).

The sidebar "Fine-tuning selected layers vs. all layers" (book p. 185) gives the reason: lower
layers capture basic language structure and semantics that transfer across tasks, while layers
near the output are more task-specific, so tuning only the top layers is often enough and is
cheaper to compute. Exercise 6.2 asks you to fine-tune the whole model instead and compare
(book p. 186).

## 🔴 Theory (Expert)
- **Two-node softmax = one-node sigmoid.** With logits $(z_0,z_1)$,
  $$p_1=\frac{e^{z_1}}{e^{z_0}+e^{z_1}}=\frac{1}{1+e^{-(z_1-z_0)}}=\sigma(z_1-z_0).$$
  Only the difference $z_1-z_0$ matters, so the two-node head
  $(W_{:,0},W_{:,1},b_0,b_1)$ is an over-parameterized version of a single-node logistic head with
  weights $W_{:,1}-W_{:,0}$ and bias $b_1-b_0$. The one-node version would use
  [[sigmoid-function|sigmoid]] + [[binary-cross-entropy]] — the "modified loss" the sidebar
  alludes to. Both give identical predictions and gradients on the difference; the $K$-node form
  simply generalizes to $K>2$.
- **A linear probe on learned features.** With the backbone frozen, the head is multinomial
  [[logistic-regression]] on features $\mathbf h(\mathbf x)$: a [[linear-mapping]]
  $\mathbb R^{768}\to\mathbb R^K$ followed by [[softmax]]. Its decision boundaries are hyperplanes
  in the 768-dimensional hidden space. Unfreezing the last block makes $\mathbf h$ itself learnable.
- **No softmax needed at inference.** Softmax is a [[monotonic-function|monotonic]] map, so
  $\arg\max_k\mathrm{softmax}(\mathbf z)_k=\arg\max_k z_k$ (used in §6.6).
- **Weight-tying lost.** GPT-2's original `out_head` shares its weights with the token embedding
  ([[weight-tying]]). The new head is untied and freshly initialized, which is why the untrained
  model scores ~50 % and the initial losses (train 2.453, val 2.583, test 2.322 on 5 batches,
  book p. 194) are far above the uniform-guess value $\ln2\approx0.693$.
- **Book link inconsistency.** The §6.5 sidebar cites "Losses Learned" at mng.bz/NRZ2 (book
  p. 183) while appendix B lists it as mng.bz/ZEJA (book p. 295). Same article title.

## Connections
- Builds on: [[linear-layer]], [[linear-mapping]], [[matrix]], [[logits]], [[gpt-architecture]], [[label-encoding]]
- Leads to: [[softmax]], [[cross-entropy-loss]], [[classification-accuracy]], [[last-token-pooling]], [[layer-freezing]]
- Used by: [[classification-fine-tuning]], [[text-classification]], [[transfer-learning]]
- Related: [[logistic-regression]], [[sigmoid-function]], [[binary-cross-entropy]], [[causal-attention-mask]], [[gpt-2]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 183–190 (sidebar "Output layer nodes" p. 183; Fig. 6.9 p. 184; `print(model)` p. 185; Listing 6.7 & freezing p. 186; Fig. 6.10 p. 187; outputs p. 188; Figs. 6.11–6.12 pp. 189–190; Exercise 6.3 p. 190)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book p. 194 (initial loss values)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 295 (one output node for binary classification)
