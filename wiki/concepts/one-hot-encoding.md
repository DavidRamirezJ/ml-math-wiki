---
title: One-Hot Encoding
type: concept
tags: [deep-learning, linear-algebra, encoding, llm]
difficulty: beginner
sources: [llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# One-Hot Encoding

> **One-liner:** One-hot encoding turns a category number into a long list of zeros with a single 1 sitting in that category's slot.

## 🟢 Intuition (Dummy)
Picture a row of light switches, one switch for every word in the vocabulary. To say "word
number 3", you flip switch 3 on and leave every other switch off. That pattern of switches is the
word's one-hot code: exactly one switch is "hot".

It's a fair, neutral way to turn a label into numbers: no word looks "bigger" than another just
because its ID is larger (ID 5 isn't "more" than ID 2). The downside is obvious: with 50,257 words
you need 50,257 switches, and almost all of them are always off. That waste is exactly why LLMs
use an [[embedding-layer]] instead, which gets the same result without building the giant
mostly-zero list.

## 🟡 Mechanics (Practitioner)

### Definition
For a vocabulary of size $V$ and a token ID $i$ (counting from 0):
$$\mathbf{x}_i = (0,\dots,0,\underbrace{1}_{\text{position } i},0,\dots,0)^{\top}\in\{0,1\}^{V}$$
Symbol by symbol:
- $\mathbf{x}_i$ = the one-hot [[vector]] for token ID $i$.
- $\underbrace{1}_{\text{position } i}$ = the single 1, placed at index $i$; every other entry is 0.
- $^\top$ = transpose: written as a row here, but meant as a column vector.
- $\{0,1\}^V$ = "a list of $V$ entries, each either 0 or 1".

Example with the book's toy vocabulary $V = 6$ (§2.7): token ID 3 ("jumps") becomes
$$\mathbf{x}_3 = [0,0,0,1,0,0]^\top .$$

### One-hot × matrix = pick a row
Multiply the one-hot row by a weight [[matrix]] $W\in\mathbb{R}^{V\times d}$:
$$\mathbf{e}_i = \mathbf{x}_i^{\top} W = W_{i,:}$$
- $\mathbf{x}_i^\top W$ = row vector (length $V$) times matrix ($V\times d$) → row vector of length $d$.
- $W_{i,:}$ = row $i$ of $W$.
- $\mathbf{e}_i$ = the token's embedding.

With the book's seed-123 table, $\mathbf{x}_3^\top W = [-0.4015,\ 0.9666,\ -1.1481]$, which is
precisely what `embedding_layer(torch.tensor([3]))` returns (book pp. 42–43).

The book's NOTE (p. 43): the embedding layer "is essentially just a more efficient way of implementing
one-hot encoding followed by matrix multiplication in a fully connected layer". The supplementary code
(https://mng.bz/ZEB5) shows the two give identical outputs. Sketch:

```python
import torch
onehot = torch.nn.functional.one_hot(torch.tensor([2, 3, 5, 1]), num_classes=6).float()
linear = torch.nn.Linear(6, 3, bias=False)
linear.weight = torch.nn.Parameter(embedding_layer.weight.T)  # Linear stores (d, V)
print(linear(onehot))   # same 4x3 result as embedding_layer(input_ids)
```
(Sketch in the spirit of the supplementary code, not printed in the book.)

### Batch form
Stack $n$ one-hot rows into $X\in\{0,1\}^{n\times V}$; then $E = XW\in\mathbb{R}^{n\times d}$. For
`input_ids = [2, 3, 5, 1]` this gives the same $4\times3$ matrix as the embedding lookup.

## 🔴 Theory (Expert)
- **Standard basis.** The one-hot vectors $\mathbf{x}_0,\dots,\mathbf{x}_{V-1}$ are exactly the
  standard basis of $\mathbb{R}^V$ (see [[basis-and-rank]]). So "multiply by one-hot" = "evaluate the
  [[linear-mapping]] $\mathbb{R}^V\to\mathbb{R}^d$ on a basis vector" = "read off one column of
  $W^\top$" (one row of $W$). That is MML Def 2.19's "columns of the transformation matrix are images
  of the basis vectors", applied to embeddings.
- **Orthonormal, equidistant.** $\mathbf{x}_i^\top\mathbf{x}_j = \delta_{ij}$, so every pair of
  distinct tokens is at the same Euclidean distance $\sqrt2$. One-hot codes carry **no notion of
  similarity**. A learned embedding maps them into $\mathbb{R}^d$ where distances *can* become
  meaningful (see [[word-embeddings]]).
- **Cost.** $\mathbf{x}_i^\top W$ costs $O(Vd)$ multiply-adds, all but $d$ of them by zero. The
  lookup costs $O(d)$. With $V=50{,}257$ this is the efficiency argument of the NOTE.
- **Gradients.** With $E=XW$, $\partial L/\partial W = X^\top\,\partial L/\partial E$, so only rows
  of tokens present in the batch get updated, which is why the embedding layer "can be seen as a
  neural network layer that can be optimized via backpropagation" (book p. 43; gradient formula
  implied, not printed).
- **Elsewhere in ML.** One-hot vectors are also the target encoding behind
  [[cross-entropy-loss]]: the loss is $-\log p_{y}$, i.e. the dot product of the one-hot target
  with $-\log\mathbf{p}$. PyTorch's `cross_entropy` takes integer class IDs and skips the one-hot
  step internally (general fact, not in §2.7).

## Connections
- Builds on: [[token-id]], [[vocabulary]], [[vector]], [[basis-and-rank]]
- Leads to: [[embedding-layer]], [[token-embedding]], [[word-embeddings]]
- Used by: [[linear-layer]], [[cross-entropy-loss]], [[label-encoding]]
- Related: [[matrix]], [[linear-mapping]], [[matrix-multiplication]]
- People: —

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** Appendix B lists two Raschka resources for Ch. 5 about how one-hot targets relate to PyTorch's cross-entropy. One is the video "L8.7.1 OneHot Encoding and Multi-category Cross Entropy". The other is the code "Understanding Onehot Encoding and Cross Entropy in PyTorch" (book p. 293). Ch. 5 itself never builds one-hot vectors. `torch.nn.functional.cross_entropy` takes the integer target IDs directly and picks the matching log-probability. This is the same number as the dot product of the log-probability vector with a one-hot target vector. That equivalence is standard math and is not printed in Ch. 5 (the call is on book pp. 139–140).

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book pp. 41–44 (NOTE on one-hot + matrix multiplication, p. 43; token ID 3 lookup, pp. 42–43).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 43–44 (the NOTE is printed on p. 43, the page where §2.8 starts; Fig. 2.17: the same ID always maps to the same vector, so position must be added separately).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.2, book pp. 139–140 (`cross_entropy` with integer targets); appendix B (Ch. 5 refs), book p. 293.
