---
title: Absolute Positional Embeddings
type: concept
tags: [llm, embeddings, transformer, gpt]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Absolute Positional Embeddings

> **One-liner:** An absolute positional embedding gives each seat number in the sentence (0, 1, 2, …) its own fixed-for-that-seat vector, and adds it to whatever word sits there.

## 🟢 Intuition (Dummy)
Think of a cinema. Every seat has a number printed on it: seat 0, seat 1, seat 2…
It doesn't matter who sits there — seat 5 is always seat 5. An absolute positional
embedding works the same way. The model keeps one special vector for "seat 0",
another for "seat 1", and so on, and glues the right one onto each word.

The book puts it plainly: "the first token will have a specific positional
embedding, the second token another distinct embedding, and so on" (book p. 45).

So when the word "fox" appears in seat 0 and again in seat 3, the two copies end up
with different input vectors, because they got different seat tags. It's the
opposite style to [[relative-positional-embedding|relative]] embeddings, which
care about the *distance* between words, not their seat numbers.

## 🟡 Mechanics (Practitioner)

### Formula
$$\mathbf{x}_t = E_{\text{tok}}[\,\text{id}_t\,] + \mathbf{p}_t, \qquad \mathbf{p}_t = E_{\text{pos}}[\,t\,]$$

Symbol by symbol:
- $\mathbf{x}_t$ = the [[input-embeddings|input embedding]] at position $t$.
- $E_{\text{tok}}[\text{id}_t]$ = the [[token-embedding]] of the word at position $t$ (row $\text{id}_t$ of the $V\times d$ token table).
- $\mathbf{p}_t$ = the positional vector for seat $t$.
- $E_{\text{pos}} \in \mathbb{R}^{L\times d}$ = the positional table: $L$ rows (one per seat, $L$ = [[context-length]]), $d$ columns ([[embedding-dimension]]). "$\in \mathbb{R}^{L\times d}$" reads "is a grid of $L$ by $d$ real numbers".
- The key word is *absolute*: $\mathbf{p}_t$ depends only on $t$ — never on the word, and never on other positions.

### Worked example (Fig. 2.18, book p. 45)
Token vectors drawn as $[1,1,1]$, positional vectors $[1.1,1.2,1.3]$, $[2.1,2.2,2.3]$,
$[3.1,3.2,3.3]$, $[4.1,4.2,4.3]$. First seat:

$$[1,1,1] + [1.1,1.2,1.3] = [2.1, 2.2, 2.3]$$

and the other seats give $[3.1,3.2,3.3]$, $[4.1,4.2,4.3]$, $[5.1,5.2,5.3]$.

### Learned, GPT style (book pp. 45, 47)
OpenAI's GPT models use absolute positional embeddings that are **optimized during
training**, not fixed in advance (book p. 45). In code, that's just a second
[[embedding-layer]]:

```python
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))  # [4, 256]
```

The table starts out random and gets nudged by [[gradient-descent]] like every
other weight.

| Setting | $L$ (rows) | $d$ (cols) | Learned numbers | Where |
|---|---|---|---|---|
| Ch. 2 demo | 4 | 256 | 1,024 | book p. 47 |
| Ch. 5 training run | 256 | 768 | 196,608 | book p. 148 (config from §5.1) |
| GPT-2 124M | 1,024 | 768 | 786,432 | book pp. 95, 164 |

(The "learned numbers" column is our multiplication $L\cdot d$.)

When the book loads OpenAI's GPT-2 weights, the positional table is copied from
the array named `wpe` (shape $1024\times768$) into `gpt.pos_emb.weight` (book p. 165).

## 🔴 Theory (Expert)
- **Hard length ceiling.** A learned absolute table is a finite lookup over
  $\{0,\dots,L-1\}$. Position $L$ simply has no row, so the model cannot be run on
  longer inputs without truncation (book p. 47) or extending/re-training the table.
  The book reads the ceiling straight off the table: `model.pos_emb.weight.shape[0]`
  (book p. 148). Fixed [[sinusoidal-positional-encoding|sinusoidal]] encodings are
  also absolute, but are defined for every $t$.
- **Which rows get trained.** Row $t$ only receives gradient from sequences that
  are at least $t+1$ tokens long. The book's [[sliding-window-sampling]] produces
  full-length windows, so every row in the table is trained.
- **Translation sensitivity.** Shifting a phrase by $k$ seats changes every
  $\mathbf{p}_t$ it touches, so the model must learn shift-invariant behaviour
  from data. This is the main argument for [[relative-positional-embedding|relative]]
  schemes, which the book says generalize better to unseen lengths (book p. 45).
- **Expressiveness.** Since $\mathbf{p}_t$ is a free vector per position, a learned
  table can represent any absolute scheme of the same width, including sinusoids —
  but only on $t < L$.

## Connections
- Builds on: [[positional-embedding]], [[embedding-layer]], [[token-embedding]], [[vector]]
- Leads to: [[input-embeddings]], [[gpt-architecture]], [[gpt-2]]
- Contrast: [[relative-positional-embedding]], [[sinusoidal-positional-encoding]]
- Uses: [[backpropagation]], [[gradient-descent]], [[context-length]]
- People: [[alec-radford]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 45–49 (definition and Fig. 2.18 p. 45; code p. 47; summary p. 49).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book p. 95 (`context_length` = 1,024).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2 p. 148 and §5.5 pp. 164–165 (`pos_emb.weight.shape[0]`, `wpe`).
