---
title: Hidden State
type: concept
tags: [deep-learning, sequence-models, rnn, nlp]
difficulty: beginner
sources: [llm-ch03-coding-attention-mechanisms]
created: 2026-10-13
updated: 2026-10-13
---

# Hidden State

> **One-liner:** A hidden state is the network's running "memory" — a vector of internal numbers that a recurrent network updates after every word it reads.

## 🟢 Intuition (Dummy)
Imagine reading a sentence one word at a time and, after each word, scribbling a
short summary on a sticky note. You never re-read the old words; you only look at
your sticky note and the new word, then rewrite the note. That sticky note is the
**hidden state**.

Older translation models (encoder–decoder [[recurrent-neural-network|RNNs]])
worked exactly like this. The encoder read the German sentence word by word and
kept updating its sticky note. When it reached the end, the *final* note had to
hold the meaning of the whole sentence. Then the decoder took that single note and
wrote the English translation one word at a time, keeping its own sticky note as it
went.

The catch is obvious once you picture it: a sticky note has a fixed size. A short
sentence fits fine, but a long, twisty sentence gets squashed — details fall off.
That squashing problem is exactly why [[attention-mechanism|attention]] was invented
(see [[long-range-dependencies]]).

## 🟡 Mechanics (Practitioner)
**What the book says (Raschka §3.1).** An RNN is a network where *outputs from
previous steps are fed as inputs to the current step*, which makes it a good fit for
sequential data like text (book pp. 52–53). At each step the encoder updates its
**hidden state** — "the internal values at the hidden layers" (book p. 53). The
encoder tries to capture the entire meaning of the input in the **final** hidden
state; the decoder takes that state and generates the output one word at a time,
updating its own hidden state as it goes (book p. 53). The book adds a useful
mental shortcut: you can think of the hidden state as an **embedding vector**, the
same kind of object as the [[token-embedding|token embeddings]] from chapter 2
(book p. 53).

**Figure 3.4's example (book p. 53).** The encoder reads 3 German tokens,
"Kannst du mir"; its hidden states feed one "memory cell (hidden state) memorizing
the entire input"; the decoder produces "Can you help". The caption calls the
hidden state "a compressed representation of the entire input sequence".

**The standard update rule** (background — the book prints *no* formulas in §3.1):
$$\mathbf h_t = f\big(W_{hh}\,\mathbf h_{t-1} + W_{xh}\,\mathbf x_t + \mathbf b\big)$$

Reading it symbol by symbol:
- $\mathbf h_t$ = the hidden state (a [[vector]]) *after* reading word number $t$.
- $\mathbf h_{t-1}$ = the hidden state from the previous step — "the old sticky note".
- $\mathbf x_t$ = the embedding vector of the word read at step $t$.
- $W_{hh}$ = a [[matrix]] of learned weights that says how the old memory feeds into the new one.
- $W_{xh}$ = a matrix of learned weights that says how the new word feeds into the memory.
- $\mathbf b$ = a learned [[bias-vector]].
- $f$ = a squashing nonlinearity, typically $\tanh$, applied element by element.

So each step is a [[linear-mapping]] of (old memory, new word), plus a bias, pushed
through a nonlinearity. Unrolled over a 3-word input:
$$\mathbf h_3 = f\big(W_{hh}\,f\big(W_{hh}\,f(W_{hh}\mathbf h_0 + W_{xh}\mathbf x_1+\mathbf b) + W_{xh}\mathbf x_2+\mathbf b\big) + W_{xh}\mathbf x_3+\mathbf b\big)$$
— one nested function. The decoder only ever gets $\mathbf h_3$.

**Tiny PyTorch sketch** (not from the book; the book deliberately skips RNN code):
```python
import torch
rnn = torch.nn.RNN(input_size=3, hidden_size=4, batch_first=True)
x = torch.randn(1, 3, 3)          # 1 sentence, 3 tokens, 3-dim embeddings
all_h, h_final = rnn(x)           # all_h: (1, 3, 4)  h_final: (1, 1, 4)
```
`h_final` is the single vector a classic encoder hands to the decoder.

**Where it matters for LLMs.** The book is explicit that you do *not* need RNN
details to build an LLM (book p. 54). What you need is the lesson: a single
fixed-size hidden state is a bottleneck. GPT-style models replace it with
[[self-attention]], where every token gets its own [[context-vector]] built from
*all* earlier tokens.

## 🔴 Theory (Expert)
- **State-space view.** The recurrence $\mathbf h_t = F_\theta(\mathbf h_{t-1},\mathbf x_t)$
  is a discrete-time dynamical system; the hidden state is a *sufficient summary* of
  the past for the model's purposes — everything the future output can depend on
  must pass through $\mathbf h_t\in\mathbb R^{d}$.
- **Capacity argument.** A fixed $d$-dimensional vector must encode sequences of
  unbounded length. With finite precision this is a lossy compression — the same
  intuition as [[dimensionality-reduction]] or [[low-rank-approximation]]: you keep
  what fits and throw the rest away. This is the "bottleneck" the book calls out
  (book pp. 53–54).
- **Training difficulty.** Gradients flow back through the chain
  $\partial\mathbf h_T/\partial\mathbf h_1=\prod_{t=2}^{T}\partial\mathbf h_t/\partial\mathbf h_{t-1}$
  ([[chain-rule]], backpropagation through time). Repeated Jacobian products shrink
  or blow up — see [[vanishing-and-exploding-gradients]]. Gated cells (LSTM, GRU)
  add additive memory paths to ease this, but do not remove the fixed-size bottleneck.
- **Contrast with attention.** In an encoder–decoder with [[bahdanau-attention]],
  the decoder keeps *all* encoder hidden states $\mathbf h_1,\dots,\mathbf h_T$ and
  mixes them with learned weights at every step. Transformers go further and drop
  recurrence entirely: the per-token representation is recomputed from the whole
  (causally masked) sequence, so there is no single running state. (Modern
  state-space models such as Mamba revive a fixed-size state with better training
  properties — [[albert-gu]], [[tri-dao]].)

## Connections
- Builds on: [[vector]], [[matrix]], [[linear-mapping]], [[recurrent-neural-network]], [[token-embedding]]
- Leads to: [[long-range-dependencies]], [[bahdanau-attention]], [[attention-mechanism]], [[self-attention]], [[context-vector]]
- Used by: [[encoder-decoder-architecture]], [[machine-translation]]
- Related math: [[dimensionality-reduction]], [[chain-rule]], [[vanishing-and-exploding-gradients]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.1, book pp. 52–54 (Fig. 3.4, p. 53). The RNN update formula and the PyTorch sketch are standard background, not printed in the book.
