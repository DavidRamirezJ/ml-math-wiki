---
title: Input Embeddings (Token + Positional)
type: concept
tags: [llm, embeddings, pipeline, pytorch]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Input Embeddings (Token + Positional)

> **One-liner:** The input embedding is what an LLM actually reads: the "what word is this" vector plus the "where is it" vector, added together.

## 🟢 Intuition (Dummy)
Think of a name badge at a conference. One sticker says *who* you are; another
sticker says *which table* you're sitting at. Stick both on the same badge and the
organisers know everything they need in one glance.

An LLM can't read letters, only numbers. So every word goes through a little
assembly line (book p. 48, Fig. 2.19):

1. Chop the text into [[token|tokens]]: "This is an example." → `This`, `is`, `an`, `example`, `.`
2. Swap each token for its ID number from the [[vocabulary]]: 40134, 2052, 133, 389, 12.
3. Look up each ID's [[token-embedding]] — the "who" sticker.
4. Look up each seat's [[positional-embedding]] — the "where" sticker.
5. **Add them.** The result is the **input embedding**, and it goes into the
   [[decoder-only-transformer|GPT-like decoder-only transformer]].

That's the end of chapter 2 of the book: the data is now ready for the "main LLM
modules" (book p. 47).

## 🟡 Mechanics (Practitioner)

### Formula, one token
$$\mathbf{x}_t = E_{\text{tok}}[\,\text{id}_t\,] + E_{\text{pos}}[\,t\,]$$

- $\mathbf{x}_t$ = the input embedding for the token at position $t$, a [[vector]] with $d$ numbers.
- $E_{\text{tok}}[\text{id}_t]$ = row $\text{id}_t$ of the token table ($V \times d$). "What word."
- $E_{\text{pos}}[t]$ = row $t$ of the position table ($L\times d$). "Which seat."
- $+$ = add matching entries. Both vectors have length $d$, so this works (book p. 45).

### Formula, whole batch
$$X_{b,t,:} = T_{b,t,:} + P_{t,:}, \qquad b \in \{0,\dots,7\},\ t\in\{0,\dots,3\}$$

- $T \in \mathbb{R}^{8\times4\times256}$ = token embeddings of the batch (8 sequences, 4 tokens, 256 numbers each).
- $P \in \mathbb{R}^{4\times256}$ = one positional vector per seat.
- $X \in \mathbb{R}^{8\times4\times256}$ = the input embeddings.
- $X_{b,t,:}$ = "sequence $b$, position $t$, all 256 coordinates" (the colon means *every* entry along that axis).
- Notice $P$ has no $b$: the *same* seat vectors are reused for all 8 sequences. That's [[broadcasting]].

### The book's code, start to finish (book pp. 46–47)
```python
vocab_size, output_dim = 50257, 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 50,257 x 256

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8,
    max_length=max_length, stride=max_length, shuffle=False)
inputs, targets = next(iter(dataloader))      # inputs: [8, 4] token IDs

token_embeddings = token_embedding_layer(inputs)          # [8, 4, 256]

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))  # [4, 256]

input_embeddings = token_embeddings + pos_embeddings      # [8, 4, 256]
```

The first row of `inputs` is `[40, 367, 2885, 1464]` (book p. 46). The book picks $d = 256$ for experiments; for scale, GPT-3 uses an embedding size of 12,288 (book p. 46). Shapes walk
like this:

| Step | Shape | Meaning |
|---|---|---|
| `inputs` | [8, 4] | 8 samples × 4 token IDs (book p. 46) |
| `token_embeddings` | [8, 4, 256] | each ID → 256 numbers (book p. 47) |
| `pos_embeddings` | [4, 256] | one vector per seat (book p. 47) |
| `input_embeddings` | [8, 4, 256] | sum, ready for the LLM (book p. 47) |

The book describes the addition as PyTorch adding "the 4 × 256–dimensional
pos_embeddings tensor to each 4 × 256–dimensional token embedding tensor in each of
the eight batches" (book p. 47).

### In the GPT model (book pp. 96–98)
In chapter 4 this lives *inside* the model's `forward` (`DummyGPTModel`, book p. 96; same code in `GPTModel`, book p. 119), followed
straight away by [[dropout]]:

```python
tok_embeds = self.tok_emb(in_idx)                                   # [B, T, 768]
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # [T, 768]
x = tok_embeds + pos_embeds                                         # [B, T, 768]
x = self.drop_emb(x)
```

With GPT-2 124M settings, $V = 50{,}257$, $L = 1{,}024$, $d = 768$ (book p. 95).
For "Every effort moves you" the IDs are 6109, 3626, 6100, 345, and each becomes a
768-dimensional vector (book p. 98).

## 🔴 Theory (Expert)
- **One-hot view.** With $\mathbf{u}_{\text{id}} \in \{0,1\}^V$ and $\mathbf{v}_t\in\{0,1\}^L$
  the [[one-hot-encoding|one-hot]] vectors of the token and the position,
  $$\mathbf{x}_t^\top = \mathbf{u}_{\text{id}_t}^\top E_{\text{tok}} + \mathbf{v}_t^\top E_{\text{pos}} = \begin{bmatrix}\mathbf{u}_{\text{id}_t}^\top & \mathbf{v}_t^\top\end{bmatrix}\begin{bmatrix}E_{\text{tok}}\\ E_{\text{pos}}\end{bmatrix}.$$
  So the input embedding is a single [[linear-mapping|linear map]] applied to the
  concatenated one-hot code of (token, position) — the book's "embedding = one-hot ×
  matrix" remark (book p. 43) extended to both tables.
- **Rank and capacity.** The set of all possible inputs is $\{E_{\text{tok}}[i]+E_{\text{pos}}[t]\}$,
  i.e. $V\cdot L$ points, but they are produced by only $(V+L)\,d$ parameters. The
  interaction between word and position is *additive*; any non-additive interplay
  must be built later by attention and the [[feed-forward-network]].
- **Gradients.** Since $\partial\mathbf{x}_t/\partial E_{\text{tok}}[\text{id}_t] = I$ and
  $\partial\mathbf{x}_t/\partial E_{\text{pos}}[t] = I$, the upstream gradient $\mathbf{g}_{b,t}$
  flows unchanged to both rows. Token row $i$ collects $\sum_{(b,t):\,\text{id}_{b,t}=i}\mathbf{g}_{b,t}$;
  position row $t$ collects $\sum_b \mathbf{g}_{b,t}$ (the adjoint of [[broadcasting]]).
  Rows of tokens absent from a batch get zero gradient.
- **Same width in and out.** Because the LLM returns one $d$-vector per input
  $d$-vector (book p. 98), $d$ is fixed throughout the residual stream; the input
  embedding sets that width.

## Connections
- Builds on: [[token-embedding]], [[positional-embedding]], [[embedding-layer]], [[token-id]], [[tokenization]], [[vector]], [[tensor]]
- Leads to: [[self-attention]], [[transformer-block]], [[gpt-architecture]], [[decoder-only-transformer]]
- Uses: [[broadcasting]], [[one-hot-encoding]], [[pytorch-dataset-and-dataloader]], [[sliding-window-sampling]]
- Related: [[absolute-positional-embedding]], [[embedding-dimension]], [[dropout]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7, book p. 43 (embedding = one-hot × matrix note).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 45–49 (Fig. 2.18 p. 45; code and shapes pp. 46–47; Fig. 2.19 p. 48; summary p. 49).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 95–98 (GPT-2 124M config; `DummyGPTModel.forward`; Fig. 4.4).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book p. 119 (`GPTModel.forward`, same embedding code).
- One-hot, rank and gradient arguments: standard derivations, not stated in the book.
