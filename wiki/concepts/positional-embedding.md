---
title: Positional Embeddings
type: concept
tags: [llm, embeddings, transformer, attention, linear-algebra]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Positional Embeddings

> **One-liner:** A positional embedding is a vector that tells the model *where* a token sits in the sentence, so "dog bites man" and "man bites dog" stop looking the same.

## 🟢 Intuition (Dummy)
Imagine you cut a sentence into word cards and drop them in a bag. Every card
says what the word is, but nothing says what order the cards were in. If you
hand that bag to someone, "dog bites man" and "man bites dog" are the exact same
bag of cards.

That is the problem an LLM has. Its [[token-embedding]] layer is a lookup table:
the word "fox" always gets the very same vector, whether it is the first word or
the fourth (book p. 44, Fig. 2.17). And the [[self-attention]] machinery that
comes later doesn't care about order either — the book calls it
*position-agnostic* (book p. 45). The book notes that a fixed, position-independent
lookup is good for reproducibility — the problem is only that nothing else in the
model supplies order, so position information must be injected (book p. 45).

The fix is simple and kind of charming: give every *seat* in the sentence its own
little name-tag vector, and stick it onto the word's vector. Seat 0 gets one tag,
seat 1 gets a different tag, and so on. Now "fox in seat 0" and "fox in seat 3"
are different vectors, and the model can tell order apart.

There are two big families of these tags (book p. 45):
- [[absolute-positional-embedding|Absolute]] — "you are in seat number 3."
- [[relative-positional-embedding|Relative]] — "you are 2 seats to the right of that other word."

GPT models use the absolute kind, and they *learn* the tags during training
instead of using a fixed formula (book p. 45).

## 🟡 Mechanics (Practitioner)

### The core formula
For the token at position $t$ in a sequence, the vector actually fed into the
model (the [[input-embeddings|input embedding]]) is

$$\mathbf{x}_t = E_{\text{tok}}[\,\text{id}_t\,] + E_{\text{pos}}[\,t\,], \qquad t = 0, 1, \dots, L-1$$

Reading it symbol by symbol:
- $\mathbf{x}_t$ = the input embedding at position $t$ (a [[vector]] of $d$ numbers).
- $t$ = the position (seat number), counting from 0 like Python does.
- $\text{id}_t$ = the [[token-id]] of the word sitting at position $t$.
- $E_{\text{tok}}$ = the token-embedding weight [[matrix]], size $V \times d$ ($V$ = [[vocabulary]] size, $d$ = [[embedding-dimension]]).
- $E_{\text{tok}}[\text{id}_t]$ = "take row number $\text{id}_t$ of that matrix" — the word's vector.
- $E_{\text{pos}}$ = the positional-embedding matrix, size $L \times d$ ($L$ = [[context-length]]).
- $E_{\text{pos}}[t]$ = "take row $t$" — the seat's name-tag vector.
- $+$ = ordinary element-wise vector addition. That only works because both vectors have the **same length $d$** (book p. 45, Fig. 2.18).

This formula is a faithful write-up of the book's code; the book itself shows it
as a picture and code, not as an equation.

### The book's toy picture (Fig. 2.18, book p. 45)
Token embeddings are drawn as all-ones (just to keep it simple), and positions get
easy-to-spot vectors:

| Position | Token emb. | + Positional emb. | = Input emb. |
|---|---|---|---|
| 0 | [1, 1, 1] | [1.1, 1.2, 1.3] | [2.1, 2.2, 2.3] |
| 1 | [1, 1, 1] | [2.1, 2.2, 2.3] | [3.1, 3.2, 3.3] |
| 2 | [1, 1, 1] | [3.1, 3.2, 3.3] | [4.1, 4.2, 4.3] |
| 3 | [1, 1, 1] | [4.1, 4.2, 4.3] | [5.1, 5.2, 5.3] |

Same word, four seats, four different input vectors. That's the whole trick.

### In PyTorch (book p. 47)
The positional embedding is just *another* `nn.Embedding` layer, but it is
indexed by position numbers instead of token IDs:

```python
context_length = max_length            # 4 in the book's demo
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # 4 x 256
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)            # torch.Size([4, 256])

input_embeddings = token_embeddings + pos_embeddings   # [8,4,256] + [4,256]
print(input_embeddings.shape)          # torch.Size([8, 4, 256])
```

- `torch.arange(context_length)` = the placeholder vector $[0, 1, \dots, L-1]$ — the seat numbers (book p. 47).
- The `[4, 256]` positional tensor gets added to each of the 8 sequences in the batch. PyTorch does this automatically through [[broadcasting]] (the book describes the effect, book p. 47).

### Inside the full GPT model (book pp. 95–97, 119)
In `GPTModel` the same idea appears in `forward`:

```python
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])   # 1024 x 768
...
pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
x = tok_embeds + pos_embeds
```

With the [[gpt-model-configuration|GPT-2 124M config]] (`context_length` = 1,024,
`emb_dim` = 768), the positional table has $1024 \times 768 = 786{,}432$
learned numbers (our arithmetic, not printed in the book). The `device=` argument
just puts the position indices on the same CPU/GPU as the input (book p. 119).

### The positional table *is* the context limit
Because there are only $L$ rows, the model physically cannot look up seat
$L$ or beyond. The book uses this directly:
- `context_size = model.pos_emb.weight.shape[0]` in `generate_and_print_sample` (book p. 148).
- `supported_context_length = model.pos_emb.weight.shape[0]` in `classify_review`, which [[padding-and-truncation|truncates]] longer inputs (book p. 201).
- When loading OpenAI's GPT-2 weights, `context_length` must be switched from 256 back to 1,024, because OpenAI's positional table `wpe` is $1024 \times 768$; the loader copies `params['wpe']` into `gpt.pos_emb.weight` (book pp. 164–165).

So "context length" is not some vague setting — it's literally the number of
rows in this matrix. Inputs longer than that must be cut (book p. 47).

## 🔴 Theory (Expert)

**Why attention needs it: permutation equivariance.** Let $X \in \mathbb{R}^{T\times d}$
hold the token vectors as rows and $\Pi$ be any $T\times T$ permutation matrix.
Unmasked self-attention $\mathrm{Attn}(X) = \mathrm{softmax}\!\big(XW_q (XW_k)^\top/\sqrt{d_k}\big)\,XW_v$
satisfies $\mathrm{Attn}(\Pi X) = \Pi\,\mathrm{Attn}(X)$: shuffle the inputs and the
outputs just shuffle the same way. Each output is a function of the *multiset* of
inputs, not their order. Adding $P = E_{\text{pos}}[0{:}T]$ breaks this symmetry,
because $\Pi X + P \neq \Pi(X + P)$ in general. (This is the formal version of the
book's "position-agnostic", book p. 45.)

*Nuance:* the [[causal-attention-mask]] used in GPT already breaks full
permutation symmetry (token $t$ only sees tokens $\le t$), so a causal model has a
weak implicit sense of position even without positional embeddings. Explicit
positional embeddings remain the standard, and they are what the book uses.

**Additive injection and geometry.** Adding $\mathbf{p}_t$ is a translation of the
token vector in $\mathbb{R}^d$ — loosely an [[affine-space]] shift. Addition keeps the
width at $d$ (concatenation would grow it to $d + d_p$). In high dimension, learned
token and position components can occupy nearly [[orthogonality|orthogonal]]
directions, so the downstream linear maps $W_q, W_k, W_v$ can still pull them apart:
$(\mathbf{e} + \mathbf{p})W = \mathbf{e}W + \mathbf{p}W$ by [[linear-mapping|linearity]].
Expanding a query–key [[inner-product]] gives four terms — token·token,
token·position, position·token, position·position — which is how attention can
learn order-sensitive patterns.

**Learned vs fixed.** GPT's table is learned by [[backpropagation]] like any other
parameter (book pp. 45, 49); the gradient into row $t$ is the sum of the upstream
gradients at position $t$ over the batch (see [[broadcasting]] — the adjoint of a
broadcast is a sum). The original transformer instead used fixed
[[sinusoidal-positional-encoding|sinusoidal encodings]] (book p. 45 mentions them,
without the formula). Fixed encodings are defined for any $t$; a learned absolute
table has no row beyond $L-1$, so it cannot extrapolate.

**Absolute vs relative.** [[absolute-positional-embedding|Absolute]] schemes encode
"where am I"; [[relative-positional-embedding|relative]] schemes encode "how far
apart are we", which the book says generalizes better to unseen lengths (book p. 45).

## Connections
- Builds on: [[token-embedding]], [[embedding-layer]], [[vector]], [[matrix]], [[embedding-dimension]], [[context-length]]
- Leads to: [[input-embeddings]], [[self-attention]], [[gpt-architecture]]
- Types: [[absolute-positional-embedding]], [[relative-positional-embedding]], [[sinusoidal-positional-encoding]]
- Uses: [[broadcasting]], [[backpropagation]]
- Used by: [[gpt-2]], [[transformer-architecture]], [[decoder-only-transformer]], [[padding-and-truncation]]
- People: [[ashish-vaswani]] (original transformer encodings), [[alec-radford]] (GPT)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.7–2.8, book pp. 43–49 (Fig. 2.17 p. 44; Fig. 2.18 and absolute/relative/learned discussion p. 45; code pp. 46–47; Fig. 2.19 p. 48; summary p. 49).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 95–97 (`context_length` 1,024; `pos_emb` in `DummyGPTModel`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–119 (`GPTModel.forward`, `device=` argument).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.2, book p. 148 (`context_size = model.pos_emb.weight.shape[0]`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 164–165 (context 256 → 1,024; `wpe` → `pos_emb`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.8, book p. 201 (`supported_context_length` in `classify_review`).
- Permutation-equivariance argument and parameter count: standard derivation, not stated in the book.
