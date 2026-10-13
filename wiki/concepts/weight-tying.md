---
title: Weight Tying
type: concept
tags: [transformer, embeddings, parameters, architecture, llm]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Weight Tying

> **One-liner:** Weight tying means the model uses the *same* big table of numbers twice — once to turn tokens into vectors at the input, and again to turn vectors back into token scores at the output.

## 🟢 Intuition (Dummy)
Imagine a bilingual dictionary. At the start of the pipeline you look up a word and get
its "meaning card" (a list of 768 numbers). At the very end, the model has a meaning card
in hand and must ask "which word is this card closest to?"

You could print **two** dictionaries: one for word → card, another for card → word. Or
you could use the **same** dictionary both ways. That second choice is **weight tying**:
GPT-2 reuses its token-embedding table as its output layer (book p. 121).

Why bother? The table is huge — 50,257 words × 768 numbers ≈ 38.6 million numbers.
Sharing it saves that much memory. The catch: in the author's experience, two separate
tables train a bit better, so the book's own model keeps them separate, like most modern
LLMs (book p. 121).

## 🟡 Mechanics (Practitioner)

### The two matrices that can be shared
- The **token embedding** $E\in\mathbb R^{V\times d}$ (`tok_emb.weight`): row $i$ is the vector for token ID $i$ ([[embedding-layer]], [[token-embedding]]).
- The **output head** $W_{\text{out}}\in\mathbb R^{V\times d}$ (`out_head.weight` of `nn.Linear(emb_dim, vocab_size, bias=False)`): turns the final hidden vector into one score ([[logits|logit]]) per vocabulary entry.

For GPT-2 small, $V=50{,}257$ (vocabulary size) and $d=768$ (embedding dimension). The
book prints both shapes and they match (book p. 121):
```python
print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
# Token embedding layer shape: torch.Size([50257, 768])
# Output layer shape: torch.Size([50257, 768])
```

### The formula
Without tying:
$$\mathbf z_t = W_{\text{out}}\,\mathbf h_t$$
With tying, set $W_{\text{out}} = E$:
$$\mathbf z_t = E\,\mathbf h_t,\qquad z_{t,i} = \mathbf e_i^{\top}\mathbf h_t$$
Symbol by symbol:
- $\mathbf h_t\in\mathbb R^{768}$ = the final hidden vector at position $t$ (after the last [[transformer-block]] and the final [[layer-normalization]]).
- $E$ = the token-embedding [[matrix]]; $\mathbf e_i$ = its row $i$, the embedding of token $i$.
- $\mathbf z_t\in\mathbb R^{50257}$ = the vector of logits — one score per vocabulary token.
- $z_{t,i}=\mathbf e_i^\top\mathbf h_t$ = the [[inner-product|dot product]] of the hidden vector with token $i$'s embedding. **A token scores high when the model's output vector points in the same direction as that token's own embedding.**

(The dot-product reading is our addition; the book states only the reuse of the weights.)

### Counting the saving (§4.6, book pp. 120–121)
```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")       # 163,009,536

total_params_gpt2 = (total_params -
                     sum(p.numel() for p in model.out_head.parameters()))
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}")     # 124,412,160
```
- The output head has $50{,}257\times768 = 38{,}597{,}376$ weights (our arithmetic; no bias).
- $163{,}009{,}536 - 38{,}597{,}376 = 124{,}412{,}160$ — the famous "124M" of GPT-2 small. The name only makes sense **with** tying ([[parameter-count-and-memory]]).
- For GPT-2 XL (Exercise 4.2, appendix C, book p. 302): $1{,}637{,}792{,}000$ total, $1{,}557{,}380{,}800$ with tying; the difference is $50{,}257\times1{,}600=80{,}411{,}200$.

### Loading OpenAI's tied weights (§5.5, Listing 5.5, book p. 166)
OpenAI's checkpoint has **no separate output-head tensor**. Its parameter dictionary has
only the keys `['blocks', 'b', 'g', 'wpe', 'wte']` (book p. 162): the transformer blocks,
the final LayerNorm shift/scale (`b`, `g`), the positional embedding `wpe` and the token
embedding `wte`, whose shape is `(50257, 768)` (book p. 163). So the last line of
`load_weights_into_gpt` copies the token embedding into the head:
```python
gpt.tok_emb.weight  = assign(gpt.tok_emb.weight,  params['wte'])
...
gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
```
The book's margin note: "The original GPT-2 model by OpenAI reused the token embedding
weights in the output layer to reduce the total number of parameters, which is a concept
known as weight tying" (book p. 166).

Small note: in §4.6 the book says it will revisit weight tying "in chapter 6"; it actually
happens in §5.5 (book pp. 121 vs 166).

**Subtle point (our observation, from the code).** `assign` wraps each value in a *new*
`torch.nn.Parameter(torch.tensor(right))`. So after loading, `tok_emb.weight` and
`out_head.weight` hold **equal values but are two separate parameters**. If you then
fine-tune, they can drift apart. True tying would share one tensor object, e.g.
`gpt.out_head.weight = gpt.tok_emb.weight`.

### Trade-off (book p. 121)
- **Pro:** "reduces the overall memory footprint and computational complexity of the model."
- **Con:** "in my experience, using separate token embedding and output layers results in better training and model performance … The same is true for modern LLMs."

## 🔴 Theory (Expert)
- **Origins.** Press & Wolf (2017), "Using the Output Embedding to Improve Language Models",
  and Inan, Khosravi & Socher (2017) proposed tying input and output embeddings for
  language models (external; not cited in the book). GPT-2 inherited the idea.
- **Gradient view.** With tying, $E$ receives two gradient contributions each step: a
  sparse one from the input lookup (only rows of tokens in the batch) and a dense one
  from the output softmax (every row, via
  $\partial\mathcal L/\partial\mathbf e_i = (p_{t,i}-\mathbb 1[i=y_t])\,\mathbf h_t$ under
  [[cross-entropy-loss]]). Rare tokens therefore still get updated through the output
  side, which is one argued benefit for small models.
- **Geometric constraint.** Tying forces "input similarity" and "output similarity" to be
  measured in one space. Token $i$'s logit is $\langle\mathbf e_i,\mathbf h_t\rangle$, so
  the softmax is a similarity search over the embedding table
  ([[inner-product]], [[softmax]]). A token's own embedding then tends to score highly as
  the next token, a bias the model must learn to suppress — one reason untied models
  can do better at scale.
- **When it pays.** The saving is $Vd$ parameters. This is a large fraction of the total
  when the model is small (the vocabulary $V$ is large compared with the per-layer
  parameters $\approx 12d^2$ times the depth $L$) (23.7% of GPT-2 small's untied 163M, but only 4.9% of GPT-2 XL's
  1.64B — our arithmetic). Hence its use in small and on-device models, and its
  abandonment in many large LLMs, as the book notes.
- **Linear-algebra reading.** Without tying, the model is free to learn any linear map
  $\mathbb R^{d}\to\mathbb R^{V}$ at the output ([[linear-mapping]]). Tying constrains that
  map to be the transpose of the input lookup: input is $\mathbf e_i^\top=\mathbf u_i^\top E$
  (one-hot $\mathbf u_i$, [[one-hot-encoding]]) and output is $E\mathbf h$ — the same
  matrix used in both directions ([[matrix-transpose]]).

## Connections
- Builds on: [[embedding-layer]], [[token-embedding]], [[linear-layer]], [[logits]], [[inner-product|dot product]], [[matrix]]
- Leads to: [[parameter-count-and-memory]], [[pretrained-weights]]
- Related: [[output-projection]], [[gpt-model-configuration]], [[vocabulary]], [[embedding-dimension]], [[softmax]], [[cross-entropy-loss]]
- Used by: [[gpt-2]], [[gpt-architecture]]
- People: [[alec-radford]] (GPT-2), [[sebastian-raschka]] (untied-is-better remark)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 120–122 (parameter count 163,009,536; shapes [50257, 768]; 124,412,160; trade-off paragraph).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 162–166 (params keys, `wte` shape, Listing 5.5, `out_head.weight ← wte`, weight-tying margin note).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, Exercise 4.2, book p. 302 (GPT-2 XL 1,637,792,000 / 1,557,380,800).
- Press & Wolf / Inan et al. citations, gradient and geometry remarks: external, not in the book.
