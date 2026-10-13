---
title: Context Length (Context Window)
type: concept
tags: [llm, transformer, hyperparameter, tokenization]
difficulty: beginner
sources: [llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Context Length (Context Window)

> **One-liner:** The context length is the maximum number of tokens an LLM can look at in one go; anything older than that falls off the edge.

*Also called:* context size, context window, `context_length`, `max_length`, `block_size`.

## 🟢 Intuition (Dummy)
Think of reading a book through a little window that only shows a fixed number of words. You can
use everything inside the window to guess the next word, but you can't see what's outside it. An
LLM works the same way. GPT-2's window is 1,024 [[token|tokens]] wide. If you paste in a longer text,
the model has to drop the oldest part (or you have to cut it before feeding it in).

Why is there a limit at all? Two reasons. First, GPT learns one position vector per slot (slot 0,
slot 1, … slot 1,023), so it literally has no vector for slot 1,024. Second, attention compares
every token with every other token, so a longer window gets expensive fast. Bigger window = more
memory and more compute.

## 🟡 Mechanics (Practitioner)

### 1. In data preparation: the input window (§2.6)
The book first meets context size when it builds training pairs: "the context size determines how
many tokens are included in the input" (book p. 36). With `context_size = 4`:
```python
context_size = 4
x = enc_sample[:context_size]        # [290, 4920, 2241, 287]
y = enc_sample[1:context_size+1]     # [4920, 2241, 287, 257]
```
For a window starting at index $s$ with length $L$:
$$x = (t_s, \dots, t_{s+L-1}), \qquad y = (t_{s+1}, \dots, t_{s+L})$$
- $t_k$ = the [[token-id]] at position $k$ of the tokenized text.
- $s$ = where the window starts; $L$ = the context length (`max_length` in the data loader).
- $y$ = the same window shifted one step right: the targets for [[next-token-prediction]].

One window holds $L$ prediction tasks (`[290] → 4920`, `[290, 4920] → 2241`, …). The data loader
`create_dataloader_v1(..., max_length=256, stride=128)` cuts the whole text into windows of this
length (book p. 38); see [[sliding-window-sampling]]. Exercise 2.2 plays with `max_length=2` and
`8` (book p. 39; answers in appendix C, p. 301).

### 2. In the model: the positional table (§2.8, §4.1)
```python
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))   # shape [4, 256]
```
"The context_length is a variable that represents the supported input size of the LLM" (book
p. 47). "Input text can be longer than the supported context length, in which case we have to
truncate the text" (book p. 47). In math:
$$E_{\text{pos}}\in\mathbb{R}^{L\times d}$$
- $E_{\text{pos}}$ = the positional [[embedding-layer]] weight; $L$ = context length (rows);
  $d$ = [[embedding-dimension]] (columns). One row per allowed position, so positions $\ge L$
  simply don't exist.

GPT-2 124M ([[gpt-model-configuration]], book p. 95):
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,  # max number of input tokens, via the positional embeddings
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```
So `pos_emb = nn.Embedding(1024, 768)`.

### 3. In attention: the size of the causal mask (§3.5–3.6)
`CausalAttention` and `MultiHeadAttention` take `context_length` and register a
`context_length × context_length` upper-triangular mask:
```python
self.register_buffer('mask',
    torch.triu(torch.ones(context_length, context_length), diagonal=1))
```
In `forward` it is sliced to the actual sequence length, `self.mask.bool()[:num_tokens, :num_tokens]`,
so shorter inputs work too (book p. 81; same pattern in `MultiHeadAttention`, pp. 86–87). Exercise 3.3 builds a GPT-2-sized module with
`block_size = 1024` (appendix C names the argument `block_size`, p. 301). See
[[causal-attention-mask]].

### 4. In generation: cropping (§4.7)
```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]    # keep only the last context_size tokens
        ...
```
"if LLM supports only 5 tokens, and the context size is 10, then only the last 5 tokens are used
as context" (code annotation, book p. 124). In math the model at step $t$ sees
$$\mathbf{x}_{\max(1,\,t-C+1):t}$$
- $C$ = `context_size`; $t$ = the current length of the text; the subscript range = "the last
  $C$ tokens, or all of them if there are fewer than $C$". See [[autoregressive-text-generation]].

### 5. In training: shrinking it to save compute (§5.1)
For pretraining on a laptop the book changes **only one value**: `"context_length": 256` instead
of 1,024. "This modification reduces the computational demands of training the model" (book p.
130). The data loaders then use `max_length = stride = GPT_CONFIG_124M["context_length"]` (book p. 143). When
the OpenAI weights are loaded in §5.5, it is set back to 1,024 because the pretrained positional
table `wpe` has 1,024 rows ("the original GPT-2 models from OpenAI were trained with a 1,024-token length", book p. 164).

### 6. In fine-tuning: the hard cap for padding and truncation (§6.3, §7.3–7.4)
- Spam classification: the longest training message is **120** tokens, so all sets are padded or
  truncated to 120 (book p. 178). "The model can handle sequences of up to 1,024 tokens, given its
  context length limit"; for longer texts pass `max_length=1024` (book p. 178).
  See [[padding-and-truncation]].
- **Exercise 6.1** (book p. 179): pad everything to 1,024. Result (appendix C, p. 306): test
  accuracy drops to **78.33%** from **95.67%**. Lots of padding hurts here.
- Instruction fine-tuning: `custom_collate_fn(..., allowed_max_length=None)` can truncate to "the
  1,024-token context size supported by GPT-2" (book p. 218), and the data loaders fix
  `allowed_max_length=1024` (book p. 224). Batch lengths like 61, 76, 73 (book p. 225) are all $\le 1024$.
  Appendix C suggests lowering it to 512 or 256 if you run out of memory (book p. 310).
  See [[collate-function]], [[dynamic-padding]].

## 🔴 Theory (Expert)
- **Why GPT's limit is hard.** With *learned* [[absolute-positional-embedding|absolute positional
  embeddings]], the model has parameters only for positions $0..L-1$. It cannot extrapolate to
  position $L$. [[relative-positional-embedding|Relative]] schemes "generalize better to sequences
  of varying lengths, even if it has not seen such lengths during training" (book p. 45), and the
  original transformer used fixed [[sinusoidal-positional-encoding|sinusoidal encodings]].
- **Cost scales with $L^2$.** The attention-score matrix $QK^\top$ is $L\times L$ per head (and
  so is the mask buffer), so memory and compute for attention grow quadratically in context
  length, while the rest of the block grows linearly. That's the reason behind the 1,024 → 256
  cut in chapter 5 (the quadratic argument is standard; the book only says "computational
  demands"). See [[scaled-dot-product-attention]] and [[flashattention]].
- **Parameters.** Context length only adds $L\cdot d$ parameters (GPT-2: $1024\times768 =
  786{,}432$), tiny next to the token table ($50{,}257\times768$). The real price is activations.
- **Factorization limit.** A model with window $C$ approximates
  $P(x_{t+1}\mid x_{1:t}) \approx P(x_{t+1}\mid x_{t-C+1:t})$: it is a (very large) order-$C$
  Markov model over tokens. Anything earlier is invisible.
- **Context vs. training length.** The input length can be smaller than the model's
  `context_length` (e.g. the 120-token spam batches in §6.3 vs. GPT-2's 1,024; in the §2.8 demo the two
  are equal, `context_length = max_length = 4`) but never larger without truncation.

## Connections
- Builds on: [[token]], [[tokenization]], [[token-id]], [[sliding-window-sampling]]
- Leads to: [[positional-embedding]], [[absolute-positional-embedding]], [[causal-attention-mask]], [[autoregressive-text-generation]], [[padding-and-truncation]]
- Used by: [[gpt-model-configuration]], [[gpt-2]], [[pytorch-dataset-and-dataloader]], [[collate-function]], [[classification-fine-tuning]], [[instruction-fine-tuning]]
- Related: [[hyperparameter]], [[embedding-layer]], [[self-attention]], [[long-range-dependencies]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book pp. 35–41 (context size 4, input/target shift p. 36, `max_length` in the data loader p. 38, Exercise 2.2 p. 39).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.8, book pp. 45–47 (positional embedding layer, "supported input size", truncation, p. 47).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5, book p. 81 and §3.6, pp. 86–87 (mask buffer sized by `context_length`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book p. 95 (`context_length: 1024`); §4.7, book pp. 124–126 (cropping in `generate_text_simple`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 130, 143 (reduced to 256 tokens; loaders use it as `max_length`/`stride`); §5.5, book p. 164 (reset to 1,024 for the OpenAI weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.3, book pp. 178–179 (120-token padding, 1,024 limit, Exercise 6.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.3, book p. 218 and §7.4, book pp. 224–225 (`allowed_max_length=1024`, batch lengths 61/76/73).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 301, 306, 310 (Exercises 2.2, 3.3, 6.1 solutions; 78.33% vs 95.67%; lowering `allowed_max_length` to 512/256).
