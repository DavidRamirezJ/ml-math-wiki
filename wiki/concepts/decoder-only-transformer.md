---
title: Decoder-Only Transformer
type: concept
tags: [llm, transformer, architecture, deep-learning, nlp]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model]
created: 2026-10-13
updated: 2026-10-13
---

# Decoder-Only Transformer

> **One-liner:** A decoder-only transformer is the "write the next word" half of the original transformer, used on its own: it reads the text so far and guesses one more word, again and again.

## 🟢 Intuition (Dummy)
The original [[transformer-architecture|transformer]] was built for translation, and it had
two halves. The **encoder** read the whole English sentence. The **decoder** wrote the German
sentence one word at a time. People at OpenAI asked a simple question: what if we throw away
the encoder and keep only the writer?

That is a decoder-only transformer. Think of a friend who plays "finish my sentence". You say
"Every effort moves you", and they say "forward". Then you add "forward" to your sentence and
ask again. Each new word goes back in as input for the next round. The friend can only look
**backwards** at the words already there, never ahead. That is the whole trick, and it is the
design behind GPT, ChatGPT and Meta's Llama.

Why is that enough? Because "guess the next word" sounds easy, but to do it well on billions of
sentences the model has to pick up grammar, facts and even how to translate. One simple job, one
simple architecture, many skills (see [[emergent-behavior]]).

## 🟡 Mechanics (Practitioner)

### What "decoder-only" means in the book
The book says that, compared with the original transformer, "the general GPT architecture is
relatively simple. Essentially, it's just the decoder part without the encoder" (§1.6, book
p. 12). Models like this generate "text by predicting text one word at a time", so they are an
**autoregressive model**: they "incorporate their previous outputs as inputs for future
predictions" (book pp. 12–13). Figure 1.8 shows three iterations. Each one is a stack
**Input text → Preprocessing steps → Decoder → Output layers**, and "the output of the previous
round serves as input to the next round" (book p. 13). The caption adds that the design is
**unidirectional, left-to-right** (book p. 13).

### The formula behind "one word at a time"
The book explains this in words. Here it is in standard notation (not printed in the book). A
text of $T$ [[token|tokens]] gets a probability by the chain rule of probability:

$$P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\dots,x_{t-1})$$

Reading it symbol by symbol:
- $x_1,\dots,x_T$ = the tokens of the text, in order ($x_1$ is the first token, $x_T$ the last).
- $P(x_1,\dots,x_T)$ = the probability of the whole text.
- $\prod_{t=1}^{T}$ = "multiply together, for $t=1,2,\dots,T$" (capital pi means product).
- $P(x_t\mid x_1,\dots,x_{t-1})$ = "the probability of token $x_t$, **given** (the bar $\mid$) all
  the tokens before it". This is exactly the [[next-token-prediction]] job.

So the decoder never needs to see the future: each factor only uses the past.

The Chapter 2 opening restates the focus of the whole book: "decoder-only LLMs based on the
transformer architecture, which underlies the models used in ChatGPT and other popular GPT-like
LLMs" (book p. 17). The pretraining signal is self-supervised: "we can use the next word in a
sentence or document as the label" (§1.6, book p. 12) — see [[self-supervised-learning]].

### The pipeline (what goes in, what comes out)
Figures 2.4 and 2.19 (book pp. 21, 48) and 3.6 (book p. 55) draw the whole thing inside a box
labeled "GPT-like decoder-only transformer":

1. Input text "This is an example." → [[tokenization|tokenized]] into This / is / an / example / .
2. [[token-id|Token IDs]]: 40134, 2052, 133, 389, 12 (book p. 48).
3. [[token-embedding|Token embeddings]] **plus** [[absolute-positional-embedding|positional embeddings]]
   = [[input-embeddings]]. GPT learns its positional embeddings during training, unlike the fixed
   [[sinusoidal-positional-encoding|positional encodings]] of the original transformer (§2.8, book p. 45).
4. A stack of [[transformer-block|transformer blocks]] with **masked** ([[causal-attention|causal]])
   [[multi-head-attention]] (ch. 3–4).
5. Output layers → [[logits]] → postprocessing → output text.

In formula form, the input at position $t$ is (notation derived from the book's code, §4.1):

$$\mathbf{x}_t = E_{\text{tok}}[\text{id}_t] + E_{\text{pos}}[t]$$

- $\mathbf{x}_t$ = the [[vector]] fed into the first block at position $t$.
- $E_{\text{tok}}$ = the token-embedding [[matrix]], one row per vocabulary entry; $E_{\text{tok}}[\text{id}_t]$ = "take the row for this token ID".
- $E_{\text{pos}}[t]$ = the row of the positional-embedding matrix for position $t$.

### What makes it "decoder" and not "encoder": the causal mask
The left-to-right rule is enforced inside attention. In causal attention "we mask out the
attention weights above the diagonal" (§3.5, Fig. 3.19, book p. 74), so token $t$ can only
attend to tokens $1,\dots,t$. In matrix form (standard notation): the
[[attention-scores|score]] matrix $\Omega = QK^\top/\sqrt{d_k}$ gets $-\infty$ in every entry
$(i,j)$ with $j>i$ before the [[softmax]]:

- $Q$, $K$ = the [[query-key-value|query and key]] matrices (one row per token).
- $K^\top$ = the [[matrix-transpose|transpose]] of $K$; $QK^\top$ is all pairwise [[inner-product|dot products]].
- $\sqrt{d_k}$ = square root of the key dimension, a scaling factor ([[scaled-dot-product-attention]]).
- $j>i$ = "key position is in the future of query position" → set to $-\infty$ → softmax gives weight $0$.

An encoder (like [[bert]]) has **no** such mask: every token sees every other token.

### The book's concrete numbers
| Model | Blocks (layers) | Embedding size | Parameters | Source |
|---|---|---|---|---|
| Original transformer | encoder and decoder blocks each repeated **6** times | — | — | book p. 13 |
| GPT-2 small (`GPT_CONFIG_124M`) | **12** layers, **12** heads | **768** | **124M** (first reported as 117M) | book pp. 93, 95 |
| GPT-3 (largest) | **96** transformer layers | **12,288** | **175B** | book pp. 13, 20 |

The full GPT-2 small config (Listing in §4.1, book p. 95):

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}
```

Run on the batch `[[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]` ("Every effort moves
you", "Every day holds a"), the model returns logits of shape `torch.Size([2, 4, 50257])`: 2
texts × 4 tokens × one score per vocabulary entry (book pp. 98–99). Note: **as many outputs as
inputs** — output $t$ is the guess for token $t+1$ (Fig. 4.4, book p. 98).

### The generation loop (book §4.7, p. 124)
This is Figure 1.8 turned into code — feed the text, keep only the last position's logits,
pick the most likely token ([[greedy-decoding]]), append, repeat:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]      # crop to the context length
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]              # last time step only
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)  # output becomes input
    return idx
```

## 🔴 Theory (Expert)
- **Same network, all prefixes at once.** Because of the causal mask, one forward pass over
  $T$ tokens computes all $T$ conditionals $P(x_t\mid x_{<t})$ in parallel. Training with
  [[cross-entropy-loss]] on all positions is therefore [[maximum-likelihood-estimation]] of the
  factorised joint above: $\max_\theta \sum_t \log P_\theta(x_t\mid x_{<t})$. Generation, in
  contrast, is inherently sequential (one token per forward pass). (Standard result; the book
  implements the loss in ch. 5.)
- **Why drop the encoder?** An encoder–decoder model learns $P(\mathbf y\mid\mathbf x)$ for a
  separate source $\mathbf x$. A decoder-only model treats the source as a **prefix** of one
  sequence: $P(\mathbf y\mid\mathbf x)=\prod_t P(y_t\mid \mathbf x, y_{<t})$ with $\mathbf x$
  simply placed first. Any task that can be written as "text in, text out" fits one model — the
  route to [[zero-shot-learning]], [[few-shot-learning]] and [[emergent-behavior]] (book
  pp. 13–14).
- **Bidirectional vs. unidirectional.** The mask restricts each position's attention to a
  lower-triangular pattern; an encoder like [[bert]] uses full bidirectional attention and is
  trained by [[masked-language-modeling]] instead (§1.4, book pp. 8–9). Bidirectional context is
  better for classification, unidirectional is what makes left-to-right sampling consistent with
  training.
- **Positions.** Self-attention itself is permutation-equivariant; order information comes only
  from the [[positional-embedding]] added at the input (§2.8). The causal mask also leaks some
  order information, but GPT still uses learned absolute positions, capped at
  `context_length` = 1,024 for GPT-2.
- **Scale and lineage.** GPT-3 is "fundamentally the same architecture" as GPT-2, scaled from
  1.5B to 175B parameters (book p. 94); Llama models keep "the same underlying concepts,
  introducing only minor modifications" (book p. 13). Typical modifications are [[rmsnorm]]
  instead of [[layer-normalization]] and [[swiglu]] instead of [[gelu]]. The book names
  SwiGLU as a smooth alternative activation (§4.3, book p. 105) and RMSNorm as "a popular
  variant of LayerNorm used in modern LLMs" (Appendix B, book pp. 292–293), but it does not
  tie them to Llama; that attribution is external context.
- **Not all LLMs are transformers.** The sidebar "Transformers vs. LLMs" notes recurrent and
  convolutional LLM alternatives aimed at efficiency (book p. 10); the decoder-only transformer
  is the book's working definition of "LLM".

## Connections
- Builds on: [[transformer-architecture]], [[encoder-decoder-architecture]], [[self-attention]], [[causal-attention-mask]], [[softmax]], [[inner-product]], [[matrix]], [[probability-distribution]], [[chain-rule]]
- Leads to: [[gpt-architecture]], [[gpt-2]], [[gpt-3]], [[autoregressive-text-generation]], [[next-token-prediction]], [[emergent-behavior]]
- Contrast with: [[bert]] (encoder-only), [[masked-language-modeling]]
- Used by: [[large-language-model]], [[pretraining]], [[fine-tuning]], [[llm-building-roadmap]]
- People: [[alec-radford]], [[ashish-vaswani]], [[tom-brown]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 7–10 (encoder vs decoder, BERT vs GPT).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book pp. 12–14 (decoder-only, autoregressive, Fig. 1.8, 6 vs 96 layers, 175B).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7 + Ch. 1 Summary, book pp. 14–16.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 2 intro + §2.1, book pp. 17–20 (focus on decoder-only LLMs; 768 / 12,288 dims).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.2, book p. 21 (Fig. 2.4) and §2.8, book pp. 43–49 (Fig. 2.19, learned absolute positions).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.2, book pp. 54–55 (Fig. 3.6); §3.5, book p. 74 (Fig. 3.19 causal mask).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 93–99 (`GPT_CONFIG_124M`, Fig. 4.4, logits shape); §4.7, book p. 124 (`generate_text_simple`).
