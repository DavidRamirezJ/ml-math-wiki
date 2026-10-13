---
title: GPT Architecture (Generative Pretrained Transformer)
type: concept
tags: [llm, transformer, gpt, deep-learning, architecture]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# GPT Architecture (Generative Pretrained Transformer)

> **One-liner:** A GPT turns tokens into vectors, pushes them through a stack of identical transformer blocks, and turns the last vector into a score for every possible next token. Then it repeats, one token at a time.

## 🟢 Intuition (Dummy)
Imagine an assembly line for words:
1. **Lookup desk.** Each word piece ([[token]]) gets swapped for a list of 768 numbers (its
   embedding), plus a second list that says *where* it sits in the sentence.
2. **Twelve identical workstations** ([[transformer-block|transformer blocks]]). At each one, every
   token first "looks back" at the earlier tokens to pick up context (attention), then gets
   polished on its own (a small feed-forward network). The vectors go in and come out the same size,
   so you can stack as many stations as you like.
3. **Scoreboard.** At the end, the last token's vector is turned into 50,257 scores, one for every
   word piece in the vocabulary. The highest score (or a random draw weighted by the scores) is the
   next token.
4. Glue that token onto the input and run the line again.

"GPT" stands for **generative pretrained transformer** (book p. 93). It's *generative* because it
writes text, *pretrained* because it first learns from a huge pile of raw text by guessing the next
word, and a *transformer* because of the attention blocks. Compared with the original transformer,
GPT is simpler: it keeps **only the decoder** and drops the encoder (book p. 12). The whole model
looks big, but it's mostly one block copied over and over (book p. 93).

## 🟡 Mechanics (Practitioner)

### The configuration (GPT-2 small, "124M", book p. 95)
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocabulary
    "context_length": 1024,  # max tokens (positional embeddings)
    "emb_dim": 768,          # each token -> 768-dim vector
    "n_heads": 12,           # attention heads
    "n_layers": 12,          # transformer blocks
    "drop_rate": 0.1,        # 10% of hidden units dropped
    "qkv_bias": False        # no bias in query/key/value Linear layers
}
```
In chapter 5 the book trains with `context_length` cut to **256** so it runs on a laptop, then sets it
back to 1,024 to load OpenAI's weights (book pp. 130–131). Loading those weights also needs
`qkv_bias=True` (the chapter 6 printout shows bias on the Q/K/V layers, book p. 185).

### The data flow (Figure 4.15, book p. 118; Listing 4.7, book p. 119)
Bottom to top: token IDs → token embedding → + positional embedding → dropout → **12 ×
transformer block** → final LayerNorm → linear output head → logits.

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
```

The same thing as formulas (our notation, written from the code):

$$X^{(0)} = \mathrm{Dropout}\big(E_{\text{tok}}[\text{ids}] + E_{\text{pos}}[0{:}T]\big)$$
$$X^{(\ell)} = \mathrm{Block}_\ell\big(X^{(\ell-1)}\big),\quad \ell = 1,\dots,L$$
$$\text{logits} = \mathrm{LN}\big(X^{(L)}\big)\,W_{\text{out}}^{\top}$$

Symbol by symbol:
- $\text{ids}$ = the input token IDs, shape $[B,T]$ ($B$ = batch size, $T$ = number of tokens).
- $E_{\text{tok}}\in\mathbb R^{V\times d}$ = the token-embedding [[matrix]] ($V$ = 50,257 rows, $d$ = 768
  columns). $E_{\text{tok}}[\text{ids}]$ means "take the row for each ID" ([[embedding-layer]]).
- $E_{\text{pos}}\in\mathbb R^{1024\times d}$ = the learned [[absolute-positional-embedding|positional embedding]]
  matrix; $[0{:}T]$ = rows for positions $0$ to $T-1$. The $+$ is [[broadcasting|broadcast]] over the batch.
- $X^{(\ell)}\in\mathbb R^{B\times T\times d}$ = the hidden vectors after block $\ell$; $L$ = `n_layers` = 12.
- $\mathrm{LN}$ = the final [[layer-normalization]].
- $W_{\text{out}}\in\mathbb R^{V\times d}$ = the output-head weights (no bias); multiplying by
  $W_{\text{out}}^{\top}$ maps each 768-vector to 50,257 [[logits]].

**Shapes** (book pp. 119–120): `in_idx` $[B,T]$ → embeddings $[B,T,768]$ → blocks keep $[B,T,768]$ →
logits $[B,T,50257]$. Running it on the batch `[[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]`
("Every effort moves you" / "Every day holds a") with `torch.manual_seed(123)` gives
`torch.Size([2, 4, 50257])` (book p. 120).

### Inside one transformer block (Listing 4.6, book p. 115)
```python
def forward(self, x):
    shortcut = x
    x = self.norm1(x); x = self.att(x); x = self.drop_shortcut(x)
    x = x + shortcut                     # residual around attention
    shortcut = x
    x = self.norm2(x); x = self.ff(x); x = self.drop_shortcut(x)
    x = x + shortcut                     # residual around feed-forward
    return x
```
$$\mathbf h = \mathbf x + \mathrm{Dropout}\big(\mathrm{MHA}(\mathrm{LN}_1(\mathbf x))\big),\qquad
\mathbf y = \mathbf h + \mathrm{Dropout}\big(\mathrm{FFN}(\mathrm{LN}_2(\mathbf h))\big)$$

- $\mathbf x$ = the block input; $\mathbf h$ = the result after the attention half; $\mathbf y$ = the
  block output, the **same shape** as $\mathbf x$.
- $\mathrm{MHA}$ = masked [[multi-head-attention]]. It mixes information **across positions**, and each
  token sees only itself and earlier tokens ([[causal-attention]]).
- $\mathrm{FFN}(\mathbf z)=W_2\,\mathrm{GELU}(W_1\mathbf z+\mathbf b_1)+\mathbf b_2$ = the
  [[feed-forward-network]], applied to **each position separately** (768 → 3,072 → 768) with [[gelu]].
- $\mathrm{LN}_1,\mathrm{LN}_2$ = two separate LayerNorms; "$\mathbf x+\dots$" = the
  [[shortcut-connections]].
- LayerNorm comes *before* each sub-layer ([[pre-layernorm-vs-post-layernorm|Pre-LayerNorm]]); the
  original transformer put it after, which "often leads to worse training dynamics" (book p. 115).

`torch.rand(2, 4, 768)` in → `torch.Size([2, 4, 768])` out (book p. 116).

### Counting parameters and memory (book pp. 120–122)
```python
total_params = sum(p.numel() for p in model.parameters())   # 163,009,536
```
- Why 163 M and not 124 M? **[[weight-tying]]**: the original GPT-2 reuses the token-embedding matrix
  as the output head. Both are $50{,}257\times768 = 38{,}597{,}376$ parameters. Subtract one:
  $163{,}009{,}536 - 38{,}597{,}376 = 124{,}412{,}160$, which is the "124M" (book p. 121).
- The book keeps the two layers **separate**: in the author's experience that trains better, and the
  same is true for modern LLMs (book p. 121).
- Memory at float32 (4 bytes): $\dfrac{163{,}009{,}536\times4}{1024^2}\approx$ **621.83 MB** (book p. 122).
  ($N$ = parameter count, 4 = bytes per 32-bit float, $1024^2$ = bytes per MB.) See [[parameter-count-and-memory]].

### Model sizes from the same class (Exercise 4.2, book p. 122; p. 117)

| Model | emb_dim | blocks | heads | params |
|---|---|---|---|---|
| GPT-2 small | 768 | 12 | 12 | 124 M |
| GPT-2 medium | 1,024 | 24 | 16 | 345 M |
| GPT-2 large | 1,280 | 36 | 20 | 762 M |
| GPT-2 XL | 1,600 | 48 | 25 | 1,542 M |

(The 345/762 M sizes come from book pp. 93, 127.) [[gpt-3]] has the same architecture with 96 layers
and 175 B parameters (book pp. 13, 94).

### Turning logits into text (Listing 4.8, book p. 124)
```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]          # crop to context window
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                  # last position only
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)    # append and repeat
    return idx
```
$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}},\qquad \hat x_{t+1}=\arg\max_i p_i=\arg\max_i z_i$$
- $z_i$ = logit for vocabulary token $i$ at the last position; $p_i$ = its [[softmax]] probability;
  $V$ = 50,257.
- $\hat x_{t+1}$ = the chosen next token. Softmax is monotonic, so the argmax of the probabilities and
  the argmax of the logits are the same (book p. 125). This is [[greedy-decoding]].

"Hello, I am" → `[15496, 11, 314, 716]` → the untrained model returns
`[15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]` → "Hello, I am Featureiman
Byeswickattribute argue". It's gibberish because the weights are still random (book pp. 125–127).
Chapter 5 fixes this with [[pretraining]].

### Reusing the architecture
- **Save/load:** `torch.save(model.state_dict(), "model.pth")`, then build a new
  `GPTModel(GPT_CONFIG_124M)` and call `load_state_dict(...)`. To keep training, save the AdamW
  optimizer state too (book pp. 159–160; [[model-checkpointing]]).
- **Classification:** swap `out_head` for `nn.Linear(768, 2)`, freeze everything, then unfreeze the
  last block and `final_norm`. The output becomes $[1,4,2]$ and the last token's row is used, e.g.
  `[-3.5983, 3.9902]` for "Do you have time" (book pp. 183–188; [[classification-head]]).
- **LoRA:** wrap every `nn.Linear` (Q/K/V/out-proj, both FFN layers, `out_head`) as
  `LinearWithLoRA`. With rank = alpha = 16 there are 2,666,528 trainable parameters instead of
  124,441,346 (book pp. 330–332; [[low-rank-adaptation]]).

## 🔴 Theory (Expert)
- **Autoregressive factorization.** A GPT models $P_\theta(x_1,\dots,x_T)=\prod_t P_\theta(x_t\mid x_{<t})$
  (standard notation; the book calls decoder-style models *autoregressive*, book pp. 12–13). The
  [[causal-attention-mask]] makes position $t$'s output depend only on $x_{\le t}$. So *one* forward
  pass over $T$ tokens produces all $T$ conditionals at once. That's why training uses targets shifted
  by one and a loss averaged over every position ([[maximum-likelihood-estimation]]).
- **Residual stream view.** Each block computes $\mathbf y=\mathbf x+F(\mathbf x)$, so
  $\partial\mathbf y/\partial\mathbf x=I+\partial F/\partial\mathbf x$. The identity term keeps the
  gradient from shrinking through 12 (or 96) blocks. The book's 5-layer demo shows the first layer's
  mean gradient going from 0.0002 without shortcuts to 0.2217 with them (book pp. 112–113).
- **Shape invariance is the design.** Every block maps $\mathbb R^{T\times d}\to\mathbb R^{T\times d}$.
  That's what lets you stack blocks and grow from 12 to 48 to 96 just by changing `n_layers`. Each
  output keeps a one-to-one match with its input position, but its *content* is re-encoded with
  context from the whole sequence (book p. 116).
- **Tied output head = similarity scoring.** With $W_{\text{out}}=E_{\text{tok}}$, the logit for token $v$
  is the [[inner-product]] $\langle \mathbf h_t, \mathbf e_v\rangle$: the model scores each
  vocabulary item by how well its embedding lines up with the final hidden state. Untying the head
  removes that constraint at a cost of 38.6 M extra parameters.
- **Where the parameters live.** For GPT-2 small, per block: FFN $=768\cdot3072+3072+3072\cdot768+768
  =4{,}722{,}432$; attention (no QKV bias, with out-projection bias) $=3\cdot768^2+768^2+768=2{,}360{,}064$
  (our check for Exercise 4.1). So the feed-forward layers hold about 2× the attention parameters. Each
  untied embedding/head matrix (38.6 M) is larger than all 12 attention modules together (28.3 M).
- **Last token as a summary.** Under causal masking only the final position attends to every token, so
  its hidden state is the natural sequence summary for classification. The book's Figure 6.12 weights
  for "time" are 0.27 / 0.24 / 0.24 / 0.25 across all four tokens (book p. 190; [[last-token-pooling]]).
- **Lineage and modern tweaks.** The GPT paper is Radford et al., "Improving Language Understanding by
  Generative Pre-Training" (book p. 12). GPT-2 is Radford et al., "Language Models Are Unsupervised
  Multitask Learners" (book p. 93). Meta's Llama still uses the same basic ideas with minor changes
  (book p. 13). Appendix B points to [[rmsnorm]] (normalize by the root mean square only), the Pre-LN
  vs. Post-LN papers, and [[flashattention]] as the common upgrades (appendix B, pp. 291–293).
- **Not all LLMs are transformers.** Recurrent and convolutional LLMs (RWKV, Hyena, Mamba) exist,
  mainly for computational efficiency. Whether they can match transformers is still open (book p. 10).

> ⚠️ Contradiction (within the book): chapter 4 lists the GPT-2 sizes as **124, 345, 762, and 1,542
> million** parameters (book pp. 93, 127). Appendix B's summary of the GPT-2 paper lists **124M, 355M,
> 774M, and 1.5B** (book p. 293). The 117 M → 124 M correction for the smallest model is mentioned on
> book p. 93. We keep the chapter 4 numbers, which match the book's own `GPTModel` code, and note the
> appendix figures as the paper's reported sizes.

## Connections
- Builds on: [[transformer-architecture]], [[encoder-decoder-architecture]], [[self-attention]], [[multi-head-attention]], [[causal-attention]], [[layer-normalization]], [[feed-forward-network]], [[gelu]], [[shortcut-connections]], [[token-embedding]], [[positional-embedding]], [[matrix]], [[linear-mapping]], [[softmax]]
- Leads to: [[pretraining]], [[llm-training-loop]], [[decoding-strategies]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]]
- Components: [[transformer-block]], [[gpt-model-configuration]], [[logits]], [[weight-tying]], [[parameter-count-and-memory]], [[greedy-decoding]], [[dropout]]
- Instances: [[gpt-2]], [[gpt-3]]; contrast [[bert]]
- Math: [[next-token-prediction]], [[maximum-likelihood-estimation]], [[cross-entropy-loss]], [[backpropagation]], [[gradient-descent]], [[probability-distribution]]
- People: [[alec-radford]] (GPT, GPT-2), [[tom-brown]] (GPT-3), [[ashish-vaswani]] (transformer), [[andrej-karpathy]] (nanoGPT, which inspired the book's module split)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 8–10 (GPT uses the decoder; text completion; zero-/few-shot; transformers vs. LLMs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.6, book pp. 12–14 (decoder-only, autoregressive, left-to-right, Figure 1.8; GPT-3 96 layers / 175 B; Llama).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.7, book pp. 14–15 (GPT as the blueprint; stage 1 step 3 "LLM architecture").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.3, book p. 55 (Figure 3.6: GPT-like decoder-only transformer pipeline).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book pp. 93–99 (GPT-2 124M, `GPT_CONFIG_124M`, `DummyGPTModel`, Figures 4.2–4.4).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4, book pp. 109–113 (shortcut connections, gradient demo).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.5, book pp. 113–117 (Listing 4.6 `TransformerBlock`, Pre-LN, shape preservation).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6, book pp. 117–122 (Listing 4.7 `GPTModel`, Figure 4.15, 163,009,536 / 124,412,160 parameters, weight tying, 621.83 MB, Exercises 4.1–4.2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 122–127 (Listing 4.8 `generate_text_simple`, greedy decoding, chapter 4 summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.1, book pp. 130–131 (context length 256 for training).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.4, book pp. 159–160 (saving/loading `state_dict` and optimizer).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.5, book pp. 183–190 (`print(model)` architecture dump, classification head, last token).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 328–336 (LoRA wrapping of all Linear layers; 2,666,528 trainable parameters).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book pp. 290–293 (GPT-2/GPT-3 papers, nanoGPT, RMSNorm, Pre-LN papers).
