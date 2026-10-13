---
title: Loading Pretrained Weights
type: concept
tags: [llm, transfer-learning, gpt-2, pytorch, model-parameters]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Loading Pretrained Weights

> **One-liner:** Instead of spending months and millions of dollars teaching a model English yourself, you download the numbers someone else's training already produced and pour them into your own copy of the same architecture.

## 🟢 Intuition (Dummy)
A model is two things: the **architecture** (the shape of the machine: layers,
sizes, wiring) and the **weights** (millions of numbers inside it that hold
everything it learned). An untrained model is a perfect machine with random
numbers inside, so it talks gibberish.

Training those numbers is the expensive part. OpenAI already did it for GPT-2
and published the result. So the book builds GPT-2's architecture from scratch,
downloads OpenAI's numbers, and copies each one into the matching slot. It is
like getting a piano built to the exact spec and then loading a professional
pianist's "muscle memory" into it.

Two things must match exactly: every slot's **shape**, and the **meaning** of each
slot (OpenAI uses different names, so you need a translation table). If you get
it right, the model writes sensible English. If one slot is wrong, it writes
nonsense. That is also the book's test: "does it make sense now?"

## 🟡 Mechanics (Practitioner)

### Why bother: the cost of pretraining
- GPT-3 pretraining: about **\$4.6 million** in cloud credits (book p. 11).
- Llama 2 7B: **184,320 A100 GPU-hours** on 2 trillion tokens. At about \$30/hour for an 8×A100 server:
  $184{,}320/8\times\$30=\$691{,}200\approx\$690{,}000$ (book p. 141).
- In general, pretraining from scratch costs "thousands to millions of dollars" (book p. 15).

Loading open weights skips this stage and gives you a [[foundation-model]] to
[[fine-tuning|fine-tune]] (book pp. 12, 15). This is
[[transfer-learning]]: fine-tuning starts gradient descent from
$\theta_0=\theta_{\text{pretrained}}$ instead of random numbers.

### Step 1: download (book pp. 161–162)
```python
from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
```
Needs `tensorflow>=2.15.0` and `tqdm>=4.66`, because OpenAI saved the weights as
a TensorFlow checkpoint. Seven files arrive; the big one is
`model.ckpt.data-00000-of-00001` (**498 MB** for 124M; **1.42 GB** for 355M, book p. 227).
`encoder.json` and `vocab.bpe` are the [[byte-pair-encoding]] tokenizer files.

```text
settings = {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
params.keys() = ['blocks', 'b', 'g', 'wpe', 'wte']
params["wte"].shape = (50257, 768)
```
- `wte` = **w**eights of **t**oken **e**mbedding, `wpe` = **w**eights of **p**ositional **e**mbedding,
  `g`/`b` = final LayerNorm gain/bias, `blocks` = the 12 transformer blocks.

In math: $W_{\text{te}}\in\mathbb R^{50257\times768}$.
- $\mathbb R^{50257\times768}$ = "a [[matrix]] of real numbers with 50,257 rows and 768 columns".
- Row $i$ is the 768-number [[token-embedding]] of token ID $i$.

### Step 2: build a matching model (book p. 164)
The four GPT-2 sizes share one architecture and differ only in width, depth and heads (Fig. 5.17, book p. 163):

| Model | Params | `emb_dim` | `n_layers` | `n_heads` |
|---|---|---|---|---|
| gpt2-small | 124M | 768 | 12 | 12 |
| gpt2-medium | 355M | 1,024 | 24 | 16 |
| gpt2-large | 774M | 1,280 | 36 | 20 |
| gpt2-xl | 1,558M | 1,600 | 48 | 25 |

```python
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs["gpt2-small (124M)"])
NEW_CONFIG.update({"context_length": 1024})   # we trained with 256; GPT-2 used 1,024
NEW_CONFIG.update({"qkv_bias": True})         # OpenAI's Q/K/V layers have biases
gpt = GPTModel(NEW_CONFIG); gpt.eval()
```
- `context_length` must be 1,024 so the positional embedding is $1024\times768$, the shape of `wpe`.
- `qkv_bias=True` turns each projection into an affine map, e.g. $\mathbf q=\mathbf x W_q+\mathbf b_q$
  ($\mathbf x$ = input vector, $W_q$ = query weight matrix, $\mathbf b_q$ = query [[bias-vector]]).
  Modern LLMs usually drop these biases, but we need them to match the checkpoint (book p. 164).

### Step 3: copy, with a shape check (book pp. 165–166)
```python
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
```
`assign` is the safety net: wrong shape → error. `torch.nn.Parameter` makes the
copied array a trainable PyTorch weight.

`load_weights_into_gpt` (Listing 5.5) is the translation table:

| OpenAI name | Our `GPTModel` slot | Note |
|---|---|---|
| `wpe`, `wte` | `pos_emb.weight`, `tok_emb.weight` | |
| `attn.c_attn.w` | `W_query`, `W_key`, `W_value` weights | one fused $768\times2304$ array, split into 3 with `np.split(..., 3, axis=-1)`, each **transposed** `.T` |
| `attn.c_attn.b` | the three Q/K/V biases | split into 3, no transpose |
| `attn.c_proj.w/b` | `att.out_proj` | weight `.T` |
| `mlp.c_fc.w/b` | `ff.layers[0]` (768 → 3,072) | weight `.T` |
| `mlp.c_proj.w/b` | `ff.layers[2]` (3,072 → 768) | weight `.T`; `layers[1]` is GELU, no weights |
| `ln_1.g/b`, `ln_2.g/b` | `norm1/norm2.scale/shift` | gain = scale $\gamma$, bias = shift $\beta$ |
| `g`, `b` | `final_norm.scale/shift` | |
| `wte` (again) | `out_head.weight` | **[[weight-tying]]**: GPT-2 reuses the token-embedding matrix as the output layer |

The book admits this mapping "took a lot of guesswork" (book p. 166).

### Step 4: check by generating (book p. 167)
```python
torch.manual_seed(123)
token_ids = generate(model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25, context_size=NEW_CONFIG["context_length"],
    top_k=50, temperature=1.5)
```
Output: *"Every effort moves you toward finding an ideal new way to practice
something! What makes us want to be on top of that?"* Coherent text means the
weights loaded correctly; "a tiny mistake in this process would cause the model
to fail" (book p. 167). See [[decoding-strategies]].

A number check (Exercise 5.5, appendix C p. 305): pretrained GPT-2 124M on
"The Verdict" gets train/val loss **3.755 / 3.560**, versus about **10.99** for
the random model and 0.391 / 6.452 for the model trained only on that story.

### The same recipe, reused later
| Where | Model | What happens next | Book |
|---|---|---|---|
| Ch. 6 (spam) | gpt2-small, `drop_rate=0.0` | sanity check prints *"Every effort moves you forward. The first step is to understand the importance of your work"*; asked whether a message is spam, it just repeats the prompt (no instruction tuning yet); then the head is swapped for 2 classes | §6.4, pp. 181–183 |
| Ch. 7 (instructions) | **gpt2-medium (355M)** | 124M has too little [[model-capacity]] for good instruction following; download ≈ 1.42 GB (≈ 3× small) | §7.5, pp. 226–229 |
| App. E (LoRA) | gpt2-small | same Listing E.4, head → `Linear(768, 2)`, baseline accuracy 46.25% / 45.00% / 48.75% (train/val/test, near 50% chance; printed at the start of §E.4, p. 328) | §E.3–E.4, pp. 326–328 |

Switching to the 1,558M model takes two string changes
(`model_size="1558M"`, `"gpt2-xl (1558M)"`) (Exercise 5.6, appendix C p. 306).

## 🔴 Theory (Expert)
- **Why the transposes.** PyTorch's `nn.Linear` stores weights as
  (out_features, in_features) and computes $\mathbf y=\mathbf xW^\top+\mathbf b$.
  OpenAI's TensorFlow layout stores (in, out) and computes $\mathbf y=\mathbf xW+\mathbf b$.
  So the same map needs $W_{\text{torch}}=W_{\text{TF}}^\top$ (see
  [[matrix-transpose]]). Biases are 1-D vectors, so no transpose. The same issue
  appears in Exercise 3.1 (appendix C p. 301).
- **Fused QKV as a block matrix.** $W_{qkv}=[\,W_q\mid W_k\mid W_v\,]\in\mathbb R^{768\times2304}$,
  so $\mathbf xW_{qkv}=[\,\mathbf xW_q\mid\mathbf xW_k\mid\mathbf xW_v\,]$. One big
  [[matrix-multiplication]] computes all three projections; splitting the columns
  into three equal blocks recovers the separate matrices.
- **Weight tying as an inner product.** With tying, the logit for token $j$ is
  $z_j=\langle\mathbf h,\mathbf w_j\rangle$, the [[inner-product]] of the final
  hidden state $\mathbf h\in\mathbb R^{768}$ with token $j$'s own input embedding.
  Tying saves $50{,}257\times768=38{,}597{,}376$ parameters: 163,009,536 untied vs.
  124,412,160 tied for GPT-2 small (§4.6, book pp. 120–121). The book's own
  `GPTModel` keeps separate layers during its own training, because in the
  author's experience untied layers train better (book p. 121); loading GPT-2
  simply copies `wte` into both. See [[parameter-count-and-memory]].
- **Shape checks are necessary but not sufficient.** Q, K and V all have shape
  $768\times768$, so swapping two of them passes `assign` but breaks the model.
  The semantic check is behavioural: coherent generation and a sensible loss.
- **Pretrained weights as an initialization.** Fine-tuning is gradient descent
  started near a good region of a very non-convex loss surface. Freezing most
  layers ([[layer-freezing]]) or adding low-rank updates
  ([[low-rank-adaptation]]) constrains how far $\theta$ moves from
  $\theta_{\text{pretrained}}$, which acts as [[regularization]] and is
  cheaper.
- **Evaluation caveat: contamination.** Exercise 5.5's train ≈ val result has two
  readings: either "The Verdict" was not in GPT-2's training data (no
  overfitting to it), or it *was*, and then the validation split is not truly
  held out. Only text written after GPT-2's training would settle it
  (appendix C p. 306).
- **Precision and frameworks.** The arrays travel TensorFlow checkpoint → NumPy
  → `torch.tensor` (float32). Saving and reloading your own models in PyTorch
  uses `state_dict()` instead ([[model-checkpointing]]).

## Connections
- Builds on: [[gpt-architecture]], [[gpt-model-configuration]], [[model-parameters]], [[linear-layer]], [[token-embedding]], [[positional-embedding]], [[layer-normalization]], [[matrix-transpose]]
- Leads to: [[fine-tuning]], [[classification-fine-tuning]], [[instruction-fine-tuning]], [[low-rank-adaptation]], [[transfer-learning]]
- Used by: [[gpt-2]], [[foundation-model]], [[custom-llm]], [[decoding-strategies]]
- Related: [[pretraining]], [[weight-tying]], [[bias-vector]], [[model-capacity]], [[model-checkpointing]], [[parameter-count-and-memory]], [[llm-training-loop]], [[llm-building-roadmap]]
- People: [[alec-radford]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.5, book pp. 11–12 (GPT-3 \$4.6M cost; reusing open weights to skip pretraining)
- Same, §1.7, book pp. 14–15 (stage 2 step 7 "Load pretrained weights"; "thousands to millions of dollars")
- Same, §4.6, book pp. 120–121 (163,009,536 vs 124,412,160 parameters with weight tying)
- Same, §5.1.3, book p. 141 (Llama 2 pretraining-cost sidebar)
- Same, §5.5, book pp. 160–168 (download, settings/params, Fig. 5.17 sizes, `assign`, Listing 5.5 `load_weights_into_gpt`, weight tying, generation check, Exercises 5.5–5.6)
- Same, §6.4, book pp. 181–183 (Listing 6.6; spam prompt fails without instruction tuning)
- Same, §7.5, book pp. 226–229 (Listing 7.7; GPT-2 medium, capacity, 1.42 GB)
- Same, appendix E §E.3, book pp. 326–327 (Listing E.4, new 2-class head); §E.4, book p. 328 (baseline accuracies)
- Same, appendix C, book pp. 305–306 (solutions to Exercises 5.5 and 5.6)
