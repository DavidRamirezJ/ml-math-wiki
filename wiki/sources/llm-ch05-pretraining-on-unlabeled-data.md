---
title: "LLM Ch.5 — Pretraining on Unlabeled Data"
type: source
tags: [llm, pretraining, training-loop, decoding, cross-entropy, source]
sources: []
created: 2026-10-13
updated: 2026-10-13
---

# LLM Ch.5 — Pretraining on Unlabeled Data

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 5, **book pp. 128–168** (PDF pp. 150–190). Related back matter: Appendix B references for Ch. 5, **book pp. 293–295**. Appendix C solutions for Exercises 5.1–5.6, **book pp. 304–306**.
- **Type:** book chapter. *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025).
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13. Every section was checked line by line against the extracted text. Figures 5.12 (loss curves) and 5.17 (GPT-2 sizes) were checked against rendered pages. All printed numbers were recomputed where possible (log-probability average, perplexity, Llama 2 cost, temperature-scaled probabilities, top-k probabilities).
- *Page rule:* PDF page = book page + 22.

This is the chapter where the GPT model from Ch. 4 finally **learns**. The book first gives the model a number that says how wrong it is (the [[cross-entropy-loss]]). Then it writes a short training loop, trains on one tiny short story for 10 epochs (about 5 minutes on a laptop), and watches the model go from random gibberish to perfect, memorized sentences. Next it adds two "creativity dials" for text generation ([[temperature-scaling]] and [[top-k-sampling]]), shows how to save and resume training, and finishes by loading OpenAI's real GPT-2 weights into the home-made model.

## Section coverage (all of Ch. 5 accounted for)

Only §5.1 and §5.3 have numbered subsections. §5.2, §5.4 and §5.5 are single sections with no 5.x.y headings.

| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Chapter intro ("This chapter covers", Figs. 5.1–5.2, sidebar "Weight parameters") | 128–130 | [[pretraining]], [[llm-building-roadmap]], [[model-parameters]], [[pretrained-weights]], [[linear-layer]] |
| 5.1 | Evaluating generative text models | 129–146 | [[llm-evaluation]], [[cross-entropy-loss]], [[training-and-validation-loss]] |
| 5.1.1 | Using GPT to generate text | 130–132 | [[gpt-model-configuration]], [[context-length]], [[tokenizer-encode-decode]], [[logits]], [[greedy-decoding]], [[train-vs-eval-mode]], [[random-seed-reproducibility]] |
| 5.1.2 | Calculating the text generation loss (sidebars "Backpropagation", "Cross entropy loss", "Perplexity") | 132–140 | [[next-token-prediction]], [[softmax]], [[argmax]], [[log-probability]], [[maximum-likelihood-estimation]], [[cross-entropy-loss]], [[perplexity]], [[backpropagation]], [[loss-function]], [[tensor-reshaping-and-transposing]] |
| 5.1.3 | Calculating the training and validation set losses (sidebar "The cost of pretraining LLMs") | 140–146 | [[train-validation-test-split]], [[training-and-validation-loss]], [[cross-validation]], [[pytorch-dataset-and-dataloader]], [[sliding-window-sampling]], [[mini-batch]], [[empirical-risk-minimization]], [[expected-value-and-mean]], [[gpu-computing]] |
| 5.2 | Training an LLM (Listing 5.3, sidebar "AdamW") | 146–151 | [[llm-training-loop]], [[epoch]], [[gradient-descent]], [[stochastic-gradient-descent]], [[adamw-optimizer]], [[weight-decay]], [[regularization]], [[overfitting]], [[overfitting-and-underfitting]], [[train-vs-eval-mode]] |
| 5.3 | Decoding strategies to control randomness | 151–152 | [[decoding-strategies]], [[greedy-decoding]], [[autoregressive-text-generation]] |
| 5.3.1 | Temperature scaling | 152–155 | [[temperature-scaling]], [[multinomial-sampling]], [[random-variable]], [[probability-distribution]], [[monte-carlo-estimation]], [[histogram]] |
| 5.3.2 | Top-k sampling | 155–157 | [[top-k-sampling]], [[causal-attention-mask]] (same $-\infty$ masking trick) |
| 5.3.3 | Modifying the text generation function (Listing 5.4) | 157–159 | [[decoding-strategies]], [[end-of-text-token]], [[temperature-scaling]], [[top-k-sampling]] |
| 5.4 | Loading and saving model weights in PyTorch | 159–160 | [[model-checkpointing]], [[train-vs-eval-mode]], [[dropout]], [[adamw-optimizer]] |
| 5.5 | Loading pretrained weights from OpenAI (Fig. 5.17, Listing 5.5) | 160–167 | [[pretrained-weights]], [[gpt-2]], [[gpt-model-configuration]], [[bias-vector]], [[affine-space]], [[matrix-transpose]], [[weight-tying]], [[layer-normalization]], [[transfer-learning]] |
| — | Summary | 167–168 | [[greedy-decoding]], [[temperature-scaling]], [[training-and-validation-loss]], [[pretraining]], [[cross-entropy-loss]], [[adamw-optimizer]], [[pretrained-weights]] |
| — | Exercises 5.1–5.6 (in §5.3, §5.4, §5.5) | 155, 158, 159, 160, 167 | see *Exercises* below |
| App. B | References, Ch. 5 | 293–295 | see *References cited* below; [[galore]], [[top-p-sampling]], [[beam-search]] |
| App. C | Solutions 5.1–5.6 | 304–306 | see *Exercises* below |

Figures: 5.1–5.17 are all covered (5.1/5.2/5.5/5.8/5.10/5.13/5.16 are the roadmap; 5.3/5.4/5.6/5.7 the generation and loss steps; 5.9 the data split; 5.11 the training loop; 5.12 the loss curves; 5.14 temperature; 5.15 top-k; 5.17 the GPT-2 sizes). Listings 5.1–5.5 are all covered. NOTE boxes on p. 141 (Project Gutenberg, 60,000+ books), p. 142 (variable-length inputs), p. 146 (appendix D: warmup, cosine annealing, gradient clipping; see [[llm-appD-training-loop-bells-and-whistles]]) and p. 162 (download problems) are covered by the pages above.

## TL;DR (5 bullets)
- **A number for "how bad".** For each position the model gives a probability to the correct next token. The loss is the **negative average log-probability** of those tokens, which is exactly [[cross-entropy-loss]]. Its exponential is [[perplexity]], the "effective number of tokens the model is unsure about". An untrained GPT-2 scores loss **10.79** and perplexity **≈48,726** on a 50,257-token vocabulary, so it is close to a blind guess (§5.1).
- **A tiny training loop.** Eight steps per [[epoch]] and batch: reset gradients, compute the loss, `loss.backward()`, `optimizer.step()`, then optionally print losses and a sample text. The optimizer is [[adamw-optimizer]] (`lr=0.0004`, `weight_decay=0.1`) (§5.2).
- **Classic overfitting on purpose.** Training on one short story (*The Verdict*, 5,145 tokens) for 10 epochs drops training loss **9.78 → 0.39** while validation loss stalls at **≈6.45**. The model memorizes whole passages word for word ([[overfitting]], [[training-and-validation-loss]]).
- **Two dials for creativity.** [[temperature-scaling]] divides the logits by $T$ before softmax ($T<1$ sharper, $T>1$ flatter), and [[multinomial-sampling]] draws the token at random instead of taking the [[argmax]]. [[top-k-sampling]] keeps only the $k$ best logits and sets the rest to $-\infty$. Both are combined in one `generate()` function ([[decoding-strategies]], §5.3).
- **Save, resume, reuse.** Save `state_dict()` of the model **and** of the optimizer (AdamW keeps per-weight moment estimates) to resume training ([[model-checkpointing]], §5.4). Then load OpenAI's GPT-2 weights: set context length 1,024 and `qkv_bias=True`, split the fused QKV matrix into three, transpose the weights, and reuse the token embedding as the output layer ([[weight-tying]]). The model then writes fluent text ([[pretrained-weights]], §5.5).

## Key concepts introduced
**Owned by this chapter:**
- [[pretrained-weights]]: parameters learned once on a large corpus and loaded as the starting point; here GPT-2 124M from OpenAI.
- [[llm-training-loop]]: the eight-step loop of Fig. 5.11 and `train_model_simple` (Listing 5.3).
- [[decoding-strategies]]: rules that turn next-token probabilities into an actual token (greedy, temperature, top-k).
- [[cross-entropy-loss]]: $\mathcal L=-\frac1N\sum_{n=1}^N\log p_\theta(y_n\mid\text{context}_n)$.
- [[perplexity]]: $\text{PPL}=e^{\mathcal L}$.
- [[log-probability]]: work with $\log p$ because it turns products into sums and keeps tiny numbers like $10^{-5}$ manageable.
- [[train-validation-test-split]]: 90% / 10% character-level split of the raw text.
- [[training-and-validation-loss]]: the two curves that show learning (both fall) and overfitting (they diverge).
- [[cross-validation]]: held-out evaluation idea behind the validation split.
- [[epoch]]: one full pass over the training set.
- [[adamw-optimizer]]: Adam with *decoupled* weight decay; the book's default optimizer.
- [[weight-decay]]: a penalty on large weights that reduces overfitting.
- [[gradient-descent]]: the update $\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L$ behind `optimizer.step()`.
- [[temperature-scaling]]: $p_i(T)=\dfrac{e^{z_i/T}}{\sum_j e^{z_j/T}}$, $T>0$.
- [[multinomial-sampling]]: draw $x_{t+1}\sim\text{Categorical}(p_1,\dots,p_V)$ with `torch.multinomial`.
- [[top-k-sampling]]: mask all logits below the $k$-th largest to $-\infty$, renormalize, sample.
- [[monte-carlo-estimation]]: 1,000 samples → frequencies ≈ probabilities (§5.3.1, Exercise 5.1).
- [[random-variable]]: the next token as a discrete random variable over the vocabulary.
- [[model-checkpointing]]: `torch.save({"model_state_dict":…, "optimizer_state_dict":…})`.
- [[train-vs-eval-mode]]: `model.train()` turns dropout on; `model.eval()` turns it off.
- [[affine-space]]: `qkv_bias=True` turns the Q/K/V projections into affine maps $xW+b$.
- [[top-p-sampling]] and [[beam-search]]: decoding methods only named in Appendix B (not taught).
- [[galore]]: memory-efficient pretraining by low-rank gradient projection (Appendix B only).
- People: [[sebastian-raschka]] (author), [[angela-fan]] (top-k sampling), [[jiawei-zhao]] (GaLore).

**Reused / extended from earlier chapters:**
[[pretraining]], [[next-token-prediction]], [[autoregressive-text-generation]], [[gpt-2]], [[gpt-model-configuration]], [[context-length]], [[tokenizer-encode-decode]], [[token-id]], [[logits]], [[softmax]], [[argmax]], [[greedy-decoding]], [[pytorch-dataset-and-dataloader]], [[sliding-window-sampling]], [[mini-batch]], [[dropout]], [[causal-attention-mask]], [[end-of-text-token]], [[linear-layer]], [[bias-vector]], [[matrix-transpose]], [[layer-normalization]], [[weight-tying]], [[transfer-learning]], [[overfitting]], [[overfitting-and-underfitting]], [[learning-rate-warmup]], [[cosine-decay]], [[gradient-clipping]].

**Math links (MML pages):** [[maximum-likelihood-estimation]], [[loss-function]], [[empirical-risk-minimization]], [[probability-distribution]], [[expected-value-and-mean]], [[histogram]], [[gradient]], [[backpropagation]], [[chain-rule]], [[automatic-differentiation]], [[stochastic-gradient-descent]], [[regularization]], [[matrix]], [[linear-mapping]], [[low-rank-approximation]].

## Notable results / numbers
**Setup (§5.1.1)**
- `GPT_CONFIG_124M`: vocab 50,257, emb 768, 12 heads, 12 layers, `drop_rate` 0.1, `qkv_bias` False. The only change from Ch. 4: `context_length` **1,024 → 256**, so it trains on a laptop (p. 130).
- Untrained output: "Every effort moves you rentingetic wasn? refres RexMeCHicular stren" (p. 132).

**The loss, step by step (§5.1.2, Fig. 5.7)**
- Two inputs of 3 tokens → logits `[2, 3, 50257]`. Target probabilities about $10^{-5}$ (e.g. 7.4541e-05) (p. 136).
- Six log-probabilities sum to −64.7638; mean **−10.7940**; negate → loss **10.7940**. `torch.nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten())` gives the same 10.7940 (pp. 137–139).
- Perplexity $e^{10.7940}\approx$ **48,725.8** (p. 140). For comparison (derived): a uniform guess gives loss $\ln 50257\approx10.825$ and perplexity 50,257.
- Random start: probabilities ≈ $1/V$: $1/7\approx0.14$ (toy vocab), $1/50257\approx0.00002$ (GPT-2) (p. 135).

**Data and losses before training (§5.1.3)**
- *The Verdict* (Edith Wharton): **20,479 characters, 5,145 tokens** (p. 141).
- 90/10 split; `batch_size=2`, `max_length=stride=256` → **9 training batches, 1 validation batch** of shape [2, 256] (pp. 143–144). Real LLMs use batch sizes of 1,024 or more (p. 143).
- Untrained losses: **train 10.9876, val 10.9811** (p. 145). These are slightly *above* the uniform value 10.825 (derived).
- Cost sidebar: Llama 2 7B needed **184,320 A100 GPU-hours** on **2 trillion tokens**; at ≈$30/hour for an 8×A100 server that is $184{,}320/8\times30=\$691{,}200\approx\$690{,}000$ (p. 141).

**Training (§5.2)**
- AdamW `lr=0.0004`, `weight_decay=0.1`, 10 epochs, `eval_freq=5`, `eval_iter=5`, seed 123; about **5 minutes on a MacBook Air** (p. 149).
- Losses (train / val): step 0 **9.781 / 9.933**; step 5 8.111 / 8.339; step 10 6.661 / 7.048; step 15 5.961 / 6.616; step 80 0.541 / 6.393; step 85 **0.391 / 6.452** (p. 149). Derived perplexities at the end: train $e^{0.391}\approx1.48$, val $e^{6.452}\approx634$.
- Samples: epoch 1 only commas; epoch 2 "and, and, and"; epoch 10 a verbatim *Verdict* sentence (p. 149). Fig. 5.12: the curves split after epoch 2 → overfitting (p. 150). Normal practice: much more data, about **one epoch** (p. 151).

**Decoding (§5.3)**
- Toy 9-token vocab, logits `[4.51, 0.89, −1.90, 6.75, 1.63, −1.62, −1.89, 6.28, 1.79]`; greedy picks "forward" (p. 153).
- 1,000 multinomial samples at $T=1$: **582 forward, 343 toward, 73 closer, 2 inches**, 0 for the rest (pp. 153–154).
- $T=0.1$ → "forward" almost 100%; $T=5$ → "pizza" about **4%** (p. 155). Recomputed exact values: $T=1$ forward 0.572; $T=0.1$ forward 0.991; $T=5$ pizza 0.043.
- Top-3: logits kept `[6.75, 6.28, 4.51]` at positions `[3, 7, 0]`; softmax → **0.0615 / 0.5775 / 0.3610**, zeros elsewhere (pp. 156–157).
- `generate(..., top_k=25, temperature=1.4)` on the overfit model: "Every effort moves you stand to work on surprise, a one of us had gone with random-" (p. 158), no longer memorized.

**Checkpoints (§5.4)**
- `torch.save(model.state_dict(), "model.pth")`; load with `map_location=device`. Full checkpoint `model_and_optimizer.pth` with both state dicts; resume with AdamW `lr=5e-4`, `weight_decay=0.1` and `model.train()` (pp. 159–160).

**OpenAI GPT-2 (§5.5)**
- Download: 7 files; the weights file `model.ckpt.data-00000-of-00001` is **498 MB** (pp. 161–162). Settings `{n_vocab 50257, n_ctx 1024, n_embd 768, n_head 12, n_layer 12}`; keys `blocks, b, g, wpe, wte`; `wte` shape **(50257, 768)** (pp. 162–163).
- Fig. 5.17 sizes: small 124M (12 blocks, 12 heads, 768), medium 355M (24, 16, 1,024), large 774M (36, 20, 1,280), XL 1,558M (48, 25, 1,600) (p. 163). Head size is 64 in all four (derived).
- Two config changes to load the weights: `context_length=1024`, `qkv_bias=True` (p. 164). The fused `c_attn` matrix (768×2304, derived) is split in three with `np.split(..., 3, axis=-1)`; weights are transposed with `.T` (p. 165); `out_head.weight ← wte` (weight tying, p. 166).
- With `top_k=50`, `temperature=1.5`: "Every effort moves you toward finding an ideal new way to practice something! What makes us want to be on top of that?" (p. 167).

## Exercises
| # | Task (book page) | Solution idea and result (App. C) |
|---|---|---|
| 5.1 | With `print_sampled_tokens`, how often is "pizza" sampled at $T = 1, 0.1, 5$? Find a faster, more accurate way (p. 155) | "pizza" appears **0 times** at the low temperatures and **32/1,000 = 3.2%** at $T=5$. The exact way is to read the probability straight from the scaled softmax, `scaled_probas[2][6]` = **4.3%**. The sample count is only a [[monte-carlo-estimation]] of it (p. 304). |
| 5.2 | Try different temperature and top-k values. Which applications want low values, which high (p. 158) | Low top-k (< 10) and $T<1$: predictable, coherent text for reports, code, Q&A, teaching material. Top-k ≈ 20–40 and $T>1$: diverse text for brainstorming and fiction (pp. 304–305). |
| 5.3 | Which `generate` settings make it deterministic, like `generate_text_simple` (p. 159) | (1) `top_k=None` with no temperature scaling → greedy argmax; (2) `top_k=1` → only one token survives (p. 305). From the code: `temperature=0.0` alone already takes the argmax branch, whatever `top_k` is. |
| 5.4 | Load the saved model and optimizer in a new session and train one more epoch (p. 160) | Load the checkpoint, rebuild `GPTModel` and AdamW, call `load_state_dict` on both, then `train_model_simple(..., num_epochs=1)`. The optimizer state must be restored too, because AdamW keeps per-weight moments (p. 305). |
| 5.5 | Compute train and validation loss of the pretrained OpenAI GPT-2 on *The Verdict* (p. 167) | 124M: **train 3.7547, val 3.5596** (derived perplexities ≈ 42.7 and 35.2). The two are close. Either *The Verdict* was not in GPT-2's training data (so no overfitting to it), or it was, and then the validation split cannot measure overfitting at all (data contamination) (pp. 305–306). |
| 5.6 | Load a larger GPT-2 (e.g. 1,558M) and compare the text with 124M (p. 167) | Change only two lines: `model_size="1558M"` and `model_name="gpt2-xl (1558M)"` (p. 306). |

## References cited (Appendix B, pp. 293–295)
1. Raschka, "L8.2 Logistic Regression Loss Function" (video): why the log makes the loss easier to optimize ([[log-probability]]).
2. Raschka, "L8.7.1 OneHot Encoding and Multi-category Cross Entropy" (video): how PyTorch's cross-entropy works ([[cross-entropy-loss]], [[one-hot-encoding]]).
3. Raschka, "Understanding Onehot Encoding and Cross Entropy in PyTorch" (code, mng.bz/o05v): code for item 2.
4. Biderman et al. (2023), "Pythia", arXiv 2304.01373: fully documented pretraining data, hyperparameters and architecture.
5. Groeneveld et al. (2024), "OLMo", arXiv 2402.00838: a second fully documented pretraining recipe.
6. Book code, "Pretraining GPT on the Project Gutenberg Dataset" (mng.bz/Bdw2): prepare 60,000 public-domain books ([[pretraining-datasets]]).
7. Ibrahim et al. (2024), "Simple and Scalable Strategies to Continually Pre-train LLMs", arXiv 2403.08763: [[learning-rate-warmup]] and [[cosine-decay]] also work for continued pretraining.
8. Wu et al. (2023), "BloombergGPT", arXiv 2303.17564: a domain LLM trained on general + finance text.
9. [[jiawei-zhao]] et al. (2024), "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection", arXiv 2403.03507: swap AdamW for `GaLoreAdamW` ([[galore]]). Editorial link (not in the book): [[low-rank-approximation]], [[singular-value-decomposition]].
10. Soldaini et al. (2024), "Dolma", arXiv 2402.00159: open corpus of three trillion tokens.
11. Gao et al. (2020), "The Pile", arXiv 2101.00027: open pretraining dataset.
12. Penedo et al. (2023), "The RefinedWeb Dataset for Falcon LLM", arXiv 2306.01116: web-only pretraining data.
13. Together AI, "RedPajama" (mng.bz/d6nw): open pretraining dataset.
14. The FineWeb Dataset (mng.bz/rVzy): more than 15 trillion tokens of cleaned, deduplicated English CommonCrawl text.
15. [[angela-fan]] et al. (2018), "Hierarchical Neural Story Generation", arXiv 1805.04833: the paper that introduced [[top-k-sampling]].
16. Wikipedia, "Top-p sampling": [[top-p-sampling]] (nucleus sampling), choose from the smallest set of tokens whose cumulative probability exceeds $p$. Not covered in the chapter.
17. Vijayakumar et al. (2016), "Diverse Beam Search", arXiv 1610.02424: [[beam-search]], keep only the best partial sequences at each step. Not covered in the chapter.

Items 10–14 are grouped by the book as open datasets of "hundreds of gigabytes to terabytes of text data". BloombergGPT and The Pile are also cited in Ch. 1.

## Errata / quirks found during validation
- **Learning rate mismatch.** Training in §5.2 uses `lr=0.0004` (p. 149), but the resume code in §5.4 and Exercise 5.4 uses `lr=5e-4` (p. 160, p. 305). The book does not comment. In practice `optimizer.load_state_dict` restores the saved learning rate, so the constructor value is overwritten (PyTorch behaviour).
- **Exercise 5.1 temperatures.** The exercise asks for $T = 1, 0.1, 5$ (the Fig. 5.14 values), but the solution says "temperature 0 or 0.1" for the 0-count case (p. 304). The main text says "pizza" is about 4% at $T=5$; the exact value is 4.3%, and the 1,000-sample estimate is 3.2%.
- **Naming slips.** §5.3.3 and Exercise 5.3 say `generate_simple` (pp. 158–159), meaning `generate_text_simple`. App. C says "train_simple_function" (p. 305), meaning `train_model_simple`. App. C p. 304 has the typo `return logitss`. The download list prints `hprams.json` (p. 162), meaning `hparams.json`.
- **Loose wording.** The book calls logits "vectors representing the probability distribution" (p. 131); they are unnormalized scores until softmax (the book itself says so on p. 139). It also calls the 50,257 output dimension "embedding dimensionality" (p. 134); it is the vocabulary dimension.
- **Fig. 5.1** labels step 7 "Load pretrained weights", while the caption calls it *saving and loading model weights* (p. 129).
- **`eos_id` check** in Listing 5.4 (`if idx_next == eos_id`) only works for batch size 1; the book does not say this.

## What this changes in the wiki
- **Adds the full pretraining pipeline.** The owned pages above ([[llm-training-loop]], [[cross-entropy-loss]], [[perplexity]], [[log-probability]], [[decoding-strategies]], [[temperature-scaling]], [[multinomial-sampling]], [[top-k-sampling]], [[model-checkpointing]], [[train-vs-eval-mode]], [[adamw-optimizer]], [[weight-decay]], [[pretrained-weights]], [[galore]], [[top-p-sampling]], [[beam-search]], plus [[sebastian-raschka]], [[angela-fan]], [[jiawei-zhao]]) give the wiki its first complete "random weights → trained language model → sampling" story.
- **Connects LLM practice to MML theory.** Cross-entropy on next tokens is [[maximum-likelihood-estimation]] of a categorical [[probability-distribution]]; averaging per-token and per-batch losses is [[empirical-risk-minimization]] with [[expected-value-and-mean]]; the training loop is mini-batch [[stochastic-gradient-descent]] with [[backpropagation]]; sampling 1,000 tokens is [[monte-carlo-estimation]] summarized in a [[histogram]]; the next token is a discrete [[random-variable]]; QKV biases make the projections affine maps ([[affine-space]]); the validation split is the simplest form of [[cross-validation]]. These MML pages ([[gradient-descent]], [[cross-validation]], [[random-variable]], [[affine-space]]) gain an LLM example section; nothing in them is replaced.
- **Overfitting.** The 0.39 vs 6.45 gap (Fig. 5.12) is the wiki's clearest overfitting example. It agrees with [[overfitting]] (MML Ch. 8) and [[overfitting-and-underfitting]] (LLM Ch. 6).
- **Sets up later chapters.** The pretrained GPT-2 from §5.5 is the start point for [[classification-fine-tuning]] (Ch. 6, see [[llm-ch06-fine-tuning-for-classification]]) and [[instruction-fine-tuning]] (Ch. 7). The training-loop extras named on p. 146 are covered in [[llm-appD-training-loop-bells-and-whistles]].
- **Cross-unit notes (coverage audit 2026-10-13).** Pages owned by other units that the coverage table relies on now carry a "From llm-ch05" note and a Ch. 5 citation: [[argmax]], [[gpu-computing]], [[linear-layer]], [[llm-building-roadmap]], [[llm-evaluation]], [[mini-batch]], [[pytorch-dataset-and-dataloader]], [[random-seed-reproducibility]], [[expected-value-and-mean]], [[overfitting-and-underfitting]], [[one-hot-encoding]], [[pretraining-datasets]], [[automatic-differentiation]].
- **No contradictions** with existing pages. The quirks above (learning rate 0.0004 vs 5e-4, Exercise 5.1 temperatures, logits wording) are recorded here so concept pages can cite them instead of silently correcting the book.
