---
title: "LLM Ch.7 — Fine-tuning to Follow Instructions"
type: source
tags: [llm, fine-tuning, instruction-tuning, evaluation, source]
sources: []
created: 2026-10-13
updated: 2026-10-13
---

# LLM Ch.7 — Fine-tuning to Follow Instructions

- **Raw file:** `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 7, **book pp. 204–249** (PDF pp. 226–271). Related back matter: Appendix B references for Ch. 7, **book pp. 296–298**. Appendix C solutions for Exercises 7.1–7.4, **book pp. 307–310**.
- **Type:** book chapter. *Build a Large Language Model (From Scratch)*, [[sebastian-raschka]] (Manning, 2025).
- **Ingested:** 2026-10-13 · **Validated against source:** 2026-10-13. Every section was checked line by line against the extracted text. Figures 7.1, 7.17 and 7.21 and the Appendix C Phi-3 template were checked against rendered pages.
- *Page rule:* PDF page = book page + 22.

This is the chapter where the GPT model becomes a **personal assistant**. A pretrained model can only *continue* text. When you give it "Convert this sentence to passive voice", it just repeats your sentence. Here the book trains GPT-2 medium (355M) on 1,100 instruction–response pairs for 2 epochs (under a minute on an A100). After that it answers the instructions. To grade the answers, a bigger model (Llama 3 8B, running locally in Ollama) scores every test response from 0 to 100. The fine-tuned model averages **50.32**.

## Section coverage (all of Ch. 7 accounted for)

Sections 7.1–7.8 have **no numbered subsections**. Only §7.9 has subsections (7.9.1–7.9.3). Inside §7.3 the book uses its own sub-steps 2.1–2.5 (Fig. 7.6). They are listed as extra rows so that nothing is skipped.

| § | Section title | Book pp. | Wiki coverage |
|---|---|---|---|
| — | Chapter intro ("This chapter covers", Fig. 7.1) | 204–205 | [[instruction-fine-tuning]], [[llm-building-roadmap]], [[classification-fine-tuning]], [[foundation-model]] |
| 7.1 | Introduction to instruction fine-tuning | 205–206 | [[instruction-fine-tuning]], [[pretraining]], [[text-completion]], [[next-token-prediction]], [[instruction-dataset]], [[supervised-learning]] |
| 7.2 | Preparing a dataset for supervised instruction fine-tuning | 207–211 | [[instruction-dataset]], [[prompt-template]], [[special-tokens]], [[train-validation-test-split]] |
| 7.3 | Organizing data into training batches | 211–223 | [[collate-function]], [[dynamic-padding]], [[ignore-index-loss-masking]], [[instruction-masking]], [[pytorch-dataset-and-dataloader]], [[tokenization]], [[end-of-text-token]], [[padding-and-truncation]], [[cross-entropy-loss]] |
| 7.3 step 2.1 | Format data with the prompt template | 212–213 | [[prompt-template]] |
| 7.3 step 2.2 | Tokenize formatted data (`InstructionDataset`) | 212–213 | [[tokenization]], [[token-id]], [[pytorch-dataset-and-dataloader]] |
| 7.3 step 2.3 | Pad to the same length (per batch, ID 50256) | 214–215 | [[dynamic-padding]], [[end-of-text-token]], [[padding-and-truncation]], [[collate-function]] |
| 7.3 step 2.4 | Create target token IDs (shift by one) | 216–218 | [[next-token-prediction]], [[collate-function]] |
| 7.3 step 2.5 | Replace padding with −100 (+ optional instruction masking) | 218–223 | [[ignore-index-loss-masking]], [[instruction-masking]], [[cross-entropy-loss]], [[context-length]] |
| 7.4 | Creating data loaders for an instruction dataset | 223–226 | [[pytorch-dataset-and-dataloader]], [[collate-function]], [[mini-batch]], [[gpu-computing]], [[random-seed-reproducibility]], [[dynamic-padding]] |
| 7.5 | Loading a pretrained LLM | 226–229 | [[pretrained-weights]], [[gpt-2]], [[gpt-model-configuration]], [[model-capacity]], [[baseline-evaluation]], [[response-extraction]], [[autoregressive-text-generation]], [[train-vs-eval-mode]] |
| 7.6 | Fine-tuning the LLM on instruction data (+ sidebar "Dealing with hardware limitations") | 229–233 | [[instruction-fine-tuning]], [[llm-training-loop]], [[adamw-optimizer]], [[weight-decay]], [[training-and-validation-loss]], [[overfitting-and-underfitting]], [[epoch]], [[gpu-computing]] |
| 7.7 | Extracting and saving responses | 233–238 | [[response-extraction]], [[llm-evaluation]], [[classification-accuracy]], [[model-checkpointing]], [[greedy-decoding]] |
| 7.8 | Evaluating the fine-tuned LLM (+ sidebars "Using larger LLMs via web APIs", "Alternative Ollama models") | 238–247 | [[llm-as-a-judge]], [[local-llm-inference-ollama]], [[llm-evaluation]], [[temperature-scaling]], [[expected-value-and-mean]], [[random-variable]], [[hyperparameter]], [[model-capacity]] |
| 7.9 | Conclusions (Fig. 7.21) | 247–248 | [[llm-building-roadmap]], [[instruction-fine-tuning]] |
| 7.9.1 | What's next? | 247–248 | [[preference-fine-tuning]], [[direct-preference-optimization]] |
| 7.9.2 | Staying up to date in a fast-moving field | 248 | — (pointers only: arXiv cs.LG, r/LocalLLaMA, author's blog) |
| 7.9.3 | Final words | 248–249 | [[sebastian-raschka]] (pointers to Axolotl and LitGPT) |
| — | Summary | 249 | [[instruction-fine-tuning]], [[collate-function]], [[llm-as-a-judge]] |
| — | Exercises 7.1–7.4 (in §7.2, §7.3, §7.6, §7.8) | 209, 223, 233, 247 | see *Exercises* below; [[low-rank-adaptation]], [[parameter-efficient-fine-tuning]] |
| App. B | References, Ch. 7 | 296–298 | see *References cited* below |
| App. C | Solutions 7.1–7.4 | 307–310 | see *Exercises* below |

Figures: the same 3-stage, 9-step pipeline diagram comes back as Figs. 7.3, 7.5, 7.14, 7.15, 7.16, 7.18 and 7.19, each time with the current step highlighted. The pipeline is: **Stage 1** prepare the dataset (1 download & format, 2 batch, 3 data loaders). **Stage 2** fine-tune (4 load pretrained LLM, 5 instruction fine-tune, 6 inspect loss). **Stage 3** evaluate (7 extract responses, 8 qualitative evaluation, 9 score responses).

## TL;DR (5 bullets)
- **Same loss, new data.** [[instruction-fine-tuning]] (also called supervised fine-tuning, SFT) does not change the objective. It is still [[next-token-prediction]] with [[cross-entropy-loss]]. The only new thing is the data: instruction–response pairs written in a fixed [[prompt-template]] (Alpaca style, §7.1–7.2, §7.6).
- **The real work is batching.** A custom [[collate-function]] does four jobs. It pads each batch only to the length of its own longest example ([[dynamic-padding]]). It shifts the targets by one position. It replaces every pad token in the targets *except the first* with −100, so the loss ignores them ([[ignore-index-loss-masking]]). It can also truncate to 1,024 tokens (§7.3).
- **Bigger model needed.** The 124M model does not have enough [[model-capacity]] to follow instructions well, so the book switches to **GPT-2 medium (355M)**. Before fine-tuning, the model just echoes the prompt. This is the [[baseline-evaluation]] (§7.5).
- **Train quickly, then check quality.** With AdamW, lr 5e-5, weight decay 0.1 and 2 epochs, the loss falls from about 3.8 to 0.30 (train) and 0.66 (val). A low loss is not proof of good answers, so the book extracts every test answer ([[response-extraction]]) and reads three of them by hand (§7.6–7.7).
- **Let a bigger LLM grade.** In [[llm-as-a-judge]], Llama 3 8B, running locally in Ollama ([[local-llm-inference-ollama]]), scores each of the 110 test answers from 0 to 100 against the reference answer. The average is **50.32**. For comparison, Llama 3 8B base scores 58.51 and Llama 3 8B instruct scores 82.6. An optional next step is [[preference-fine-tuning]], for example [[direct-preference-optimization]] (§7.8–7.9).

## Key concepts introduced
**Owned by this chapter (new pages):**
- [[instruction-dataset]]: a list of `{instruction, input, output}` records. The book's set has 1,100 of them, and `input` can be empty.
- [[prompt-template]]: a fixed way to turn one record into one training string, for example the Alpaca style (`### Instruction:` / `### Input:` / `### Response:`) or the Phi-3 style (`<|user|>` / `<|assistant|>`).
- [[collate-function]]: the function a `DataLoader` uses to merge a list of samples into one batch tensor.
- [[dynamic-padding]]: pad each batch only to its own longest sequence, so different batches have different lengths (8×61, 8×76, …).
- [[ignore-index-loss-masking]]: target value −100 means "skip this position". The mean is taken only over the positions that are kept, $\mathcal L = -\frac{1}{|\mathcal I|}\sum_{i\in\mathcal I}\log p_\theta(y_i\mid\cdot)$.
- [[instruction-masking]]: also set the target positions of the *instruction* tokens to −100, so the model is graded only on its response. This is optional, and the research on whether it helps is mixed.
- [[model-capacity]]: how many patterns a model can store. Roughly, it grows with the parameter count (124M vs 355M).
- [[baseline-evaluation]]: measure the model *before* training so you can see what training changed.
- [[response-extraction]]: `generated_text[len(input_text):].replace("### Response:", "").strip()`.
- [[llm-evaluation]]: the three families of LLM evaluation: benchmarks (MMLU), human preference (Chatbot Arena), and automated LLM judges (AlpacaEval).
- [[llm-as-a-judge]]: a stronger LLM gives each answer a score against a reference answer. The scores are averaged.
- [[local-llm-inference-ollama]]: Ollama, an inference-only wrapper around llama.cpp, with a REST API on port 11434.
- [[preference-fine-tuning]]: an optional step after SFT that aligns the model with what humans prefer.
- [[direct-preference-optimization]]: DPO, the preference-tuning method in the book's bonus code (`04_preference-tuning-with-dpo`).
- [[zhengyan-shi]] (person): first author of "Instruction Tuning With Loss Over Instructions" (2024).

**Reused / extended from earlier chapters:**
[[instruction-fine-tuning]], [[fine-tuning]], [[classification-fine-tuning]], [[pretraining]], [[foundation-model]], [[text-completion]], [[next-token-prediction]], [[supervised-learning]], [[llm-building-roadmap]], [[tokenization]], [[token-id]], [[special-tokens]], [[end-of-text-token]], [[padding-and-truncation]], [[context-length]], [[pytorch-dataset-and-dataloader]], [[mini-batch]], [[train-validation-test-split]], [[random-seed-reproducibility]], [[gpu-computing]], [[pretrained-weights]], [[gpt-2]], [[gpt-model-configuration]], [[train-vs-eval-mode]], [[autoregressive-text-generation]], [[greedy-decoding]], [[temperature-scaling]], [[cross-entropy-loss]], [[perplexity]], [[llm-training-loop]], [[training-loop]], [[adamw-optimizer]], [[weight-decay]], [[epoch]], [[training-and-validation-loss]], [[overfitting-and-underfitting]], [[hyperparameter]], [[classification-accuracy]], [[model-checkpointing]], [[low-rank-adaptation]], [[parameter-efficient-fine-tuning]], [[lora-rank-and-alpha]].

**Math links (MML pages):** [[loss-function]], [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[probability-distribution]], [[softmax]], [[gradient-descent]], [[stochastic-gradient-descent]], [[backpropagation]], [[expected-value-and-mean]], [[covariance-and-variance]], [[random-variable]], [[cross-validation]], [[overfitting]], [[regularization]], [[low-rank-approximation]], [[basis-and-rank]], [[matrix]].

## Notable results / numbers
**Dataset (§7.2)**
- **1,100** instruction–response pairs in a 204 KB JSON file (p. 207). Example records: `data[50]` (spelling of "Ocassion") and `data[999]` (antonym of "complicated", with an empty input) (p. 208).
- The split is 85 / 10 / 5 % → **935 train / 110 test / 55 validation** (p. 210). Note the slice order: train, then *test*, then validation.

**Batching (§7.3–7.4)**
- Pad token = `<|endoftext|>` = **50256**. Ignore value = **−100** (the PyTorch `cross_entropy` default). GPT-2 context = **1,024** tokens (pp. 214, 218, 222).
- Toy collate: inputs `[[0,1,2,3,4],[5,6,P,P,P],[7,8,9,P,P]]` → targets `[[1,2,3,4,P],[6,P,−100,−100,−100],[8,9,P,−100,−100]]`, with P = 50256 (pp. 220–221). Exactly one end-of-text token is kept in each target, so the model learns when to stop.
- Cross-entropy demo: `loss_1 = 1.1269` (2 tokens). Adding a third token gives 0.7936. Giving that third token the target −100 brings the loss back to 1.1269, and `loss_1 == loss_3` (pp. 221–222). Check: $\tfrac12[\log(1+e^{2})+\log(1+e^{-2})]=\tfrac12(2.1269+0.1269)=1.1269$.
- Loaders use `batch_size=8` and seed 123. Train has `shuffle=True, drop_last=True`; val/test have `False, False`. Batch shapes vary: 8×61, 8×76, 8×73, …, 8×69 (pp. 225–226).

**Model (§7.5)**
- **GPT-2 medium (355M)**: `emb_dim` 1024, 24 layers, 16 heads (head dim 64), vocab 50,257, context 1,024, `drop_rate` 0.0, `qkv_bias` True. The download is **1.42 GB**, about 3× the small model (p. 227).
- Baseline on `val_data[0]` ("Convert the active sentence to passive: 'The chef cooks the meal every day.'"): the model writes `### Response:` and then **repeats the input sentence** (p. 229).

**Training (§7.6)**
- Initial loss (5 batches): **3.826 train / 3.762 val** (p. 230). That is a [[perplexity]] of about $e^{3.8}\approx 45$ (derived).
- AdamW, `lr = 5e-5`, `weight_decay = 0.1`, **2 epochs**, `eval_freq = 5`, `eval_iter = 5`, seed 123 (p. 231).
- Log: 2.637/2.626 at step 0 → 0.520/0.665 at step 115 → **0.300/0.657** at step 230. **0.87 min** on an A100 (pp. 231–232). The epoch-1 sample says "The meal is *prepared* every day by the chef." The epoch-2 sample says "The meal is *cooked* every day by the chef." (p. 232). The book warns that more epochs may increase overfitting (p. 232).
- Run times for 2 epochs (p. 231): 355M takes 15.78 min on CPU (M3 MacBook Air), 1.83 min on an L4 and 0.86 min on an A100. 124M takes 5.74 / 0.69 / 0.39 min.

**Responses (§7.7)**
- Generation uses `max_new_tokens=256` and `eos_id=50256`. Generating all **110** test responses takes about 1 min on an A100 or about 6 min on an M3 (`110/110 [01:05, 1.68 it/s]`) (pp. 234, 238).
- Hand check of 3 examples (pp. 235–236): simile "as fast as a bullet" (correct), "cumulus" instead of "cumulonimbus" (close but wrong), "Jane Austen" (correct). The model is saved as `gpt2-medium355M-sft.pth` (p. 238).

**Judge scoring (§7.8)**
- Judge = Llama 3 8B instruct, 4.7 GB download, about 16 GB RAM. Alternatives: phi3 (3.8B, about 8 GB RAM) or `llama3:70b` (pp. 240–241). Options: `seed 123`, `temperature 0`, `num_ctx 2048` (pp. 242–243).
- Free-text judge scores on the 3 examples: **85 / 40 / 95** (pp. 244–245).
- Integer-only prompt over all test entries: **110 of 110 parsed, average 50.32** (p. 246), where $\bar s=\frac1N\sum_i s_i$. Reference scores with the same method: Llama 3 8B base **58.51**, Llama 3 8B instruct **82.6** (p. 247).
- Ollama is not fully deterministic across operating systems, so the book suggests repeating the evaluation and averaging the scores (p. 247).
- Ways to improve the score: tune lr / batch size / epochs; use a larger or more varied dataset; try other prompt formats; use a larger pretrained model (p. 247).

## Exercises
| # | Task (book page) | Solution idea and result (App. C) |
|---|---|---|
| 7.1 | *Changing prompt styles*: after the Alpaca run, fine-tune with the Phi-3 style from Fig. 7.4 and compare the response quality (p. 209) | Rewrite `format_input` to emit `<|user|>\n{instruction}` (plus the input on a new line if present). Strip `<|assistant|>:` when extracting the response. Training is about **17% faster** (shorter prompts), and the judge score stays **about 50** (pp. 307–308). |
| 7.2 | *Instruction and input masking*: replace the instruction and input target tokens with −100 (Fig. 7.13) and check whether it helps (p. 223) | `InstructionDataset` also stores `instruction_length`. The collate function adds `targets[:instruction_length-1] = -100` (the −1 is because targets are shifted by one). The score gets **about 4 points worse**, which agrees with Shi et al. (2024) (pp. 308–309). See [[instruction-masking]]. |
| 7.3 | *Fine-tuning on the original Alpaca dataset* (52,002 entries, about 50× larger, longer entries) (p. 233) | Point the URL to `tatsu-lab/stanford_alpaca/main/alpaca_data.json` and use a GPU. If you run out of memory, lower `batch_size` 8 → 4/2/1 and/or `allowed_max_length` 1024 → 512/256 (pp. 309–310). The promised example outputs are not printed in the book; they are in the repo. |
| 7.4 | *Parameter-efficient fine-tuning with LoRA* (App. E): compare run time and performance (p. 247) | Freeze all weights, then `replace_linear_with_lora(model, rank=16, alpha=16)`. On an L4 it takes **1.30 min vs 1.80 min** (about 28% faster), with the score still **about 50** (p. 310). See [[low-rank-adaptation]], [[lora-rank-and-alpha]], [[parameter-efficient-fine-tuning]]. |

Together these give a small ablation study. The **prompt format barely matters** for the score but changes the speed. **Masking the instruction slightly hurts** on this small dataset. **LoRA matches full fine-tuning** while training only a small fraction of the parameters.

## References cited (Appendix B, pp. 296–298)
1. "Stanford Alpaca: An Instruction-Following Llama Model" (github.com/tatsu-lab/stanford_alpaca): 52,000 pairs, one of the first popular public [[instruction-dataset]]s.
2. Zhou et al. (2023), "LIMA: Less Is More for Alignment", arXiv 2305.11206: another public instruction dataset.
3. Ding et al. (2023), "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations" (UltraChat), arXiv 2305.14233: 805,000 pairs.
4. Alpaca GPT4 (mng.bz/Aa0p): 52,000 Alpaca-like pairs generated with GPT-4 instead of GPT-3.5.
5. Abdin et al. (2024), "Phi-3 Technical Report", arXiv 2404.14219: a 3.8B model whose instruct variant is comparable to GPT-3.5.
6. Xu et al. (2024), "Magpie", arXiv 2406.08464: 300,000 synthetic pairs from Llama 3 instruct. A base model tuned on them matches the official instruct model.
7. [[zhengyan-shi]] et al. (2024), "Instruction Tuning With Loss Over Instructions", arXiv 2405.14394: **not** masking instructions can help, especially when instructions are long and outputs short, or when there are few examples (see [[instruction-masking]]).
8. Kim et al. (2023), "Prometheus", arXiv 2310.08491: an open judge LLM that matches GPT-4 on long-form evaluation. The book does not use it because Ollama did not support it at the time.
9. Deshwal and Chawla (2024), "PHUDGE: Phi-3 as Scalable Judge", arXiv 2405.08029: a second open judge model.
10. Kim et al. (2024), "Prometheus 2", arXiv 2405.01535: the successor judge model.
11. Gekhman et al. (2024), "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?", arXiv 2405.05904: knowledge comes mainly from pretraining. New facts in fine-tuning are learned slowly and increase hallucination.
12. Raschka, "LLM Training: RLHF and Its Alternatives" (mng.bz/ZVPm): [[preference-fine-tuning]].
13. Raschka, "Tips for LLM Pretraining and Evaluating Reward Models" (mng.bz/RNXj): preference tuning and reward models.

Also named in the chapter text (not in App. B): MMLU (arXiv 2009.03300), LMSYS Chatbot Arena, AlpacaEval, GPT-4 and the OpenAI API, Llama 3 (Meta AI), Ollama and llama.cpp, Axolotl, LitGPT. InstructGPT ([[long-ouyang]] et al. 2022) is listed under Ch. 1 in App. B, but it is the same idea developed in this chapter.

## Errata / quirks found during validation
- **Fig. 7.12 caption (p. 219)** says "Step 2.4", but the figure shows step 2.5 (the −100 replacement).
- **p. 237** says rating "all 1,100 responses" would take a lot of effort, but the test set has **110** entries. 1,100 is the size of the whole dataset.
- **p. 241** prints the phi3 command as `ollama run llama3`. The intended command is `ollama run phi3`.
- **Fig. 7.17 (p. 233)**: the caption calls the validation line "dotted", but it is drawn dash-dot. (The same quirk appears in Fig. 6.16.)
- **Fig. 7.4 vs. JSON (p. 208)**: the figure prints `'Occasion'.` and the data prints `'Occasion.'`.
- **p. 224**: the commented-out `mps` line has a stray `"`. The judge in Example 3 oddly writes "my own response" (p. 245), a reminder that judge output can contain errors too.
- **App. C Ex. 7.1 (p. 307)**: the printed template shows `<user>`/`<assistant>` without pipes, while the code uses `<|user|>`. The `format_input` shown never adds `<|assistant|>`, but the extraction code strips `"<|assistant|>:"`.
- **App. B (p. 297)** credits "Instruction Tuning With Loss Over Instructions" to "Shi" alone, and Gekhman's paper to "Gekhman" alone. Both papers have several authors.

## What this changes in the wiki
- **Adds 14 concept pages and 1 person page** (listed above). This completes the second "pretrained model → useful task" pipeline. Ch. 6 made a *specialist* (classifier). Ch. 7 makes a *generalist* (assistant).
- **Fills in [[instruction-fine-tuning]].** Ch. 6 defined it only by contrast. That page can now cite the full recipe from this chapter.
- **Connects to the MML math.** Masked cross-entropy is [[maximum-likelihood-estimation]] over a *subset* of positions, a masked [[empirical-risk-minimization]] average. The judge's average score is a sample mean ([[expected-value-and-mean]]) of a noisy [[random-variable]], and repeated runs lower its variance ([[covariance-and-variance]]). LoRA (Ex. 7.4) is a [[low-rank-approximation]] of the weight update, $\Delta W = BA$ with rank $r\ll d$ ([[basis-and-rank]]).
- **A dispute is recorded, not hidden.** [[instruction-masking]] is common practice, but Shi et al. (2024) and the book's own Ex. 7.2 (about −4 points) both find that *not* masking works as well or better on small datasets. The page should present both sides.
- **Evaluation gets its own page.** [[llm-evaluation]] and [[llm-as-a-judge]] add a point the classification chapter did not need: for free-text output there is no simple [[classification-accuracy]], so training loss and answer quality have to be measured separately.
- **No contradictions** with existing pages. The book quirks above are recorded here so the concept pages can cite them instead of silently overwriting anything.
