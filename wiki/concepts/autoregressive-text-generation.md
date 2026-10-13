---
title: Autoregressive Text Generation
type: concept
tags: [llm, text-generation, probability, decoding, gpt]
difficulty: intermediate
sources: [llm-ch01-understanding-large-language-models, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch07-fine-tuning-to-follow-instructions, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Autoregressive Text Generation

> **One-liner:** A GPT-style model writes text one token at a time, and every new token it writes gets glued onto the input so it can be used to pick the next one.

## 🟢 Intuition (Dummy)
Picture a very fast game of "finish my sentence." You say *"Hello, I am"*. The model
says *"a"*. Now the sentence is *"Hello, I am a"*, and the model looks at the **whole
thing** again and adds *"model"*. Then *"ready"*, then *"to"*, *"help"*, *"."* — six
rounds later you have *"Hello, I am a model ready to help."* (Figure 4.16, book
pp. 122–123).

That's all "autoregressive" means: *auto* = self, *regressive* = predicting from what
came before. The model feeds on its **own previous outputs**. Because each new word
is chosen while looking at everything already written, the text hangs together — the
book says this improves the **coherence** of the generated text (book pp. 12–13).

Two things to keep in mind:
- The model never "plans" the full sentence. It only ever answers "what's the next
  token?" — and repeats.
- *How* you pick that next token matters a lot. Always grabbing the single most
  likely one is safe but boring (and can parrot training text). Rolling weighted dice
  gives more creative text. That choice is called the **decoding strategy**.

## 🟡 Mechanics (Practitioner)

### The probability rule behind it
The model approximates the probability of the next token given the past. By the
chain rule of probability, a whole sequence factorises into these next-token pieces:
$$P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_1,\dots,x_{t-1}).$$
Reading it symbol by symbol:
- $x_1,\dots,x_T$ = the tokens of a text, $T$ of them in total; $x_t$ is the one at position $t$.
- $P(x_1,\dots,x_T)$ = how likely the whole text is.
- $\prod_{t=1}^{T}$ = "multiply together, for $t$ running from 1 to $T$."
- $P(x_t\mid x_1,\dots,x_{t-1})$ = "probability of token $x_t$ **given** all tokens before it."

(The book describes this in words; the formula is standard notation added by the wiki.)
Generation just walks this product left to right: sample $x_t$ from the conditional,
append it, move on. That's why GPT reads strictly **left to right** (unidirectional,
Figure 1.8 caption, book p. 13) — enforced inside the model by the
[[causal-attention-mask]].

### One generation step, with the book's numbers (Figure 4.17, book p. 124)
1. **Encode** "Hello, I am" → token IDs `[15496, 11, 314, 716]`.
2. The GPT model returns a **4 × 50,257 matrix** of [[logits]]: one row per input
   position, one column per vocabulary entry ($V = 50{,}257$ for GPT-2).
3. **Keep only the last row** — it scores the *next* token:
   `[-0.4929, ..., 2.4812, ..., -0.6093]`.
4. **[[softmax]]** turns the scores into probabilities:
   $$p_i=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}$$
   - $z_i$ = logit (raw score) of vocabulary token $i$,
   - $e^{z_i}$ = the exponential, which makes every score positive,
   - $\sum_{j=1}^{V}e^{z_j}$ = the sum over all $V$ tokens, so the $p_i$ add up to 1.
   Result: `[0.0001, ..., 0.0200, ..., 0.0001]`.
5. **Pick the token.** Greedy decoding takes the biggest:
   $$\hat x_{t+1}=\arg\max_{i}\,p_i=\arg\max_i\,z_i$$
   - $\hat x_{t+1}$ = the chosen token ID for the next position (the hat means "our pick"),
   - $\arg\max_i$ = "the index $i$ where the value is largest."
   The two are equal because softmax is **monotonic** — it keeps the order of its
   inputs (book p. 125). Here the winner is ID **257** = "a".
6. **Append** 257 to the input and repeat.

Six iterations give the IDs 257 "a", 2746 "model", 3492 "ready", 284 "to", 1037
"help", 13 "." (Figure 4.18, book p. 126).

### The book's code — Listing 4.8 (book p. 124)
```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]      # crop to what the model supports
        with torch.no_grad():
            logits = model(idx_cond)           # (batch, n_tokens, vocab_size)
        logits = logits[:, -1, :]              # last position only -> (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)   # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)                 # (batch, n_tokens+1)
    return idx
```
- `idx[:, -context_size:]` is the **context window**: if the model supports 5 tokens and
  you have 10, only the last 5 are used (book p. 125). In symbols the model sees
  $x_{\max(1,t-C+1)},\dots,x_t$, with $C$ = [[context-length]].
- `torch.cat` is the "auto" in autoregressive: the output goes back in as input.

**Untrained vs. trained.** With random weights the same loop prints gibberish:
*"Hello, I am Featureiman Byeswickattribute argue"* (book p. 127). The loop is fine;
the conditional probabilities are garbage until [[pretraining]] fixes them.

### Decoding strategies (§5.3, book pp. 151–159)
After pretraining on the tiny "The Verdict" story, greedy decoding regurgitates the
training text: *"Every effort moves you know," was one of the axioms he laid down…*
(book p. 152). Two tools add controlled randomness:

**[[temperature-scaling]]** — divide the logits by $T>0$ before softmax:
$$p_i(T)=\frac{e^{z_i/T}}{\sum_{j=1}^{V}e^{z_j/T}}$$
- $T=1$: plain softmax; $T<1$: sharper, more confident; $T>1$: flatter, more uniform (book p. 154).

Toy example (book p. 153): 9-word vocabulary, logits
`[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]` after "every effort moves you".
Greedy always says "forward". [[multinomial-sampling]] 1,000 times at $T=1$ gives
**582 × forward, 343 × toward, 73 × closer, 2 × inches** (book pp. 153–154). At
$T=0.1$ it's "forward" almost 100% of the time; at $T=5$ "pizza" shows up ~4% of the
time (book p. 155) — creative, but nonsense.

**[[top-k-sampling]]** — keep only the $k$ largest logits, set the rest to $-\infty$
(so $e^{-\infty}=0$), then softmax. With $k=3$:
`[4.51, -inf, -inf, 6.75, -inf, -inf, -inf, 6.28, -inf]` →
`[0.0615, 0, 0, 0.5775, 0, 0, 0, 0.3610, 0]` (book pp. 156–157).

**The upgraded loop — `generate` (Listing 5.4, book pp. 157–158)** adds `top_k`,
`temperature` (default 0.0 = greedy) and `eos_id` (stop early when the
[[end-of-text-token]] appears). With `top_k=25, temperature=1.4` the small model
writes *"Every effort moves you stand to work on surprise, a one of us had gone with
random-"* (book p. 158). With OpenAI's GPT-2 124M weights and `top_k=50,
temperature=1.5`: *"Every effort moves you toward finding an ideal new way to practice
something! What makes us want to be on top of that?"* (book p. 167).

In Ch. 7 the same `generate` produces answers to instructions with
`max_new_tokens=256, eos_id=50256`; since it returns prompt + answer, the response is
recovered by slicing off the prompt, `generated_text[len(input_text):]` (book
pp. 234–235). Appendix E reuses `generate_text_simple` as a sanity check that
pretrained weights loaded correctly (book p. 327).

## 🔴 Theory (Expert)
- **Exact factorisation, approximate model.** The chain-rule product is an identity
  for *any* joint distribution over sequences; "autoregressive" is a modelling choice
  to parameterise each factor $p_\theta(x_t\mid x_{<t})$ with one shared network.
  Training by [[maximum-likelihood-estimation]] maximises
  $\sum_t\log p_\theta(x_t\mid x_{<t})$, i.e. minimises [[cross-entropy-loss]] (§5.1).
- **Teacher forcing vs. free running.** During training every position sees the *true*
  prefix (all $T$ next-token losses computed in one parallel pass thanks to the causal
  mask). At generation time the model conditions on its *own* samples, so errors can
  compound — "exposure bias" (general background, not in the book).
- **Decoding ≠ the model.** Greedy and beam search approximate the mode
  $\arg\max_{\mathbf y}\prod_t p(y_t\mid y_{<t})$ (greedy is only locally optimal);
  sampling draws from the distribution itself. Temperature is the Boltzmann/Gibbs
  trick: $p_i\propto e^{z_i/T}$; $T\to0^+$ gives a one-hot argmax, $T\to\infty$ gives
  the uniform distribution $1/V$. Top-k truncates and renormalises — conditioning the
  categorical on its $k$ most probable outcomes. See [[decoding-strategies]],
  [[top-p-sampling]], [[beam-search]].
- **Determinism (Exercise 5.3, book p. 159).** `temperature=0.0` takes the argmax
  path regardless of `top_k`; `top_k=1` also leaves a single non-zero probability
  (the wiki's answer, not printed in the section).
- **Cost.** Each step re-runs the model on up to $C$ tokens; naive generation of $n$
  tokens is therefore $O(n)$ forward passes, each quadratic in context length because
  of attention. KV caching avoids recomputing past keys/values (outside the main
  text of the book).
- **Encoder–decoder precursor.** The original transformer already generated
  translations one word at a time, conditioning also on the source: $p(y_t\mid
  y_{<t},\mathbf x)$ (Figure 1.4, book p. 8). GPT drops the encoder and conditions
  only on the prefix (book p. 12) — see [[decoder-only-transformer]].

## Connections
- Builds on: [[next-token-prediction]], [[probability-distribution]], [[softmax]], [[logits]], [[argmax]], [[monotonic-function]], [[causal-attention-mask]], [[context-length]]
- Leads to: [[greedy-decoding]], [[decoding-strategies]], [[temperature-scaling]], [[top-k-sampling]], [[multinomial-sampling]], [[top-p-sampling]], [[beam-search]], [[response-extraction]]
- Used by: [[gpt-architecture]], [[decoder-only-transformer]], [[text-completion]], [[instruction-fine-tuning]], [[llm-applications]], [[machine-translation]]
- Related: [[pretraining]], [[cross-entropy-loss]], [[maximum-likelihood-estimation]], [[random-variable]], [[end-of-text-token]], [[tokenizer-encode-decode]]
- People: [[alec-radford]], [[ashish-vaswani]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book p. 8 (Figure 1.4, word-by-word translation); §1.6, book pp. 12–13 (autoregressive, Figure 1.8).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 122–127 (Figures 4.16–4.18, Listing 4.8, greedy decoding, monotonic softmax).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book pp. 130–132 (generation cycle Figure 5.3); §5.3, book pp. 151–159 (temperature, top-k, Listing 5.4); §5.5, book p. 167 (GPT-2 output).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book p. 228; §7.7, book pp. 234–237 (generation for instruction responses).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.3, book p. 327 (generation as weight-loading check).
