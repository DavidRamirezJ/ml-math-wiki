---
title: Greedy Decoding
type: concept
tags: [llm, text-generation, decoding, probability]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# Greedy Decoding

> **One-liner:** Greedy decoding writes text by always picking the single most likely next token, one step at a time, with no randomness.

*Merged here:* argmax greedy decoding.

## 🟢 Intuition (Dummy)
Imagine you're walking through a maze and at every fork you take the path that *looks*
best right now, without ever looking further ahead. That's a **greedy** strategy.

An LLM writes text one [[token]] at a time ([[autoregressive-text-generation]]). At each
step it gives every word in its vocabulary a score: "how likely is this to come next?"
Greedy decoding simply takes the **top-scoring word**, glues it onto the text, and asks
again. No dice, no surprises.

Two consequences follow:
- It is **deterministic**: same prompt in, same text out, every time (book p. 152).
- It can be **boring or repetitive**, and it can even replay memorized text — because it
  never takes the second-best option, even when that option is nearly as good.

That's why chapter 5 adds [[temperature-scaling]] and [[top-k-sampling]]: they let the
model sometimes choose a less likely word, which "introduces variability and creativity"
(book p. 125).

## 🟡 Mechanics (Practitioner)

### The formula
At step $t$ the model outputs a [[logits|logit]] vector $\mathbf z_t$ for the **last**
position. Greedy decoding picks
$$\hat x_{t+1} = \arg\max_{i\in\{0,\dots,V-1\}} \operatorname{softmax}(\mathbf z_t)_i = \arg\max_{i} z_{t,i}$$
and appends it: $\mathbf x_{1:t+1} = [\mathbf x_{1:t},\, \hat x_{t+1}]$.

Symbol by symbol:
- $\mathbf z_t\in\mathbb R^{V}$ = the logits: one raw score per vocabulary entry; $z_{t,i}$ is the score of token $i$.
- $V$ = vocabulary size (50,257 for GPT-2), so token IDs run from 0 to 50,256.
- $\operatorname{softmax}(\mathbf z)_i=\dfrac{e^{z_i}}{\sum_{j}e^{z_j}}$ = turns scores into probabilities that are positive and sum to 1 ([[softmax]], [[probability-distribution]]).
- $\arg\max_i$ = "the index $i$ where the value is biggest" — not the biggest value itself, but its **position** ([[argmax]]). That position *is* the token ID.
- $\hat x_{t+1}$ = the chosen next token ID.
- $[\mathbf x_{1:t},\hat x_{t+1}]$ = the old sequence with the new token stuck on the end.

**Why the second "=" holds.** Softmax is [[monotonic-function|monotonic]]: "it preserves
the order of its inputs when transformed into outputs" (book p. 125). The biggest logit
always becomes the biggest probability, so you can skip softmax and take `argmax` of the
logits directly — the book says the softmax step is "redundant" here and keeps it only for
intuition (book p. 125).

### The book's implementation (Listing 4.8, book p. 124)
```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]          # crop to the supported context
        with torch.no_grad():
            logits = model(idx_cond)               # (batch, n_tokens, vocab_size)
        logits = logits[:, -1, :]                  # last time step: (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)     # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)   # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)    # (batch, n_tokens+1)
    return idx
```
- `idx[:, -context_size:]` keeps only the last `context_size` tokens ([[context-length]]). Book example: if the model supports 5 tokens and the context is 10, only the last 5 are used.
- `logits[:, -1, :]` keeps only the **last** row — the prediction for the next token.
- `torch.argmax(..., keepdim=True)` is the greedy choice; `torch.cat` appends it.

### Worked example — one step (Fig. 4.17, book p. 124)
"Hello, I am" → token IDs `[15496, 11, 314, 716]` → the model returns a $4\times50{,}257$
matrix → take the last row `[-0.4929, ..., 2.4812, ..., -0.6093]` → softmax
`[0.0001, ..., 0.0200, ..., 0.0001]` → the biggest value is at index **257** → decode
257 → "a" → append. Fig. 4.18 (book p. 126) runs six such steps:
257 "a", 2746 "model", 3492 "ready", …, 284 "to", 1037 "help", 13 "." →
"Hello, I am a model ready to help."

With the **untrained** model the same code gives
`[15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]` →
"Hello, I am Featureiman Byeswickattribute argue" (book pp. 126–127). Greedy decoding
is only as good as the probabilities it is given.

### Worked example — the toy vocabulary (§5.3.1, book pp. 152–153)
Vocabulary of 9 words; context "every effort moves you"; next-token logits
`[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]` for
`closer, every, effort, forward, inches, moves, pizza, toward, you`.
```python
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
print(inverse_vocab[next_token_id])     # forward
```
The largest logit, 6.75, sits at index 3 → "forward". Our softmax arithmetic: "forward"
≈ 0.572 and "toward" ≈ 0.358. Greedy picks "forward" **every time**, even though
"toward" has a 36% probability. With `torch.multinomial` sampling instead
([[multinomial-sampling]]), 1,000 draws gave 582 × forward, 343 × toward, 73 × closer,
2 × inches (book pp. 153–154).

### Greedy decoding across the book
- **After pretraining (§5.3, book p. 152).** After training on "The Verdict", `generate_text_simple` with 25 new tokens gives the same text every run: *"Every effort moves you know," was one of the axioms he laid down across the Sevres and silver of an exquisitely appointed lun* (book p. 152). The book later calls this "a memorized passage from the training set" (book p. 158).
- **Evaluation view (Fig. 5.4, book p. 133; p. 134).** `torch.argmax(probas, dim=-1, keepdim=True)` applied to every position shows what the model would greedily predict; e.g. "every effort moves" → `[1, 4, 5]` → "effort moves you" in the 7-word toy vocabulary.
- **`generate` (Listing 5.4, book pp. 157–158).** The new function keeps greedy as the default: with the default `temperature=0.0` (the `else` branch, `torch.argmax(logits, dim=-1, keepdim=True)` applied directly to the logits) it "carries out greedy next-token selection as before when temperature scaling is disabled" (book p. 158). An optional `top_k` filter runs first but does not change the argmax.
- **Pretrained GPT-2 sanity check (§6.4, book p. 182).** `generate_text_simple` with OpenAI's weights and 15 new tokens gives "Every effort moves you forward. The first step is to understand the importance of your work" — coherent, so the weights loaded correctly.
- **Chapter 5 summary (book p. 167).** "By default, the next token is generated by converting the model outputs into probability scores and selecting the token from the vocabulary that corresponds to the highest probability score, which is known as 'greedy decoding.'"
- **Exercise 5.3 (book p. 159; solution p. 305).** Two ways to make `generate` deterministic: set `top_k=None` and apply no temperature scaling, or set `top_k=1`.

## 🔴 Theory (Expert)
- **Greedy ≠ most likely sequence.** The model defines
  $P(x_{1:T})=\prod_t P(x_{t+1}\mid x_{1:t})$. Greedy maximizes each factor separately;
  the sequence-level optimum $\arg\max_{x_{1:T}}\sum_t\log P(x_{t+1}\mid x_{1:t})$
  ([[log-probability]]) can start with a lower-probability token that leads to much
  better continuations. Exact search is exponential in $T$ ($V^T$ sequences). [[beam-search]]
  keeps the $k$ best partial sequences as a middle ground; the book only points to it
  (appendix B, book p. 295; Vijayakumar et al., 2016).
- **Zero-temperature limit.** With [[temperature-scaling]],
  $\operatorname{softmax}(\mathbf z/T)\to$ a one-hot vector on $\arg\max_i z_i$ as $T\to0^+$
  (for a unique maximum). So greedy decoding = sampling at temperature 0, and also = top-$k$
  sampling with $k=1$ ([[top-k-sampling]]). The book observes that $T=0.1$ already picks
  "forward" almost 100% of the time (book p. 155).
- **The mode of a categorical.** Each greedy step returns the mode of the next-token
  distribution. Repeatedly taking modes is a known cause of **degenerate repetition**
  loops in open-ended text; stochastic methods ([[top-p-sampling]], top-$k$, temperature)
  were proposed partly to fix this (Fan et al., 2018 for top-$k$, cited on book p. 295).
- **Ties.** If two logits are exactly equal, `torch.argmax` returns the first index; the
  choice is deterministic but arbitrary. In practice exact ties are rare with float logits.
- **Cost.** Greedy needs one forward pass per new token and no extra bookkeeping. Without
  a KV-cache, `generate_text_simple` re-processes the whole (cropped) context each step.
- **Batch subtlety.** `generate`'s early stop `if idx_next == eos_id: break` only works
  cleanly for batch size 1 (our observation; see [[end-of-text-token]]).

## Connections
- Builds on: [[logits]], [[softmax]], [[argmax]], [[monotonic-function]], [[probability-distribution]], [[next-token-prediction]]
- Leads to: [[decoding-strategies]], [[temperature-scaling]], [[top-k-sampling]], [[multinomial-sampling]], [[top-p-sampling]], [[beam-search]]
- Related: [[autoregressive-text-generation]], [[context-length]], [[tokenizer-encode-decode]], [[train-vs-eval-mode]], [[log-probability]]
- Used by: [[gpt-architecture]], [[llm-training-loop]], [[pretrained-weights]], [[instruction-fine-tuning]]
- People: [[angela-fan]] (top-k sampling, the main alternative)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 122–127 (Fig. 4.16–4.18, Listing 4.8, monotonic softmax and "greedy decoding" on p. 125).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.1–5.1.2, book pp. 131–135 (Fig. 5.3, Fig. 5.4, argmax over positions).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3, book pp. 151–159 (deterministic output p. 152, toy logits pp. 152–154, Listing 5.4, Exercise 5.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5 chapter summary, book p. 167; §6.4, book p. 182.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, Exercise 5.3, book p. 305; appendix B, Ch. 5 references, book p. 295.
- Sequence-optimality, temperature-limit, repetition and tie remarks: standard results, not in the book.

## From llm-ch07-fine-tuning-to-follow-instructions
- In Ch. 7 every response is generated with greedy decoding. The `generate(...)` calls pass only `max_new_tokens`, `context_size` and `eos_id=50256`. They pass no `temperature` and no `top_k`, so the defaults from §5.3 apply and each step takes $\arg\max$ over the logits. Read it as: "pick the token whose score is the largest." The baseline check uses `max_new_tokens=35` (§7.5, book p. 228). The 110 test responses use `max_new_tokens=256` (§7.7, book pp. 234, 237).
- Why this matters: the same input always gives the same output, so the scores that [[llm-as-a-judge]] gives in §7.8 come from fixed text. Any remaining run-to-run noise comes from the judge, not from the fine-tuned model. See [[response-extraction]], [[llm-ch07-fine-tuning-to-follow-instructions]].
- Source: `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.5, book p. 228; §7.7, book pp. 234, 237.
