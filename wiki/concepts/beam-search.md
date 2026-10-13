---
title: Beam Search
type: concept
tags: [llm, decoding, search, text-generation, probability]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Beam Search

> **One-liner:** Instead of committing to the single best next word at each step, beam search keeps a few of the best partial sentences alive in parallel and picks the overall winner at the end.

## 🟢 Intuition (Dummy)
Imagine finding the best route through a city, one junction at a time.
- **Greedy** ([[greedy-decoding]]): at every junction take the street that looks best *right now*.
  Quick, but a nice-looking street can lead to a dead end.
- **Check every route**: guaranteed best, but the number of routes explodes.
- **Beam search**: keep, say, the **3 best routes so far** (the "beam"). At each junction extend
  all 3 in every direction, score them, and again keep only the best 3. At the end, take the best
  full route.

For an LLM, a "route" is a partial sentence and the score is how probable the model thinks that
whole sentence is. Beam search is a middle ground between "fast but short-sighted" and "perfect but
impossible".

The book does **not** implement beam search. It lists it as further reading: "an alternative
decoding algorithm that generates output sequences by keeping only the top-scoring partial
sequences at each step to balance efficiency and quality" (book p. 295).

## 🟡 Mechanics (Practitioner)

### Scoring a sequence
A GPT model gives next-token probabilities $p(x_t\mid x_{<t})$. By the chain rule of probability,
a whole continuation $y=(y_1,\dots,y_T)$ after a prompt $x$ has probability

$$p(y\mid x)=\prod_{t=1}^{T} p(y_t\mid x, y_{<t}) \quad\Longrightarrow\quad \text{score}(y)=\sum_{t=1}^{T}\log p(y_t\mid x, y_{<t})$$

Reading it symbol by symbol:
- $x$ = the prompt (e.g. "Every effort moves you");
- $y_t$ = the token generated at step $t$; $y_{<t}$ = all tokens generated before step $t$;
- $p(y_t\mid x,y_{<t})$ = the model's probability for $y_t$ given everything so far (one softmax output);
- $\prod_{t=1}^{T}$ = "multiply these together for $t=1$ to $T$";
- $\log$ turns the product into a sum $\sum$ ([[log-probability]]), which avoids tiny numbers
  underflowing to 0 ([[numerical-stability]]). Higher (closer to 0) is better.

### The algorithm (beam width $B$)
1. Start with one hypothesis: the prompt, score 0.
2. For each of the (up to) $B$ hypotheses, run the model, take the last-position logits
   (as in the book's `generate`, book p. 157), and compute log-probabilities for all $V$ tokens.
3. Form all $B\times V$ extensions; each new score = old score + $\log p(\text{new token})$.
4. Keep the $B$ highest-scoring extensions. Hypotheses that produced the end token
   (`eos_id`, e.g. 50256 `<|endoftext|>`) are set aside as finished.
5. Repeat until `max_new_tokens` or all beams are finished; return the best finished sequence.

With $B=1$ this is exactly greedy decoding (the book's default `temperature=0.0` path).

### Worked example (step 1 from the book, step 2 invented for illustration)
Step 1 uses the book's toy next-token distribution after "every effort moves you" (§5.3, softmax
of the book's logits, computed here): forward 0.572, toward 0.358, closer 0.061, …

Step 2 probabilities are **made up** to show the idea:
- after "… forward": best next token has $p=0.40$
- after "… toward": best next token has $p=0.90$

| path | probability | log-score |
|---|---|---|
| forward → (best) | $0.572\times0.40=0.229$ | $-0.559-0.916=-1.475$ |
| toward → (best) | $0.358\times0.90=0.322$ | $-1.027-0.105=-1.133$ |

Greedy takes "forward" at step 1 and is stuck with 0.229. Beam search with $B=2$ kept "toward"
alive and finds the more probable two-token continuation, 0.322.

### Beam search vs sampling (the book's methods)
| | greedy | beam search | temperature / top-k / [[top-p-sampling]] |
|---|---|---|---|
| Random? | no | no | yes |
| Looks ahead? | no | yes, $B$ paths | no |
| Typical use | quick tests | translation, summarization, short factual answers | open-ended, creative text |
| In the book? | yes (§5.3) | reading list only (p. 295) | top-k + temperature yes (§5.3); top-p reading list only |

## 🔴 Theory (Expert)
- **What it approximates.** The ideal "most probable output" is
  $\hat y=\arg\max_y p(y\mid x)$, a maximum-a-posteriori search over $V^{T}$ sequences
  ($50{,}257^{T}$ for GPT-2), which is intractable. Beam search is a heuristic, pruned breadth-first
  search: cost $O(T\cdot B\cdot V)$ score evaluations and $B$ forward passes per step, instead of
  exponential. It is **not guaranteed** to find the $\arg\max$; a sequence pruned early cannot come
  back ([[argmax]]).
- **Length bias.** Each extra token adds a negative log-probability, so raw scores favour short
  outputs. A common fix is length normalization, $\text{score}(y)/|y|^{\alpha}$ with
  $\alpha\in[0,1]$.
- **Diversity problem and Diverse Beam Search.** The $B$ beams often differ only in a word or two.
  Vijayakumar et al. (2016), the paper the book cites (book p. 295), split the beam into groups and
  add a dissimilarity penalty between groups, so the returned candidates are genuinely different.
- **Why high probability ≠ good open-ended text.** For story-like generation, maximizing likelihood
  tends to produce bland, repetitive text, which is why chat LLMs usually sample
  (temperature, top-k, top-p) instead. Beam search shines when there is roughly one correct answer,
  such as [[machine-translation]].
- **Batched implementation.** The $B$ hypotheses are run as a batch of size $B$ through the same
  model; the book's batch dimension in `idx` of shape (batch, n_tokens) is exactly what makes this
  cheap on a GPU ([[gpu-computing]]).

## Connections
- Builds on: [[autoregressive-text-generation]], [[next-token-prediction]], [[softmax]], [[log-probability]], [[greedy-decoding]]
- Leads to: [[decoding-strategies]]
- Compare: [[top-k-sampling]], [[top-p-sampling]], [[temperature-scaling]], [[multinomial-sampling]]
- Related: [[argmax]], [[maximum-likelihood-estimation]], [[machine-translation]], [[end-of-text-token]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Chapter 5 references, book p. 295 (beam search definition; "not covered in chapter 5"; Vijayakumar et al. 2016, arXiv 1610.02424).
- Same PDF, §5.3, book pp. 152–158 (greedy decoding, toy vocabulary and logits, `generate` with `eos_id`).
- Algorithm, scoring formula, length normalization, worked example (step 2 invented): standard background, not printed in the book.
