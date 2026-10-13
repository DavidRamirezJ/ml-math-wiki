---
title: Top-k Sampling
type: concept
tags: [llm, decoding, sampling, text-generation, probability]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Top-k Sampling

> **One-liner:** Before rolling the dice for the next word, throw away every word except the $k$ most likely ones, so the model can be creative without ever picking total nonsense.

## 🟢 Intuition (Dummy)
[[multinomial-sampling]] (spinning a weighted wheel) and [[temperature-scaling]]
(making the wheel's slices more even) make an LLM's text more varied. The catch: a
hot, flat wheel gives every word a chance, including silly ones — the book's model
sometimes wrote "every effort moves you **pizza**".

Top-k sampling is a bouncer at the door. It says: "Only the top $k$ candidates get on
the wheel. Everyone else's slice is zero." With $k=3$, only "forward", "toward" and
"closer" are allowed. You still get variety among sensible choices, but "pizza" can
never win.

## 🟡 Mechanics (Practitioner)

### The formula
Let $z_1,\dots,z_V$ be the [[logits]] and let $\tau_k$ be the $k$-th largest logit.
Masked logits:
$$\tilde z_i = \begin{cases} z_i & \text{if } z_i \ge \tau_k \\ -\infty & \text{if } z_i < \tau_k \end{cases}$$
Then the usual [[softmax]] (optionally with temperature $T$) and a random draw:
$$p_i = \frac{e^{\tilde z_i/T}}{\sum_{j=1}^{V} e^{\tilde z_j/T}}, \qquad x_{t+1}\sim\mathrm{Categorical}(p_1,\dots,p_V)$$

Reading it symbol by symbol:
- $z_i$ = the raw score (logit) of token $i$.
- $\tau_k$ = the cut-off: the score of the $k$-th best token (the Greek letter "tau").
- $\tilde z_i$ ("z-tilde") = the score after masking. Tokens in the top $k$ keep their
  score; all others get $-\infty$ (negative infinity).
- $e^{-\infty}=0$, so masked tokens get probability exactly 0, and the remaining $k$
  probabilities automatically sum to 1 (book p. 156). This is the same $-\infty$ trick as
  the [[causal-attention-mask]] in Chapter 3 (book p. 156).
- $T$ = the [[temperature-scaling|temperature]] (use $T=1$ if you only want top-k).
- $x_{t+1}\sim\mathrm{Categorical}(\cdot)$ = draw the next token at random with these
  probabilities ([[multinomial-sampling]]).

### The book's worked example (§5.3.2, Figure 5.15, book pp. 156–157), $k=3$
| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| token | closer | every | effort | forward | inches | moves | pizza | toward | you |
| logit | 4.51 | 0.89 | −1.90 | 6.75 | 1.63 | −1.62 | −1.89 | 6.28 | 1.79 |
| masked | 4.51 | −inf | −inf | 6.75 | −inf | −inf | −inf | 6.28 | −inf |
| softmax | 0.0615 | 0 | 0 | 0.5775 | 0 | 0 | 0 | 0.3610 | 0 |

```python
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
# Top logits:    tensor([6.7500, 6.2800, 4.5100])
# Top positions: tensor([3, 7, 0])

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],   # below the 3rd-best score?
    input=torch.tensor(float('-inf')),              # then set to -inf
    other=next_token_logits)                        # else keep
# tensor([4.5100, -inf, -inf, 6.7500, -inf, -inf, -inf, 6.2800, -inf])

topk_probas = torch.softmax(new_logits, dim=0)
# tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])
```
`torch.topk` returns the $k$ largest values (sorted, descending) and their positions;
`top_logits[-1]` is the smallest of them, i.e. $\tau_k$. Notice the surviving
probabilities (0.0615, 0.5775, 0.3610) are each a bit larger than the untruncated ones
(0.061, 0.572, 0.358): the mass that "pizza" & co. used to hold has been shared out.

### In the full generator (Listing 5.4, book pp. 157–158)
```python
if top_k is not None:
    top_logits, _ = torch.topk(logits, top_k)
    min_val = top_logits[:, -1]
    logits = torch.where(logits < min_val,
                         torch.tensor(float('-inf')).to(logits.device),
                         logits)
if temperature > 0.0:
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
else:
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
```
Order matters: filter first (top-k), then apply temperature, then sample.

Settings used in the book:
- Own 124M model trained on *The Verdict*: `top_k=25, temperature=1.4`, 15 new tokens →
  *"Every effort moves you stand to work on surprise, a one of us had gone with random-"*
  (book p. 158), instead of the memorized greedy passage.
- OpenAI GPT-2 124M weights: `top_k=50, temperature=1.5`, 25 new tokens →
  *"Every effort moves you toward finding an ideal new way to practice something! What
  makes us want to be on top of that?"* (book p. 167).

Choosing $k$ (Exercise 5.2, appendix C pp. 304–305): small $k$ (< 10) with $T<1$ for
reports, code, Q&A; larger $k$ (about 20–40) with $T>1$ for brainstorming and fiction.
With `top_k=1` the output becomes deterministic, because only one token survives
(Exercise 5.3, appendix C p. 305).

## 🔴 Theory (Expert)
- **Top-k = conditioning.** Let $S_k$ be the set of the $k$ highest-probability tokens.
  Top-k sampling draws from $p(i\mid i\in S_k)=\dfrac{p_i\,\mathbb 1[i\in S_k]}{\sum_{j\in S_k}p_j}$.
  Masking logits to $-\infty$ before softmax gives exactly this renormalized distribution,
  because the softmax denominator then only sums over $S_k$.
- **Ties.** The book's rule "$z_i <\tau_k \Rightarrow -\infty$" keeps *every* token whose
  logit equals $\tau_k$, so with ties more than $k$ tokens can survive.
- **Interaction with temperature.** Since $z\mapsto z/T$ preserves order, $S_k$ is the same
  whether you apply temperature before or after filtering; only the probabilities inside
  $S_k$ change. With the toy logits and $k=3$: $T=1$ gives (0.062, 0.578, 0.361) for
  (closer, forward, toward); $T=5$ gives (0.251, 0.392, 0.357).
- **Fixed $k$ is a blunt instrument.** When the model is confident (one token has 0.99),
  $k=50$ still admits 49 near-zero options; when it is unsure (a flat distribution over
  hundreds of plausible words), $k=50$ may cut good options. [[top-p-sampling|Top-p
  (nucleus) sampling]] adapts the set size: keep the smallest set whose cumulative
  probability exceeds $p$ (appendix B, book p. 295 — mentioned, not implemented, in the book).
- **Origin.** Top-k sampling was introduced by Fan, Lewis & Dauphin (2018), "Hierarchical
  Neural Story Generation", arXiv 1805.04833 (appendix B, book p. 295) — see [[angela-fan]].
  Appendix B also points to diverse [[beam-search]] (Vijayakumar et al. 2016) as a further
  decoding strategy not covered in Chapter 5.

## Connections
- Builds on: [[logits]], [[softmax]], [[argmax]], [[multinomial-sampling]], [[probability-distribution]]
- Shares a trick with: [[causal-attention-mask]] ($-\infty$ masking)
- Works with: [[temperature-scaling]], [[random-seed-reproducibility]]
- Leads to: [[top-p-sampling]], [[beam-search]], [[decoding-strategies]], [[autoregressive-text-generation]]
- Used by: [[gpt-2]] text generation in Ch. 5 (§5.3.3, §5.5)
- People: [[angela-fan]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.2, book pp. 155–157 (concept, $-\infty$ masking, Figure 5.15, `torch.topk`/`torch.where` code).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.3, book pp. 157–159 (Listing 5.4 `generate`, `top_k=25, temperature=1.4`, Exercises 5.2–5.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book p. 167 (`top_k=50, temperature=1.5` with GPT-2 weights).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, book p. 295 (Fan et al. 2018; top-p sampling; diverse beam search).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 304–305 (solutions to Exercises 5.2–5.3).
