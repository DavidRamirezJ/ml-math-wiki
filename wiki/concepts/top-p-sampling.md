---
title: Top-p (Nucleus) Sampling
type: concept
tags: [llm, decoding, probability, text-generation]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Top-p (Nucleus) Sampling

> **One-liner:** When an LLM picks its next word, top-p sampling lets it choose only from the smallest group of most-likely words that together cover, say, 90% of the probability, and throws the rest away.

## 🟢 Intuition (Dummy)
At every step, a language model gives each word in its vocabulary a "how likely is this next?"
score. Picking only the single top word ([[greedy-decoding]]) is safe but boring. Picking from
*all* words sometimes produces nonsense like "every effort moves you pizza" (book p. 155).

[[top-k-sampling]] fixes this with a fixed-size shortlist: "only the top 3 words may compete".
But a fixed size is clumsy:
- when the model is very sure (one word has 95%), even 3 candidates let in junk;
- when the model is unsure (twenty words are all plausible), 3 candidates is too strict.

**Top-p** uses a *flexible* shortlist. Line the words up from most to least likely and keep adding
them until their probabilities add up past a threshold $p$ (e.g. 0.9). That group is the
**nucleus**. Confident step → tiny nucleus. Unsure step → big nucleus. Then draw the next word at
random from the nucleus only.

The book mentions top-p only as further reading. It is **not implemented in Chapter 5** (book
p. 295). This page fills in the details so you can compare it with the top-k code the book does build.

## 🟡 Mechanics (Practitioner)

### The book's definition
Top-p sampling "selects from the smallest set of top tokens whose cumulative probability exceeds a
threshold $p$, while top-k sampling picks from the top $k$ tokens by probability" (book p. 295).

### As formulas
Let $p_1,\dots,p_V$ be the next-token probabilities from the [[softmax]] of the [[logits]]
(optionally after [[temperature-scaling]]). Sort them from largest to smallest:
$p_{(1)}\ge p_{(2)}\ge\dots\ge p_{(V)}$. Then

$$k^* = \min\Big\{k : \sum_{j=1}^{k} p_{(j)} \ge p\Big\}, \qquad \mathcal V^{(p)} = \{(1),\dots,(k^*)\}$$

$$\tilde p_i = \begin{cases}\dfrac{p_i}{\sum_{j\in\mathcal V^{(p)}} p_j} & i\in\mathcal V^{(p)}\\[2mm] 0 & \text{otherwise}\end{cases}$$

Reading it symbol by symbol:
- $V$ = vocabulary size (50,257 for GPT-2);
- $p_i$ = probability of token $i$; $p_{(j)}$ with brackets = the $j$-th **largest** probability;
- $\sum_{j=1}^{k} p_{(j)}$ = "add up the top $k$ probabilities" (the cumulative probability);
- $p$ (no subscript) = the threshold you choose, e.g. 0.9 — careful, same letter, different job;
- $\min\{k : \dots\}$ = "the smallest $k$ for which the sum reaches $p$";
- $\mathcal V^{(p)}$ = the nucleus: the set of the top $k^*$ tokens;
- $\tilde p_i$ = the new probability: inside the nucleus, divide by the nucleus total so they add up
  to 1 again; outside, 0.

Then sample: $x_{t+1}\sim\text{Categorical}(\tilde p)$ ([[multinomial-sampling]]).

(The book says "exceeds"; implementations usually use "reaches or exceeds" ($\ge$). The difference
only matters on exact ties.)

### Worked example with the book's toy vocabulary
The book's 9-token toy example (§5.3) uses the logits
`[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]` for
`closer, every, effort, forward, inches, moves, pizza, toward, you` (book p. 153; repeated in Figure 5.15, p. 156). Softmax at
temperature 1 (computed here) and the running sum:

| rank | token | $p_i$ | cumulative |
|---|---|---|---|
| 1 | forward | 0.5721 | 0.5721 |
| 2 | toward | 0.3576 | **0.9297** |
| 3 | closer | 0.0609 | **0.9906** |
| 4 | you | 0.0040 | 0.9946 |
| 5 | inches | 0.0034 | 0.9980 |
| … | every, moves, pizza, effort | ≤ 0.0016 each | 1.0000 |

- **$p = 0.9$:** the sum passes 0.9 at rank 2, so the nucleus is {forward, toward}. Renormalized:
  forward $0.5721/0.9297 = 0.615$, toward $0.385$.
- **$p = 0.95$:** we need rank 3, so the nucleus is {forward, toward, closer}. Renormalized:
  $0.5775, 0.3610, 0.0615$, exactly the book's **top-k = 3** result (book p. 157).

Here top-p with $p=0.95$ and top-k with $k=3$ agree. On a flatter distribution, the same
$p=0.95$ would keep many more tokens, while $k=3$ would still keep three. That adaptivity is the
whole point.

### A PyTorch sketch (not from the book)
This slots into the book's `generate` function (Listing 5.4) next to the top-k block, before the
temperature/multinomial step:

```python
def top_p_filter(logits, top_p=0.9):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cum_probs > top_p                  # tokens past the threshold
    remove[..., 1:] = remove[..., :-1].clone()  # shift right: keep the token that crosses p
    remove[..., 0] = False                      # always keep the top token
    sorted_logits[remove] = float("-inf")       # same -inf masking trick as top-k
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
```

Like the book's top-k code, it masks rejected tokens with $-\infty$, so after softmax they get
probability exactly 0 ($e^{-\infty}=0$, the same trick as the [[causal-attention-mask]] of §3.5.1, book p. 156).

## 🔴 Theory (Expert)
- **Truncation + renormalization = conditioning.** $\tilde p$ is the model distribution conditioned
  on the event $\{x\in\mathcal V^{(p)}\}$: $\tilde p(x)=p(x\mid x\in\mathcal V^{(p)})$. Top-k is the
  same operation with a fixed-cardinality set; top-p fixes the *mass* instead
  ([[probability-distribution]]).
- **Minimality and uniqueness.** Taking tokens in sorted order makes the nucleus the smallest set
  (by count) with mass $\ge p$: any other set of $k$ tokens has mass $\le\sum_{j\le k}p_{(j)}$. With
  ties in probability the set is unique only up to tie-breaking.
- **Edge cases.** $p\to0$ keeps only the top token, so top-p becomes [[greedy-decoding]] (when
  sampling would otherwise be random). $p=1$ keeps every token with nonzero probability: plain
  sampling. Combining with temperature matters: order of operations (temperature first, then
  nucleus) changes which tokens survive, because $T$ reshapes the cumulative sums.
- **Why truncate at all.** A softmax never gives exactly 0, so over tens of thousands of tokens the
  long "tail" of tiny probabilities can hold real mass. Drawing from it now and then derails long
  generations. Holtzman et al. (2020), "The Curious Case of Neural Text Degeneration", proposed
  nucleus sampling for this reason (background; that paper is not in the book's bibliography, which
  points to Wikipedia instead).
- **Relation to entropy.** The nucleus size $k^*$ tracks the model's uncertainty: a low-entropy
  (peaked) distribution gives small $k^*$, a high-entropy (flat) one gives large $k^*$. Top-k
  ignores this signal.

## Connections
- Builds on: [[softmax]], [[logits]], [[probability-distribution]], [[multinomial-sampling]], [[temperature-scaling]], [[top-k-sampling]]
- Leads to: [[decoding-strategies]], [[autoregressive-text-generation]]
- Compare: [[greedy-decoding]], [[beam-search]]
- Related: [[cumulative-distribution-function]], [[random-variable]], [[causal-attention-mask]]
- People: [[angela-fan]] (top-k sampling, the method top-p refines)

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Chapter 5 references, book p. 295 (definition of top-p vs top-k; "not covered in chapter 5"; Wikipedia "Top-p sampling").
- Same PDF, §5.3.2–5.3.3, book pp. 155–158 (toy logits, −inf masking, top-k = 3 probabilities 0.0615/0.5775/0.3610, `generate` Listing 5.4).
- Worked table, formulas and PyTorch sketch: derived here, not printed in the book. Holtzman et al. (2020) is background.
