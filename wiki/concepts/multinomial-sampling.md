---
title: Multinomial (Categorical) Sampling
type: concept
tags: [probability, sampling, llm, decoding, text-generation]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Multinomial (Categorical) Sampling

> **One-liner:** Instead of always picking the most likely next word, roll a loaded die whose sides are weighted by the model's probabilities, so likely words usually win but not every time.

## 🟢 Intuition (Dummy)
Picture a spinning prize wheel. Every word the model knows gets a slice, and the
slice size is that word's probability. "forward" gets a big slice (57%), "toward" a
medium one (36%), "closer" a thin one (6%), and "pizza" a sliver you can barely see.
Spin the wheel, and wherever the pointer lands, that's your next word.

Most spins land on "forward". But now and then you'll get "toward" or "closer". That's
the whole point: always picking the biggest slice (called [[greedy-decoding]]) makes
the model say the exact same thing every time and parrot its training text. Spinning
the wheel makes its writing varied and more original.

The book's word for "spin the wheel" is **multinomial sampling**, because PyTorch's
function for it is called `torch.multinomial`.

## 🟡 Mechanics (Practitioner)

### The formula
The model gives [[logits]] $z_1,\dots,z_V$; [[softmax]] turns them into probabilities
$p_1,\dots,p_V$. Greedy decoding then does:
$$\hat{x}_{t+1} = \arg\max_i\; p_i$$
Multinomial (categorical) sampling instead does:
$$x_{t+1} \sim \mathrm{Categorical}(p_1,\dots,p_V), \qquad P(x_{t+1}=i) = p_i$$

Reading it symbol by symbol:
- $x_{t+1}$ = the ID of the next token (position $t+1$ in the text). It is a
  [[random-variable]] — its value comes out of a random draw.
- $\sim$ = "is drawn from" / "is distributed as".
- $\mathrm{Categorical}(p_1,\dots,p_V)$ = the distribution over $V$ categories (here, the
  $V$ tokens in the [[vocabulary]]) where category $i$ has probability $p_i$.
- $P(x_{t+1}=i)=p_i$ = "the chance the next token is token $i$ is exactly the model's
  probability $p_i$" — the selection is **proportional to the probability score**
  (book p. 153).
- $\arg\max_i$ (in the greedy line) = "the index $i$ with the largest value"
  ([[argmax]]).

### The book's toy example (§5.3.1, book pp. 152–154)
Vocabulary `{"closer":0, "every":1, "effort":2, "forward":3, "inches":4, "moves":5,
"pizza":6, "toward":7, "you":8}` and logits after "every effort moves you":
`[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]`.

```python
probas = torch.softmax(next_token_logits, dim=0)

next_token_id = torch.argmax(probas).item()          # greedy
print(inverse_vocab[next_token_id])                  # forward

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()   # sampling
print(inverse_vocab[next_token_id])                  # forward (still the most likely)
```

To see the randomness, the book samples 1,000 times and counts the results with
`torch.bincount`:
```python
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
```
Output (book pp. 153–154):

| token | closer | every | effort | forward | inches | moves | pizza | toward |
|---|---|---|---|---|---|---|---|---|
| count out of 1,000 | 73 | 0 | 0 | **582** | 2 | 0 | 0 | 343 |
| true $p_i$ (softmax) | 0.061 | 0.002 | 0.000 | 0.572 | 0.003 | 0.000 | 0.000 | 0.358 |

("you" isn't printed: `bincount` only goes up to the largest ID actually drawn, 7.)
So if the generator used `multinomial` instead of `argmax`, it would sometimes write
"every effort moves you toward", "… inches" or "… closer" instead of "… forward"
(book p. 154). The counts are close to — but not equal to — $1000\times p_i$; that gap is
the subject of [[monte-carlo-estimation]].

**Exercise 5.1 (solution in appendix C, p. 304).** Running `print_sampled_tokens` on the
[[temperature-scaling|temperature-scaled]] probabilities: "pizza" is drawn 0× at $T=0.1$
but 32× out of 1,000 at $T=5$ — an estimate of 3.2%, against the exact softmax value
4.3% (`scaled_probas[2][6]`). Reading $p_i$ directly is both faster and exact.

### Where it sits in the generator (Listing 5.4, book pp. 157–158)
```python
if temperature > 0.0:
    logits = logits / temperature                         # temperature scaling
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)    # (batch, 1)
else:
    idx_next = torch.argmax(logits, dim=-1, keepdim=True) # greedy
```
Sampling is almost always combined with [[temperature-scaling]] (reshape the
probabilities) and [[top-k-sampling]] (throw away all but the $k$ best tokens first).
`torch.manual_seed(123)` makes the "random" draws repeatable
([[random-seed-reproducibility]]).

### How to get deterministic output again (Exercise 5.3, appendix C p. 305)
Either leave `temperature` at 0 / `top_k=None` (pure greedy), or set `top_k=1` so only
one token survives and the "random" draw has only one possible outcome.

## 🔴 Theory (Expert)
- **Categorical vs. multinomial.** One draw with probabilities $\mathbf p$ is a
  *categorical* random variable (a multinomial with $n=1$ trial). The vector of counts
  from $n$ independent draws, $(N_1,\dots,N_V)$, follows a *multinomial* distribution:
  $$P(N_1=k_1,\dots,N_V=k_V)=\frac{n!}{k_1!\cdots k_V!}\prod_{i=1}^V p_i^{k_i},\qquad \sum_i k_i=n.$$
  The book's bincount output $(73,0,0,582,2,0,0,343,0)$ is one sample from
  $\mathrm{Multinomial}(1000,\mathbf p)$, with $\mathbb E[N_i]=np_i$ and
  $\operatorname{Var}(N_i)=np_i(1-p_i)$ (see [[expected-value-and-mean]],
  [[covariance-and-variance]]).
- **How the draw is implemented: inverse-CDF sampling.** Build the cumulative sums
  $F_i=\sum_{j\le i}p_j$ (the [[cumulative-distribution-function]] of the categorical),
  draw $u\sim\mathrm{Uniform}(0,1)$, and return the smallest $i$ with $u\le F_i$. Each
  token's interval $(F_{i-1},F_i]$ has length $p_i$, so $P(\text{return } i)=p_i$ exactly.
  (Alternative: the **Gumbel-max trick** — $\arg\max_i(z_i+g_i)$ with i.i.d. Gumbel noise
  $g_i$ is distributed as $\mathrm{softmax}(z)$, so sampling equals "argmax of noisy
  logits".)
- **Ancestral sampling of sequences.** Repeating the draw token by token samples a
  whole sequence from the model's joint distribution, by the chain rule of probability:
  $p(x_1,\dots,x_n)=\prod_t p(x_t\mid x_{<t})$. Greedy decoding instead approximates the
  *mode* step by step — and the greedy sequence need not be the most probable sequence
  (motivation for [[beam-search]]).
- **Why sampling reduces parroting.** The book's small model, trained 10 epochs on one
  short story, memorized passages ([[overfitting]]); greedy decoding then deterministically
  replays them (book pp. 151–152). Sampling spreads probability mass over alternatives and
  breaks the exact replay.
- **Tail risk.** Plain sampling gives every token with $p_i>0$ a chance; over the
  50,257-token GPT-2 vocabulary, many tiny tails add up to a noticeable probability of a
  junk token. Truncation ([[top-k-sampling]], [[top-p-sampling]]) fixes this by
  conditioning the categorical on a subset and renormalizing.

## Connections
- Builds on: [[random-variable]], [[probability-distribution]], [[softmax]], [[logits]]
- Contrasts with: [[greedy-decoding]], [[argmax]]
- Works with: [[temperature-scaling]], [[top-k-sampling]], [[random-seed-reproducibility]]
- Leads to: [[decoding-strategies]], [[autoregressive-text-generation]], [[monte-carlo-estimation]], [[top-p-sampling]], [[beam-search]]
- Related: [[histogram]] (the `bincount` frequency table), [[cumulative-distribution-function]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3 intro and §5.3.1, book pp. 151–155 (greedy vs. `torch.multinomial`, 1,000-sample bincount 73/582/343/2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.3, book pp. 157–159 (Listing 5.4 `generate`, Exercise 5.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 304–305 (solutions to Exercises 5.1 and 5.3).
