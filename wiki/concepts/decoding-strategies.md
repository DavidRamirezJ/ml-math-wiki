---
title: Decoding Strategies
type: concept
tags: [llm, text-generation, sampling, probability]
difficulty: intermediate
sources: [llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-10-13
updated: 2026-10-13
---

# Decoding Strategies

> **One-liner:** A decoding strategy is the rule for picking the next word from the model's list of probabilities: always take the top one, or roll weighted dice, maybe after reshaping the odds.

## 🟢 Intuition (Dummy)
After each step, the LLM hands you a menu: "next word 'forward' 57%, 'toward'
36%, 'closer' 6%, 'pizza' almost 0%…". The model does not choose. **You** choose,
with a decoding strategy.

- **Greedy:** always order the most popular dish. Safe and repeatable, but
  boring. A model trained on a small book will just recite that book.
- **Sampling:** roll a die weighted by the percentages. Usually you get
  "forward", sometimes "toward". Now the text varies.
- **Temperature:** a knob that reshapes the odds before you roll. Turn it down,
  and the favourite gets even more likely (almost greedy). Turn it up, and the
  odds even out, so you get more surprise (and more "pizza").
- **Top-k:** cross every dish off the menu except the top *k* before rolling,
  so silly options can never come up.

The book combines these to get text that is both varied and sensible.

## 🟡 Mechanics (Practitioner)
Each generation step gives a [[logits|logit]] vector $\mathbf z\in\mathbb R^{V}$
for the **last** position ($V=50{,}257$ for GPT-2). The book's toy example uses a
9-word vocabulary and these logits for the context "every effort moves you"
(book pp. 152–153):

| index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| token | closer | every | effort | forward | inches | moves | pizza | toward | you |
| logit $z_i$ | 4.51 | 0.89 | −1.90 | 6.75 | 1.63 | −1.62 | −1.89 | 6.28 | 1.79 |

### 1. Greedy decoding (§4.7, book pp. 124–125)
$$\hat x_{t+1}=\arg\max_{i}\;p_i=\arg\max_i\;z_i$$
- $\hat x_{t+1}$ = the token ID we choose for the next position $t+1$ (the hat means "our pick").
- $\arg\max_i$ = "the index $i$ where the value is largest" (see [[argmax]]).
- $p_i$ = softmax probability of token $i$.
- The second "=" holds because [[softmax]] is a [[monotonic-function|monotonic]]
  (order-keeping) function, so softmax is not even needed for argmax (book p. 125).

Toy result: index 3, "forward". Same input → same output, every time (book p. 152).
This is [[greedy-decoding]].

### 2. Multinomial sampling (book p. 153)
$$x_{t+1}\sim\operatorname{Categorical}(p_1,\dots,p_V),\qquad P(x_{t+1}=i)=p_i$$
- $\sim$ = "is drawn at random from".
- $\operatorname{Categorical}(p_1,\dots,p_V)$ = a weighted die with $V$ faces, face $i$ having probability $p_i$.
- $P(x_{t+1}=i)=p_i$ = "the chance we pick token $i$ is exactly its probability".

`torch.multinomial(probas, num_samples=1)` does this. Sampling 1,000 times with
seed 123 gave **582 × forward, 343 × toward, 73 × closer, 2 × inches**, and 0 for
the rest (book pp. 153–154). This is [[multinomial-sampling]].

### 3. Temperature scaling (book pp. 154–155)
$$p_i(T)=\frac{e^{z_i/T}}{\sum_{j=1}^{V}e^{z_j/T}},\qquad T>0$$
- $T$ = temperature, a positive number we choose.
- $z_i/T$ = each logit divided by $T$ before the softmax.
- The rest is the ordinary softmax.

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)
```
| $T$ | effect (book) | forward / toward / closer / pizza (computed from the logits) |
|---|---|---|
| 1 | unchanged | 0.572 / 0.358 / 0.061 / ≈0.0001 |
| 0.1 | sharper, "forward" almost 100% | 0.991 / 0.009 / ≈0 / ≈0 |
| 5 | flatter, "pizza" ≈ 4% | 0.242 / 0.220 / 0.155 / 0.043 |

Exercise 5.1 (answer in appendix C, book p. 304): with 1,000 samples "pizza"
appears 0 times at $T=0.1$ and 32 times at $T=5$ (3.2%); the exact probability is
4.3%. Counting samples is a [[monte-carlo-estimation|Monte Carlo]] estimate of
the probability. See [[temperature-scaling]].

### 4. Top-k sampling (book pp. 155–157)
Keep only the $k$ largest logits, set the rest to $-\infty$, then softmax:
$$\tilde z_i=\begin{cases}z_i & \text{if } z_i\ge\tau_k\\ -\infty & \text{otherwise}\end{cases}\qquad p_i=\frac{e^{\tilde z_i}}{\sum_j e^{\tilde z_j}}$$
- $\tau_k$ = the $k$-th largest logit (the cut-off).
- $\tilde z_i$ = the masked logit (tilde = "modified version").
- $e^{-\infty}=0$, so masked tokens get probability exactly 0 and the others still sum to 1.

With $k=3$: `torch.topk` gives values `[6.75, 6.28, 4.51]` at positions
`[3, 7, 0]`; the masked softmax is
`[0.0615, 0, 0, 0.5775, 0, 0, 0, 0.3610, 0]` (book pp. 156–157). It is the same
$-\infty$ trick as the [[causal-attention-mask]] (book p. 156). See [[top-k-sampling]].

### Putting it together: the `generate` function (Listing 5.4, book pp. 157–158)
```python
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                       # last position only
        if top_k is not None:                           # 1) top-k filter
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0:                           # 2) temperature + sample
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:                                           # 3) otherwise greedy
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:                          # stop at end-of-sequence
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```
Order: filter to top-$k$ → divide by $T$ → softmax → sample. `eos_id` lets
generation stop early at the [[end-of-text-token]].

### What it looks like on real models
| Model | Settings | Output | Book |
|---|---|---|---|
| Small GPT trained on "The Verdict" | greedy, 25 tokens | *Every effort moves you know," was one of the axioms he laid down across the Sevres and silver of an exquisitely appointed lun* (memorized) | p. 152 |
| Same model | `top_k=25`, `temperature=1.4`, seed 123 | *Every effort moves you stand to work on surprise, a one of us had gone with random-* | p. 158 |
| OpenAI GPT-2 124M | `top_k=50`, `temperature=1.5`, seed 123 | *Every effort moves you toward finding an ideal new way to practice something! What makes us want to be on top of that?* | p. 167 |

### Choosing settings (Exercises 5.2–5.3, appendix C pp. 304–305)
- **Small top-k (< 10) and $T<1$:** predictable and coherent. Good for reports,
  technical analysis, code, question answering, teaching material.
- **Larger top-k (about 20–40) and $T>1$:** more diverse. Good for brainstorming and fiction.
- **Deterministic output:** `top_k=None` with no temperature (greedy), or `top_k=1`.

## 🔴 Theory (Expert)
- **Limits of temperature.** As $T\to0^+$, $\operatorname{softmax}(\mathbf z/T)$
  approaches the one-hot vector on $\arg\max\mathbf z$ (greedy). As $T\to\infty$
  it approaches the uniform distribution $1/V$. The entropy of $p(T)$ grows with
  $T$. The form $p_i\propto e^{z_i/T}$ is the Boltzmann/Gibbs distribution with
  energy $E_i=-z_i$.
- **Temperature keeps the ranking.** For any $T>0$, $z_i>z_j\Rightarrow p_i(T)>p_j(T)$,
  so temperature changes *how far apart* the probabilities are, never their order.
  Top-k then truncates by rank, so the two commute in which tokens survive.
- **Top-k is conditioning.** Masking to a set $S$ and renormalizing gives
  $p(i\mid i\in S)=p_i/\sum_{j\in S}p_j$, the original distribution conditioned on
  the top-$k$ set. Its weakness is that $k$ is fixed: when the model is confident,
  $k$ may let in junk; when it is unsure, $k$ may cut good options. That motivates
  [[top-p-sampling]] (nucleus), which picks the smallest set whose probability
  mass exceeds $p$.
- **Greedy is not "most likely sentence".** Greedy maximizes each step's
  probability, not the whole sequence's $\prod_t p(x_{t+1}\mid x_{\le t})$. A
  locally lower-probability token can lead to a higher-probability sequence.
  [[beam-search]] keeps several partial sequences scored by summed
  [[log-probability|log probabilities]] to approximate the sequence-level argmax.
- **Decoding does not change the model.** Training ([[cross-entropy-loss]])
  fixes $p_\theta$. Decoding is a separate inference-time choice. Turning on
  sampling cut verbatim recitation in the book's tiny model (book pp. 151, 158),
  but the memorization itself (an [[overfitting]] effect) is still in the
  weights.
- **Batch caveat.** In Listing 5.4, `if idx_next == eos_id` assumes batch size 1;
  with a batch you would track a per-row "finished" flag.

## Connections
- Builds on: [[logits]], [[softmax]], [[argmax]], [[probability-distribution]], [[random-variable]], [[autoregressive-text-generation]]
- Leads to: [[greedy-decoding]], [[multinomial-sampling]], [[temperature-scaling]], [[top-k-sampling]], [[top-p-sampling]], [[beam-search]]
- Used by: [[text-completion]], [[gpt-2]], [[llm-applications]], [[instruction-fine-tuning]] (response generation)
- Related: [[causal-attention-mask]], [[monte-carlo-estimation]], [[end-of-text-token]], [[overfitting]], [[pretrained-weights]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 122–127 (greedy decoding, Listing 4.8 `generate_text_simple`, softmax monotonic, p. 125)
- Same, §5.2, book p. 151 (memorization motivates decoding strategies)
- Same, §5.3 intro and §5.3.1, book pp. 151–155 (toy logits, `torch.multinomial`, 582/343/73/2 counts, temperature, Fig. 5.14, Exercise 5.1)
- Same, §5.3.2, book pp. 155–157 (top-k, $-\infty$ mask, Fig. 5.15)
- Same, §5.3.3, book pp. 157–159 (Listing 5.4 `generate`, Exercises 5.2–5.3)
- Same, §5.5, book p. 167 (GPT-2 124M output with top_k=50, temperature=1.5)
- Same, appendix C, book pp. 304–305 (solutions to Exercises 5.1–5.3)
