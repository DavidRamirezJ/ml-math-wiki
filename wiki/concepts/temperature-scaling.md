---
title: Temperature Scaling
type: concept
tags: [llm, decoding, probability, softmax, text-generation]
difficulty: intermediate
sources: [llm-ch05-pretraining-on-unlabeled-data, llm-ch07-fine-tuning-to-follow-instructions]
created: 2026-10-13
updated: 2026-10-13
---

# Temperature Scaling

> **One-liner:** Temperature is a single knob that makes a language model's next-word choices more cautious (low temperature) or more adventurous (high temperature) by dividing its raw scores before turning them into probabilities.

## 🟢 Intuition (Dummy)
When an LLM is about to write the next word, it doesn't just "know" the answer. It
has a list of every word it knows, each with a score saying how much it likes that
word. Those scores are turned into probabilities, and then a word is picked.

Temperature is a dial on that last step:

- **Turn it down (cold, e.g. 0.1):** the favorite word becomes a near-certain winner.
  The model plays it safe and becomes boring and repetitive — almost like always
  picking the top word.
- **Leave it at 1:** you get the model's "honest" probabilities, untouched.
- **Turn it up (hot, e.g. 5):** the probabilities flatten out, so the long-shot words
  get picked more often. More variety, more creativity… and more nonsense.

The book's running example makes this vivid: after *"every effort moves you"*, a cold
model says "forward" practically every time, while a hot model sometimes says
"every effort moves you **pizza**" (about 4% of the time at temperature 5).

The name comes from physics: hot particles bounce around randomly, cold ones settle
into their lowest-energy spot. Same picture here.

## 🟡 Mechanics (Practitioner)

### The formula
Plain [[softmax]] turns the model's [[logits]] into probabilities:
$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$
Temperature scaling just divides every logit by a number $T>0$ first (book p. 154:
"dividing the logits by a number greater than 0"):
$$p_i(T) = \frac{e^{z_i/T}}{\sum_{j=1}^{V} e^{z_j/T}}, \qquad T>0$$

Reading it symbol by symbol:
- $z_i$ = the logit (raw, unnormalized score) the model gives token number $i$.
- $T$ = the temperature, a positive number you choose.
- $z_i/T$ = the "scaled logit". Dividing by a small $T$ stretches the gaps between
  scores; dividing by a big $T$ squashes them together.
- $e^{(\cdot)}$ = the exponential, which makes every score positive.
- $\sum_{j=1}^{V}$ = "add up over all $V$ tokens in the [[vocabulary]]", so the
  probabilities sum to 1.
- $p_i(T)$ = the probability of picking token $i$ at temperature $T$.

What the dial does (book pp. 154–155):
- $T=1$ → exactly the ordinary softmax (no scaling).
- $T>1$ → a flatter, more **uniform** distribution.
- $T<1$ → a "sharper or more peaky", more **confident** distribution.

Temperature never changes the *ranking* of the tokens — only how far apart their
probabilities are. That is why the [[argmax]] (the top token) is the same at every $T$.

### The book's worked example (§5.3.1, book pp. 152–155)
A toy 9-token vocabulary, with next-token logits after "every effort moves you":

| token | closer | every | effort | forward | inches | moves | pizza | toward | you |
|---|---|---|---|---|---|---|---|---|---|
| logit $z_i$ | 4.51 | 0.89 | −1.90 | 6.75 | 1.63 | −1.62 | −1.89 | 6.28 | 1.79 |
| $p_i(T{=}1)$ | 0.061 | 0.002 | 0.000 | **0.572** | 0.003 | 0.000 | 0.000 | 0.358 | 0.004 |
| $p_i(T{=}0.1)$ | 0.000 | 0.000 | 0.000 | **0.991** | 0.000 | 0.000 | 0.000 | 0.009 | 0.000 |
| $p_i(T{=}5)$ | 0.155 | 0.075 | 0.043 | **0.242** | 0.087 | 0.045 | 0.043 | 0.220 | 0.090 |

(The book plots these as Figure 5.14 without printing the numbers; the values above
are recomputed from the book's logits.) Reading the table:
- At $T=1$, "forward" gets picked about **60%** of the time with
  [[multinomial-sampling]] (book p. 154).
- At $T=0.1$, "forward" is picked almost **100%** of the time — this behaves like
  [[greedy-decoding]] (book p. 155).
- At $T=5$, the distribution is much flatter: "pizza" jumps to about **4.3%**
  (book p. 155; exact value `scaled_probas[2][6]` in appendix C, p. 304).

**Exercise 5.1 result (appendix C, p. 304).** With `print_sampled_tokens` (1,000 draws,
`torch.manual_seed(123)`), "pizza" is sampled **0×** at $T=0.1$ (and at $T=1$), but
**32×** at $T=5$ — an estimated $32/1000=3.2\%$, versus the exact probability **4.3%**.
The "faster and more accurate way" the exercise asks for is to read the probability
directly from the softmax output instead of sampling (see [[monte-carlo-estimation]]).
For comparison, at $T=1$ the 1,000 draws give 582× "forward", 343× "toward",
73× "closer", 2× "inches" (book pp. 153–154).

### The code (book p. 154)
```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]   # original, T = 0.1 (sharper), T = 5 (flatter)
scaled_probas = [softmax_with_temperature(next_token_logits, T)
                 for T in temperatures]
```

Inside the book's final `generate` function (Listing 5.4, book pp. 157–158),
temperature is applied after optional [[top-k-sampling]]:
```python
if temperature > 0.0:
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
else:                                     # temperature == 0 -> greedy
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
```
Note the convention: `temperature=0.0` (the default) is treated as a special case
meaning "no sampling, just argmax", because you can't literally divide by zero.

### Settings used in the book
- §5.3.3 (own small model): `top_k=25, temperature=1.4` → *"Every effort moves you
  stand to work on surprise, a one of us had gone with random-"* (book p. 158).
- §5.5 (OpenAI GPT-2 124M weights): `top_k=50, temperature=1.5`, `max_new_tokens=25`
  → *"Every effort moves you toward finding an ideal new way to practice something!
  What makes us want to be on top of that?"* (book p. 167).
- §7.8 (Llama 3 as a judge via Ollama): `"seed": 123, "temperature": 0` — labelled
  "settings for deterministic responses" (book pp. 242–243). Temperature 0 = greedy
  decoding, which aims to make the judge's scores reproducible. Caveat from the book:
  Ollama is "not entirely deterministic across operating systems", so scores can still
  vary slightly; repeat the evaluation and average the scores for robust results
  (book p. 247; also pp. 241, 244).

### Forcing deterministic output (Exercise 5.3, solution in appendix C, p. 305)
The `generate` function behaves like `generate_text_simple` (always the same output)
if you either (1) set `top_k=None` and apply no temperature scaling (`temperature=0.0`,
the default, so the `argmax` branch runs), or (2) set `top_k=1`, so only one token
survives the mask and sampling has a single choice.

### When to use low vs. high temperature (Exercise 5.2, solution in appendix C, pp. 304–305)
- **Low** temperature (< 1), small top-k (< 10): predictable, coherent text — formal
  documents, technical analysis, code generation, question answering, educational content.
- **High** temperature (> 1), larger top-k (about 20–40): diverse text — brainstorming,
  fiction and other creative writing.

## 🔴 Theory (Expert)
- **Boltzmann/Gibbs form.** $p_i(T)\propto e^{-E_i/T}$ with "energy" $E_i=-z_i$ is
  exactly the Boltzmann distribution of statistical mechanics; the softmax is its
  normalized form. (Book uses the term without the physics derivation.)
- **Limits.** As $T\to0^+$, $p(T)$ converges to a one-hot vector on $\arg\max_i z_i$
  (if the maximum is unique) — the zero-temperature limit is [[greedy-decoding]].
  As $T\to\infty$, $z_i/T\to0$ for all $i$, so $p_i(T)\to 1/V$, the uniform distribution.
  Proof sketch for the low limit: divide numerator and denominator by $e^{z_{\max}/T}$;
  every term $e^{(z_j-z_{\max})/T}$ with $z_j<z_{\max}$ goes to 0.
- **Entropy rises monotonically with $T$.** For the toy logits, the Shannon entropy
  $H(p)=-\sum_i p_i\ln p_i$ is about 0.05 nats at $T=0.1$, 0.91 at $T=1$ and 2.00 at
  $T=5$, approaching $\ln 9\approx2.20$ (uniform). Formally $\frac{dH}{dT} = \frac{1}{T^3}\operatorname{Var}_{p}(z)\ge0$.
- **Order preservation.** $z\mapsto z/T$ is a strictly increasing ([[monotonic-function|monotonic]])
  map for $T>0$, and softmax preserves order, so temperature reshapes the distribution
  without changing which token is most likely. That is why temperature alone never
  removes junk tokens — that is the job of truncation methods like [[top-k-sampling]]
  and [[top-p-sampling]].
- **Exponential-family view.** The softmax is a [[exponential-family|categorical
  exponential family]] with natural parameters $z$; temperature rescales the natural
  parameters, $\eta = z/T$ — equivalently, it is the inverse temperature $\beta=1/T$ of
  statistical physics.
- **Relation to training.** The model is trained (cross-entropy) at $T=1$, so $T=1$ is
  the only setting that reproduces the distribution the model actually learned
  ([[maximum-likelihood-estimation]]). Any other $T$ is a deliberate, post-hoc distortion
  for decoding. ("Temperature scaling" is also the name of a calibration method that
  *fits* $T$ on validation data; the book only uses it as a decoding knob.)

## Connections
- Builds on: [[logits]], [[softmax]], [[probability-distribution]], [[exponential-family]]
- Works with: [[multinomial-sampling]], [[top-k-sampling]], [[argmax]], [[greedy-decoding]]
- Leads to: [[decoding-strategies]], [[top-p-sampling]], [[autoregressive-text-generation]]
- Used by: [[gpt-2]] text generation (§5.5), [[llm-as-a-judge]] with deterministic settings (§7.8), [[local-llm-inference-ollama]]
- Related: [[random-seed-reproducibility]], [[monte-carlo-estimation]], [[overfitting]] (memorization motivates diversity)
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.1, book pp. 152–155 (toy logits, `softmax_with_temperature`, Figure 5.14, Exercise 5.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.3, book pp. 157–159 (Listing 5.4 `generate`, `temperature=1.4`, Exercise 5.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book p. 167 (`top_k=50, temperature=1.5` with GPT-2 weights; chapter summary).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, book pp. 242–243 (`temperature: 0` in the Ollama `query_model` options) and p. 247 (Ollama not fully deterministic across operating systems).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix C, book pp. 304–305 (solutions to Exercises 5.1–5.3).
