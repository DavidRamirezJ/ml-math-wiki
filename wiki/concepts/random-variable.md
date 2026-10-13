---
title: Random Variable
type: concept
tags: [probability, foundations, llm]
difficulty: beginner
sources: [mml-ch06-probability-and-distributions, llm-ch05-pretraining-on-unlabeled-data]
created: 2026-06-04
updated: 2026-10-13
---

# Random Variable

> **One-liner:** A random variable is a rule that turns the messy outcome of a random experiment into a number you actually care about.

## 🟢 Intuition (Dummy)
Flip two coins. The raw outcome is something like "heads then tails" — awkward to do math with. But maybe all you care about is *how many heads* came up: 0, 1, or 2. A random variable is the little machine that reads the raw outcome and spits out that number. It is "random" only because its *input* (the experiment) is random; the machine itself is a fixed, deterministic lookup table.

The name is famously misleading: a random variable is **neither random nor a variable — it is a function.**

### In LLMs
When a language model writes, the **next word is a random variable**. The model doesn't
output a word; it outputs a probability for every word it knows (all 50,257 GPT-2 tokens).
The actual word only appears when you "run the experiment" — draw from those
probabilities. In the book's Chapter 5 example, after "every effort moves you" the next
word is "forward" with probability about 0.57, "toward" about 0.36, "closer" about 0.06,
and a long tail of unlikely words. Each time you generate, you get one *value* of that
random variable — usually "forward", sometimes "toward" (book pp. 152–154).

## 🟡 Mechanics (Practitioner)
A **probability space** has three pieces (Kolmogorov's axioms):
- **Sample space** $\Omega$ — set of all possible outcomes (e.g. $\{hh, ht, th, tt\}$).
- **Event space** $\mathcal A$ — collection of subsets of $\Omega$ we can assign probability to (often the power set for discrete problems).
- **Probability measure** $P$ — assigns each event $A\in\mathcal A$ a number $P(A)\in[0,1]$ with $P(\Omega)=1$.

A **random variable** is a function
$$ X : \Omega \to T $$
into a **target space** $T$ (the values/states we care about). The probability transfers to $T$ via the **pre-image**:
$$ P_X(S) = P(X\in S) = P\big(X^{-1}(S)\big) = P(\{\omega\in\Omega : X(\omega)\in S\}),\quad S\subseteq T. $$
This induced law $P_X = P\circ X^{-1}$ is the [[probability-distribution]] of $X$.

**Worked example (two biased coins, $P(\$)=0.3$).** Let $X$ = number of \$ drawn, $T=\{0,1,2\}$. Then $X(\$\$)=2,\ X(\$\pounds)=X(\pounds\$)=1,\ X(\pounds\pounds)=0$, and
$$ P(X{=}2)=0.3^2=0.09,\quad P(X{=}1)=2(0.3)(0.7)=0.42,\quad P(X{=}0)=0.7^2=0.49. $$

### In LLMs: the next token $X_{t+1}$ (Build an LLM §5.3)
The book's text generator treats the next token ID as a discrete random variable with
target space $T=\{0,1,\dots,V-1\}$ (the [[token-id|token IDs]]):
$$P(X_{t+1}=i \mid x_1,\dots,x_t) = p_i = \mathrm{softmax}(\mathbf z)_i$$
Symbol by symbol:
- $X_{t+1}$ = the (random) token at the next position $t+1$; capital letter because it is
  the random variable, not one value.
- $\mid x_1,\dots,x_t$ = "given the tokens already written" — the distribution is
  *conditional* on the context.
- $p_i$ = the model's probability for token $i$; $\mathbf z$ = the [[logits]] vector.
- $V$ = vocabulary size: 9 in the book's toy example, 50,257 for GPT-2.

Toy example (book pp. 152–154), logits `[4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]`
for `closer, every, effort, forward, inches, moves, pizza, toward, you`:
- [[greedy-decoding]] ignores the randomness and returns the mode, $\arg\max_i p_i$ = 3 ("forward").
- [[multinomial-sampling]] (`torch.multinomial(probas, num_samples=1)`) produces an actual
  *realization* of $X_{t+1}$. 1,000 realizations gave forward 582, toward 343, closer 73,
  inches 2 — empirical frequencies close to $p=(0.572, 0.358, 0.061, 0.003)$, see
  [[monte-carlo-estimation]].
- [[temperature-scaling]] and [[top-k-sampling]] don't change the experiment — they change
  the *distribution* $P_X$ the variable is drawn from.

**ML usage:** every quantity we model — a class label, a pixel value, a model parameter, a prediction — is a random variable. $T$ finite/countable ⇒ **discrete**; $T=\mathbb R$ or $\mathbb R^D$ ⇒ **continuous**.

## 🔴 Theory (Expert)
- For continuous $T$, the naive power-set event space is too large; $\mathcal A$ must be a **Borel $\sigma$-algebra** (closed under complements and countable unions/intersections) and $P$ a **measure** on it. MML deliberately side-steps the measure-theoretic construction (p. 180–181) and "pretends" continuous spaces behave like finite discrete ones.
- Measurability: $X$ must be a **measurable function**, i.e. $X^{-1}(S)\in\mathcal A$ for every measurable $S\subseteq T$, so that $P_X(S)$ is well-defined.
- For continuous $X$, $P(X=x)=0$ for every single $x$ (a set of measure zero); probability lives on *intervals/regions*, described by a [[probability-distribution|density]].
- The book flags the rampant abuse of "probability distribution" to mean the pmf, the pdf, **and** the cdf — disambiguate by context.

### In LLMs: what is the sample space when we "sample a token"?
- In practice the outcome $\omega$ is the state of a **pseudo-random number generator**.
  `torch.manual_seed(123)` fixes that state, which is why the book's outputs are
  reproducible: with the seed fixed, the "random" variable becomes a deterministic
  function of the seed ([[random-seed-reproducibility]]). The function view of a random
  variable is literally how the code works: $X = g(U)$, with $U\sim\mathrm{Uniform}(0,1)$
  and $g$ the inverse [[cumulative-distribution-function]] of the categorical
  distribution (inverse-CDF sampling).
- A generated text is a **sequence of dependent random variables** $X_1,X_2,\dots$; by the
  product rule its joint law factorizes as $\prod_t P(X_t\mid X_{<t})$ — the
  [[autoregressive-text-generation|autoregressive]] factorization the GPT model is trained
  to approximate by [[maximum-likelihood-estimation]] ([[cross-entropy-loss]]).
- Counting $N$ i.i.d. realizations per category gives a **multinomial** random vector —
  the reason PyTorch's sampler is called `torch.multinomial`.

## Connections
- Builds on: [[vector]]
- Leads to: [[probability-distribution]], [[expected-value-and-mean]], [[bayes-theorem]]
- Used by: [[gaussian-distribution]], [[exponential-family]], essentially every probabilistic ML model; next-token generation via [[multinomial-sampling]], [[temperature-scaling]], [[top-k-sampling]], [[monte-carlo-estimation]]
- People: —

## Sources
- `raw/mml-book.pdf` §6.1, printed pp. 172–178 (PDF pp. 178–184). Example 6.1 (biased-coin) verbatim.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.3.1, book pp. 152–155 (next token as a sampled categorical variable; toy logits; 1,000-sample bincount; `torch.manual_seed`).

## From llm-ch07-fine-tuning-to-follow-instructions
- In §7.8 the judge score $S_i \in \{0,\dots,100\}$ for test example $i$ acts like a random variable, because Ollama is not fully deterministic across operating systems (book p. 247). The book reports the sample mean $\bar s = \frac{1}{N}\sum_{i=1}^{N} s_i$. Read it symbol by symbol: $\bar s$ is "s-bar", the average score. $\frac{1}{N}\sum_{i=1}^{N}$ means "add up the $N$ scores and divide by $N$". $s_i$ is the score of example $i$. With $N = 110$, the result is 50.32. The book suggests repeating the evaluation and averaging the results to reduce the noise. See [[expected-value-and-mean]], [[llm-as-a-judge]], [[llm-ch07-fine-tuning-to-follow-instructions]].
- Source: `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, book pp. 246–247.
