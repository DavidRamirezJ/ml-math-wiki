---
title: Self-Supervised Learning
type: concept
tags: [machine-learning, llm, pretraining, learning-paradigms]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch02-working-with-text-data, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Self-Supervised Learning

> **One-liner:** Self-supervised learning is training where the data labels itself — for an LLM, the "right answer" for every position is just the next word that's already sitting in the text.

## 🟢 Intuition (Dummy)
Ordinary [[supervised-learning]] needs a teacher: someone has to sit down and tag
thousands of emails "spam" or "not spam." That's slow and costly.

Now take any sentence, say *"The model is simply trained to predict the next word."*
Hide the last word. What's the answer? *"word"* — and you didn't need anyone to tell
you; it was right there. Slide along the sentence and every word becomes a little quiz
whose answer is the next word. That's the trick. The book calls it a form of
**self-labeling**: we use the structure of the data itself as the label (book p. 12).

Why does this matter so much? Because the internet is full of text and nearly none of
it is labeled. If labels are made "on the fly" from the text, you can train on
**massive unlabeled datasets** (book p. 12) — which is exactly how LLMs get
[[pretraining|pretrained]].

## 🟡 Mechanics (Practitioner)

### The book's definition
In pretraining, LLMs use self-supervised learning, "where the model generates its own
labels from the input data" (NOTE box, book p. 6). Traditional ML and deep networks in
the conventional supervised paradigm need human labels; LLM pretraining does not.

### Turning raw text into (input, label) pairs
For a token sequence $t_1,t_2,\dots,t_T$, every prefix is an input and the following
token is its label:
$$\text{input}=(t_1,\dots,t_i),\qquad \text{label}=t_{i+1},\qquad i=1,\dots,T-1.$$
Symbol by symbol:
- $t_i$ = the [[token-id]] at position $i$;
- $T$ = the length of the text in tokens;
- $(t_1,\dots,t_i)$ = everything up to position $i$ (the "context");
- $t_{i+1}$ = the very next token — the free label.

So one text of $T$ tokens gives $T-1$ training examples, and nobody had to label any
of them. (Wiki notation; the book shows it with code.)

**The book's example (§2.6, book p. 36)** with `context_size = 4` on *The Verdict*:
```python
x = enc_sample[:context_size]          # [290, 4920, 2241, 287]
y = enc_sample[1:context_size+1]       # [4920, 2241, 287, 257]
```
The target `y` is just the input shifted by one. Spelled out:
```
[290]                    ----> 4920      " and"                        -> " established"
[290, 4920]              ----> 2241      " and established"            -> " himself"
[290, 4920, 2241]        ----> 287       " and established himself"    -> " in"
[290, 4920, 2241, 287]   ----> 257       " and established himself in" -> " a"
```
One window of length 4 packs 4 prediction tasks. See [[sliding-window-sampling]].

### What the model learns from it
The model learns the conditional probability
$$P(t_{i+1}\mid t_1,\dots,t_i)$$
- read "the probability of the next token, **given** all tokens so far."

It is trained by maximising that probability for the true next token, i.e. minimising
the [[cross-entropy-loss]] (Ch. 5). In code (§5.1, book p. 133) the "labels" are
literally the inputs shifted:
```python
inputs  = torch.tensor([[16833, 3626, 6100],   # "every effort moves"
                        [40,    1107,  588]])  # "I really like"
targets = torch.tensor([[3626, 6100,   345],   # " effort moves you"
                        [1107,  588, 11311]])  # " really like chocolate"
```

### How it compares to the other paradigms
| Paradigm | Where labels come from | Book example |
|---|---|---|
| [[supervised-learning]] | humans | spam / not-spam emails (§A.1, book p. 253); classification fine-tuning (Ch. 6) |
| **self-supervised** | the data itself | next-word prediction in pretraining (book pp. 6, 12) |
| unsupervised | no labels at all | (clustering, [[dimensionality-reduction]] — not covered by the book) |

Appendix A notes that LLM pretraining still fits the supervised workflow picture —
it's just that "the labels during pretraining can be derived from the text itself"
(book p. 254; the book
does not use the word "self-supervised" there).

## 🔴 Theory (Expert)
- **Supervised loss, unsupervised data.** Self-supervision is a *construction*: a
  fixed pretext map $g:\text{text}\to\{(\mathbf x_i,y_i)\}$ turns unlabeled data into a
  supervised problem, then ordinary [[empirical-risk-minimization]] applies:
  $\min_\theta\frac1N\sum_i -\log p_\theta(y_i\mid\mathbf x_i)$. For next-token
  prediction, summing over all prefixes equals the negative log-likelihood of the
  whole sequence by the chain rule, so self-supervised pretraining is exactly
  [[maximum-likelihood-estimation]] of a density model $p_\theta(t_{1:T})$.
- **Causal vs. masked pretext tasks.** GPT predicts the next token using only the left
  context (enforced by the [[causal-attention-mask]]); BERT instead predicts
  **randomly masked words** using both sides (book pp. 8–9) — see
  [[masked-language-modeling]], [[bert]]. Both are self-supervised; the causal one
  gives a generative model, the masked one gives strong encoders for classification.
- **No leakage.** The model must not see the label: in Figure 2.12, "the LLM can't
  access words past the target" (book pp. 35–36). Every position's target is the
  next token, so attention must be masked to positions $\le i$ — otherwise the task
  is trivial copying.
- **Why it scales.** Label cost is zero, so the dataset size is limited only by
  available text and compute — GPT-3 used 300 B tokens (book p. 11). Rich structure in
  the pretext target forces the network to learn syntax, semantics and some world
  knowledge, the basis for [[emergent-behavior]] (book p. 14).
- **Subtlety.** "Self-supervised" is sometimes loosely grouped under "unsupervised
  pretraining" (e.g. the GPT-2 paper title "Language Models are Unsupervised Multitask
  Learners") — the data is unlabeled, the objective is supervised. Outside the book.

## Connections
- Builds on: [[supervised-learning]], [[machine-learning]], [[probability-distribution]], [[tokenization]]
- Leads to: [[pretraining]], [[next-token-prediction]], [[sliding-window-sampling]], [[masked-language-modeling]], [[foundation-model]]
- Uses: [[cross-entropy-loss]], [[maximum-likelihood-estimation]], [[empirical-risk-minimization]], [[causal-attention-mask]]
- Related: [[large-language-model]], [[gpt-architecture]], [[bert]], [[autoregressive-text-generation]], [[emergent-behavior]]
- People: [[alec-radford]], [[jacob-devlin]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.3, book p. 6 (NOTE: self-supervised learning, raw text).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 8–9 (BERT masked word prediction); §1.5, book p. 11; §1.6, book pp. 12, 14 (self-labeling, labels on the fly, Figure 1.7).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.6, book pp. 35–36 (input–target shift, Figure 2.12).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1, book p. 133 (inputs/targets tensors).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1, book pp. 253–254 (supervised workflow, labels from text).
