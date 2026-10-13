---
title: Argmax
type: concept
tags: [functions, optimization, decoding, classification, llm]
difficulty: beginner
sources: [llm-ch04-implementing-a-gpt-model, llm-ch06-fine-tuning-for-classification, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Argmax

> **One-liner:** Argmax doesn't tell you *how big* the biggest number is. It tells you *where* the biggest number sits.

## 🟢 Intuition (Dummy)
Picture a row of buckets with scores written on them. "Max" answers "what's the highest score?" **Argmax** answers a different question: "*which bucket* has the highest score?" You get back a position (a label, an index), not the score itself.

That's exactly what a language model needs. At the end of each step, the model has a score for every word piece in its [[vocabulary]] (50,257 of them for [[gpt-2]]). We don't care that the winning score is 2.48. We care that the winner is bucket number 257, because bucket 257 *is* the token "a". The same trick turns a spam classifier's two scores into a decision: bucket 0 = "not spam", bucket 1 = "spam".

A nice bonus: if you stretch or squash all the scores in a way that keeps their order (like turning them into probabilities with [[softmax]]), the winner doesn't change. That's why the book says the softmax step is optional when all you want is the argmax.

## 🟡 Mechanics (Practitioner)

### The definition
$$\hat{k} = \arg\max_{k \in \{0,\dots,K-1\}} z_k$$

Symbol by symbol:
- $\mathbf{z} = (z_0, z_1, \dots, z_{K-1})$ = a [[vector]] of $K$ scores (in an LLM, the [[logits]]).
- $k$ = a position (index) in that vector; $\{0,\dots,K-1\}$ = all positions, counted from 0 like PyTorch does.
- $\max_k z_k$ would be "the largest value". Adding **arg** in front means "the *argument* (the input $k$) that produces that largest value".
- $\hat{k}$ = the winning index. The hat $\hat{\ }$ is the usual mark for "a prediction".

### Use 1: picking the next token (greedy decoding, §4.7)
The book's GPT model returns a $4 \times 50{,}257$ matrix of logits for the input "Hello, I am" (one row per input token). Only the **last row** matters for the next token (book p. 124):

1. Last-row logits: `[-0.4929, ..., 2.4812, ..., -0.6093]`
2. After [[softmax]]: `[0.0001, ..., 0.0200, ..., 0.0001]`
3. Argmax → index **257** → decoded as **"a"**, which is appended to the input.

As a formula:
$$\hat{x}_{t+1} = \arg\max_{i \in \{0,\dots,V-1\}} p_i = \arg\max_{i} z_i$$
- $\hat{x}_{t+1}$ = the [[token-id]] chosen for the next position $t+1$.
- $V$ = vocabulary size (50,257), so valid indices run from 0 to 50,256.
- $p_i = \mathrm{softmax}(\mathbf{z})_i$ = the probability of token $i$; $z_i$ = its raw logit.
- The second "=" says: you get the same winner whether you look at probabilities or logits.

Always taking the argmax like this is called **[[greedy-decoding]]** (book p. 125). The book's Listing 4.8:

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]                              # (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

### Use 2: turning class scores into a label (§6.6, §6.8, §A.7)
For spam classification the output head has only **2** units, so the last token gives 2 logits instead of 50,257 (book p. 190). The rule is the same:
$$\hat{y} = \arg\max_{k} \operatorname{softmax}(\mathbf{z})_k = \arg\max_{k} z_k$$
- $\hat{y}$ = predicted class: 0 = "not spam", 1 = "spam" (book p. 191).

The book's example (book p. 192): the last-token logits are `tensor([[-3.5983, 3.9902]])`. The bigger number is in position 1, so

```python
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())   # 1  ("spam")
```

and because softmax keeps the order, the shortcut works too: `torch.argmax(logits)` gives the same 1. The final `classify_review` function (book p. 201) skips softmax entirely: `predicted_label = torch.argmax(logits, dim=-1).item()`.

In appendix A (book pp. 276–277), a small 2-class network outputs 5 rows of logits, e.g. row 1 = `[2.8569, -4.1618]` → probabilities `[0.9991, 0.0009]`. Both

```python
torch.argmax(probas, dim=1)    # tensor([0, 0, 0, 1, 1])
torch.argmax(outputs, dim=1)   # tensor([0, 0, 0, 1, 1])
```

give the same labels.

### The `dim` argument
- `dim=1` (or `dim=-1` for the last axis) = "find the winner **in each row**". For a batch of shape `[batch, classes]` you get one label per example, shape `[batch]`.
- `dim=0` = "find the winner **in each column**" (book p. 277).
- `keepdim=True` keeps the reduced axis with size 1, so the result is `(batch, 1)`. That shape is what `torch.cat` needs to append the new token in Listing 4.8.

### Where argmax feeds next
Counting how often the argmax label equals the true label gives **[[classification-accuracy]]** (Listing 6.8):
$$\text{accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[\hat{y}^{(i)} = y^{(i)}\right]$$
- $N$ = number of examples; $\hat{y}^{(i)}$ = the argmax prediction for example $i$; $y^{(i)}$ = its true label; $\mathbb{1}[\cdot]$ = 1 if the statement inside is true, else 0.

## 🔴 Theory (Expert)
- **Formal definition.** For $f: S \to \mathbb{R}$, $\arg\max_{s\in S} f(s) = \{ s \in S : f(s) \ge f(s')\ \forall s' \in S\}$. It is a **set**, which may hold several points (ties) or none (e.g. $f(x)=x$ on $(0,1)$). On a finite set like a vocabulary it is never empty. `torch.argmax` breaks ties by returning the **first** maximal index (PyTorch docs; not in the book).
- **Invariance under strictly increasing maps.** If $g$ is strictly increasing, then $\arg\max_s g(f(s)) = \arg\max_s f(s)$. This is why softmax ($e^{z_k}$ over a shared positive denominator), $\log$, and division by a temperature $T>0$ never change the argmax. See [[monotonic-function]], [[log-probability]], [[temperature-scaling]]. The book's statement "the softmax step is redundant" (book p. 125) is this fact.
- **Argmax is not differentiable.** As a function of $\mathbf{z}$, the index $\arg\max_k z_k$ is piecewise constant. Its gradient is 0 almost everywhere and undefined on the tie boundaries. So accuracy (built from argmax) can't be trained with [[gradient-descent]]. The book makes this point in §6.6 (book p. 193) and trains [[cross-entropy-loss]] instead, as a [[surrogate-loss]].
- **Softmax is the smooth "soft arg max".** $\mathrm{softmax}(\mathbf{z}/T) \to \mathbf{e}_{\hat{k}}$ (the one-hot vector of the argmax, when it is unique) as $T \to 0^+$. So greedy decoding is the zero-temperature limit of [[multinomial-sampling]]. See [[temperature-scaling]], [[one-hot-encoding]].
- **Decision-theory view.** Picking $\arg\max_k P(y=k\mid\mathbf{x})$ is the **MAP / Bayes decision rule**. It minimizes the expected 0–1 loss under the model's own distribution. Argmax of a discrete [[probability-distribution]] is its **mode**.
- **Greedy ≠ best sequence.** Per-step argmax maximizes each factor $P(x_{t+1}\mid x_{1:t})$ on its own. It does *not* in general find $\arg\max_{x_{1:T}} \prod_t P(x_{t+1}\mid x_{1:t})$. That search is exponentially large, which is why [[beam-search]] exists, and why the book moves on to [[top-k-sampling]] and temperature in Ch. 5 to make text less repetitive. See [[decoding-strategies]].
- **Argmax elsewhere in ML math.** [[maximum-likelihood-estimation]] is $\hat{\boldsymbol\theta} = \arg\max_{\boldsymbol\theta} p(\mathcal{D}\mid\boldsymbol\theta)$. The BPE merge rule picks $\arg\max_{(a,b)} \mathrm{count}(a,b)$ ([[byte-pair-encoding]]).

## Connections
- Builds on: [[vector]], [[logits]], [[softmax]], [[monotonic-function]], [[probability-distribution]]
- Leads to: [[greedy-decoding]], [[classification-accuracy]], [[decoding-strategies]], [[beam-search]], [[top-k-sampling]], [[temperature-scaling]]
- Used by: [[autoregressive-text-generation]], [[classification-fine-tuning]], [[text-classification]], [[maximum-likelihood-estimation]], [[byte-pair-encoding]]
- Contrast: [[surrogate-loss]], [[cross-entropy-loss]] (a differentiable stand-in for argmax-based accuracy)
- People: [[sebastian-raschka]]

> **From [[llm-ch05-pretraining-on-unlabeled-data]]:** Ch. 5 uses argmax in two places. In §5.1.1–5.1.2 it applies `torch.argmax(probas, dim=-1, keepdim=True)` to the softmax output of the untrained model to turn each position's 50,257 probabilities into one predicted token ID (book pp. 134–135). In §5.3 the book names this rule **greedy decoding** and replaces it with sampling: `torch.argmax(probas).item()` always picks "forward" from the toy 9-token vocabulary, while `torch.multinomial` picks it only about 58% of the time (book pp. 152–154). Read $\arg\max_i p_i$ as "the index $i$ whose probability $p_i$ is largest". See [[greedy-decoding]] and [[multinomial-sampling]].

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.7, book pp. 123–125 (Fig. 4.17: last-row logits → softmax → argmax → ID 257 "a"; Listing 4.8; softmax is monotonic, so softmax is redundant before argmax; greedy decoding).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.6, book pp. 190–193 (Fig. 6.14; 2-class logits `[-3.5983, 3.9902]` → label 1; Listing 6.8 accuracy; accuracy is not differentiable).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.8, book p. 201 (`classify_review` uses `torch.argmax(logits, dim=-1)`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7, book pp. 276–277 (argmax on probabilities and on logits both give `tensor([0, 0, 0, 1, 1])`; `dim=1` vs `dim=0`).
- The set definition, tie-breaking, the zero-temperature limit, and the MAP decision view are standard math, not printed in the book.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1.1–5.1.2, book pp. 134–135 (`torch.argmax` on probabilities); §5.3–5.3.1, book pp. 152–155 (greedy decoding replaced by `torch.multinomial`); §5.3.3, book p. 158 (argmax branch when no temperature is set).
