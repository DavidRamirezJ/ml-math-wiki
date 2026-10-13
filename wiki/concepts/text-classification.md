---
title: Text Classification (Spam Detection)
type: concept
tags: [nlp, classification, supervised-learning, fine-tuning, llm]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch06-fine-tuning-for-classification, llm-appE-parameter-efficient-fine-tuning-with-lora, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Text Classification (Spam Detection)

> **One-liner:** Text classification means reading a piece of text and sticking exactly one label from a fixed list on it, like "spam" or "not spam".

## 🟢 Intuition (Dummy)
Picture a mail sorter with only two bins: **junk** and **keep**. Every letter that comes in
gets dropped into one of those two bins. The sorter never writes you a reply and never invents
a third bin. It just picks.

That's text classification. You give it a text message, and it answers with one label from a
short list it learned during training. Raschka's running example is the classic one: a phone
text like *"You are a winner you have been specially selected to receive $1000 cash or a $2000
award."* should land in the **spam** bin, while *"Hey, just wanted to check if we're still on
for dinner tonight? Let me know!"* goes into **not spam** (book p. 171).

Why does this matter? Spam filters, sentiment analysis ("is this review happy or angry?"), topic
tagging of news articles (sports / politics / technology), even spotting toxic posts on social
media: they're all the same "pick a bin" job (book pp. 9, 171).

## 🟡 Mechanics (Practitioner)

### The old way vs. the deep-learning way
- **Traditional ML** (book p. 4): an expert hand-crafts [[feature-extraction|features]] from the
  email: how often trigger words like "prize", "win", "free" show up, the number of exclamation
  marks, ALL-CAPS words, suspicious links. Then a simple model learns from those numbers.
- **Deep learning / LLMs**: the network learns its own features from the raw text. But either way,
  you **still need labels**: someone (an expert or users clicking "report spam") has to mark
  examples as spam or not spam (book p. 4). That's what makes it [[supervised-learning]].

### The learning goal, written as math
The book describes training as "minimizing the error in its predictions on a training dataset"
(book p. 3). In symbols (this is the standard [[empirical-risk-minimization]] form):

$$\boldsymbol\theta^{*}=\arg\min_{\boldsymbol\theta}\ \frac{1}{N}\sum_{n=1}^{N}\ell\big(f_{\boldsymbol\theta}(\mathbf x_n),\,y_n\big)$$

Reading it symbol by symbol:
- $\boldsymbol\theta$ = all the model's knobs (weights). $\boldsymbol\theta^{*}$ = the best setting we find.
- $\arg\min_{\boldsymbol\theta}$ = "the value of $\boldsymbol\theta$ that makes what follows as small as possible".
- $N$ = how many training messages we have; $n$ counts through them, $1,2,\dots,N$.
- $\mathbf x_n$ = the $n$-th message (as token IDs); $y_n$ = its true label (0 = not spam, 1 = spam).
- $f_{\boldsymbol\theta}(\mathbf x_n)$ = the model's guess for that message.
- $\ell(\cdot,\cdot)$ = the [[loss-function]]: a number saying how wrong the guess was (in Ch. 6 it's [[cross-entropy-loss]]).
- $\frac{1}{N}\sum$ = "average over all messages".

### How the LLM does it (Ch. 6 pipeline)
1. **Data** — the UCI *SMS Spam Collection*: 5,572 messages, 4,825 "ham" (not spam) and 747
   spam (book pp. 172–174). It's lopsided, so the book [[class-imbalance|undersamples]] ham to
   747, giving 1,494 messages.
2. **Labels to numbers** — [[label-encoding]]: ham → 0, spam → 1 (book p. 174).
3. **Split** — 70 % train / 10 % validation / 20 % test ([[train-validation-test-split]], book p. 175).
4. **Model** — start from pretrained GPT-2 124M weights ([[transfer-learning]], book pp. 181–182)
   and swap the 50,257-way vocabulary output layer for a 2-output
   [[classification-head]] (book pp. 183–186).
5. **Predict** — run the message through the model, take the [[logits]] of the **last token**
   ([[last-token-pooling]]), and pick the bigger one:

$$\hat y=\arg\max_{c\in\{0,1\}} z_c$$

- $\hat y$ ("y-hat") = the predicted class.
- $c$ = a candidate class, either 0 or 1.
- $z_c$ = the logit (raw score) the model gives to class $c$.
- $\arg\max_c$ = "which $c$ gives the biggest $z_c$?" (see [[argmax]]).

```python
with torch.no_grad():
    logits = model(input_tensor)[:, -1, :]      # last token, shape [1, 2]
predicted_label = torch.argmax(logits, dim=-1).item()
return "spam" if predicted_label == 1 else "not spam"
```
(from `classify_review`, Listing 6.12, book p. 201)

### The book's numbers
| Stage | Train acc. | Val acc. | Test acc. |
|---|---|---|---|
| Before fine-tuning (10 batches) | 46.25 % | 45.00 % | 48.75 % |
| After 5 epochs (full sets) | 97.21 % | 97.32 % | 95.67 % |

Before fine-tuning it's basically coin-flipping (random = 50 % on a balanced 2-class set)
(book p. 193); after fine-tuning it's excellent (book p. 200). Both example messages above are
classified correctly (book p. 202). See [[classification-accuracy]].

### Two ways to get an LLM to classify
| | Classification fine-tuning | Instruction fine-tuning |
|---|---|---|
| Input | just the text, no instruction | instruction + text ("Is the following text 'spam'? Answer with 'yes' or 'no'.") |
| Output | only the trained labels | free text, many tasks |
| Data / compute | less | more |
| Limitation | stuck with its classes | — |

(sidebar "Choosing the right approach", book p. 171). See [[classification-fine-tuning]] and
[[instruction-fine-tuning]]. Appendix E repeats the exact same spam pipeline but fine-tunes with
[[low-rank-adaptation|LoRA]] instead (book pp. 324–325).

## 🔴 Theory (Expert)
- **Closed label set.** A classifier models a [[probability-distribution]] over a finite set
  $\mathcal Y=\{0,\dots,K-1\}$: $p_{\boldsymbol\theta}(y\mid\mathbf x)=\mathrm{softmax}(\mathbf z)_y$.
  It cannot say anything outside $\mathcal Y$ (book p. 171). A generative LLM instead models
  $p(x_{t+1}\mid x_{\le t})$ over 50,257 tokens, which is why one prompt can ask it for any task.
  Classification fine-tuning trades that generality for a much easier, lower-data problem.
- **Training = MLE.** Minimizing average cross-entropy $-\log p_{\boldsymbol\theta}(y_n\mid\mathbf x_n)$
  is [[maximum-likelihood-estimation]] of a categorical model. The quantity we actually care
  about, accuracy (0–1 loss), is not differentiable, so cross-entropy acts as a
  [[surrogate-loss]].
- **Encoder vs. decoder classifiers.** BERT-style encoders, trained with
  [[masked-language-modeling]], see context in both directions and are the classic choice for
  classification (sentiment, document categorization; X uses BERT for toxic-content detection,
  book p. 9). A decoder-only GPT sees only leftward context under the
  [[causal-attention-mask]], so it must read the label off the last token. Appendix B points to
  work (Label-Supervised LLaMA; LLM2Vec) showing that removing the causal mask during
  classification fine-tuning can help (book p. 296).
- **Prior shift.** Balancing the training set changes the class prior from
  $\hat p(\text{spam})=747/5572\approx13.4\%$ to $1/2$. If deployment data keeps the real 13 %
  rate, the posterior is shifted toward spam. The book doesn't correct for this; see
  [[class-imbalance]].

## Connections
- Builds on: [[supervised-learning]], [[loss-function]], [[empirical-risk-minimization]], [[softmax]], [[argmax]]
- Leads to: [[classification-fine-tuning]], [[classification-head]], [[classification-accuracy]], [[class-imbalance]], [[label-encoding]]
- Used by: [[bert]], [[gpt-2]], [[low-rank-adaptation]], [[llm-applications]]
- Related: [[natural-language-processing]], [[instruction-fine-tuning]], [[transfer-learning]], [[logistic-regression]]
- People: [[sebastian-raschka]], [[jacob-devlin]]

> **From [[llm-appA-introduction-to-pytorch]]:** §A.1.2 uses an email spam classifier to explain [[supervised-learning]]: the training set holds emails with "spam" / "not spam" labels, and the trained model predicts the label of new emails (Figure A.3, book pp. 253–254). The book then says that if we train LLMs to classify texts, the workflow is the same as Figure A.3. For text generation the figure still applies, but the labels come from the text itself ([[next-token-prediction]], book p. 254).

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1, book pp. 1, 3–4 (spam filter example, handcrafted features, labels still required)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.4, book pp. 8–9 (BERT for text classification)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.1, book pp. 169–171 (classification vs instruction fine-tuning, Figs. 6.2–6.3)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §6.2, book pp. 172–175; §6.6, book p. 193; §6.7, book p. 200; §6.8, book pp. 200–202 (pipeline and accuracies)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.2, book pp. 324–325 (same dataset reused for LoRA)
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B (Ch. 6 references), book pp. 295–296
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.2 (spam classifier, Figure A.3, LLM text classification follows the same workflow), book pp. 253–254 (PDF pp. 275–276).
