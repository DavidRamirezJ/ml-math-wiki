---
title: Manual Feature Extraction
type: concept
tags: [machine-learning, feature-engineering, deep-learning, foundations]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models]
created: 2026-10-13
updated: 2026-10-13
---

# Manual Feature Extraction

> **One-liner:** Manual feature extraction means a human expert decides which measurable clues (features) a model gets to look at, instead of the model figuring out the clues by itself.

## 🟢 Intuition (Dummy)
Imagine teaching a friend to spot spam emails, but they are only allowed to look at a short checklist
you wrote: "How many times does it say *free*? How many `!` marks? Any words in ALL CAPS? Any weird
links?" Your friend never sees the actual email, only your checklist answers. If you picked good
questions, they do well. If you forgot an important clue, they are stuck.

That checklist is **manual feature extraction**. Traditional [[machine-learning]] works this way: a
human designs the features. [[deep-learning]] skips the checklist. You hand the network the raw text,
and it learns its own clues along the way. That is a big reason deep learning (and so
[[large-language-model|LLMs]]) took over language tasks.

## 🟡 Mechanics (Practitioner)

**What the book says.** In traditional ML, "human experts might manually extract features from email
text such as the frequency of certain trigger words (for example, 'prize,' 'win,' 'free'), the number
of exclamation marks, use of all uppercase words, or the presence of suspicious links" (book p. 4).
The dataset built from these expert-defined features is then used to train the model. Deep learning,
in contrast, "does not require manual feature extraction", meaning "human experts do not need to identify
and select the most relevant features for a deep learning model" (book pp. 3–4).

**Features become a vector.** Each email is turned into a [[vector|feature vector]]. With the book's
four spam clues:
$$\mathbf x = \begin{bmatrix} x_1\\ x_2\\ x_3\\ x_4\end{bmatrix} =
\begin{bmatrix} \text{count of trigger words}\\ \text{count of ``!''}\\ \text{count of ALL-CAPS words}\\ \text{has suspicious link (0/1)}\end{bmatrix}\in\mathbb R^4 .$$
Symbol by symbol: $\mathbf x$ (bold) = the whole feature vector for one email; $x_1,\dots,x_4$ = its
four numbers; $\mathbb R^4$ = "a list of 4 real numbers".

**Worked example** (numbers are illustrative, the features are the book's). The email
*"WIN a FREE prize!!! Click here: bit.ly/xyz"* gives trigger words = 3 (win, free, prize), `!` = 3,
ALL-CAPS words = 2 (WIN, FREE), suspicious link = 1, so
$$\mathbf x = (3,\;3,\;2,\;1)^\top .$$
A simple classifier such as [[logistic-regression]] then computes a score
$z=\mathbf w^\top\mathbf x+b$, where $\mathbf w$ = one learned weight per feature (how much that clue
matters), $b$ = a learned offset ([[bias-vector|bias]]), and $\mathbf w^\top\mathbf x$ = the
[[inner-product|dot product]] $w_1x_1+w_2x_2+w_3x_3+w_4x_4$. Only $\mathbf w$ and $b$ are learned. The *features
themselves* were chosen by a human.

**Labels are still needed either way.** Both traditional ML and deep learning for spam classification
still need someone (an expert or users) to collect the spam / not-spam labels (book p. 4). Removing
manual features does not remove manual labels; that is what [[self-supervised-learning]] does later.

**The deep-learning alternative in this book.** An LLM receives raw text as [[token-id|token IDs]],
maps each to a learned [[token-embedding]] (e.g. 768 numbers per token in GPT-2 124M), and the
[[transformer-architecture|transformer layers]] build their own internal features. Nobody tells the
model "count the exclamation marks".

## 🔴 Theory (Expert)
- **Feature map view.** Manual feature extraction fixes a map $\phi:\text{raw input}\to\mathbb R^D$
  by hand and learns only a simple model on top, $f(\mathbf x)=g_\theta(\phi(\mathbf x))$. Deep
  learning makes $\phi$ itself parametric, $\phi_\psi$, and learns $\psi$ and $\theta$ jointly by
  [[backpropagation]] through the whole stack. This is often called *representation learning* (a
  gloss; the book does not use the term).
- **Inductive bias trade-off.** Hand-made features inject expert knowledge, which helps a lot with
  little data. Learned features need much more data and compute, but they are not limited by what the
  expert thought of. LLMs sit at the extreme: billions of tokens, almost no hand-made features (only
  the [[tokenization|tokenizer]] is designed by people).
- **Link to kernels.** Choosing $\phi$ by hand is the same move the [[kernel-trick]] makes implicitly:
  a linear model in feature space is a nonlinear model in input space. Deep nets replace a fixed
  kernel with a learned feature space.
- **Bag-of-features loses order.** The four spam features ignore word order and context entirely.
  Capturing context is exactly what [[attention-mechanism|attention]] adds, which is why hand-made
  features struggle with tasks like "write an email from these keywords" (book p. 1).

## Connections
- Builds on: [[vector]], [[machine-learning]]
- Leads to: [[deep-learning]], [[token-embedding]], [[word-embeddings]]
- Used by: [[logistic-regression]], [[text-classification]], [[supervised-learning]]
- Related: [[kernel-trick]], [[dimensionality-reduction]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1 "What is an LLM?" (spam-feature example), book pp. 3–4 (PDF pp. 25–26).
