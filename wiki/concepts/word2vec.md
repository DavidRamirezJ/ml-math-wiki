---
title: Word2Vec
type: concept
tags: [nlp, embeddings, representation-learning]
difficulty: intermediate
sources: [llm-ch02-working-with-text-data]
created: 2026-10-13
updated: 2026-10-13
---

# Word2Vec

> **One-liner:** Word2Vec is an early, very popular way to learn word embeddings by training a small neural network to guess which words appear near each other.

## 🟢 Intuition (Dummy)
"You shall know a word by the company it keeps." If two words keep showing up
next to the same neighbours — "duck" and "goose" both appear near "pond",
"feathers", "fly" — they probably mean something similar.

Word2Vec turns that idea into a game. Give the network a word, and ask it to
guess the words around it (or the other way round: give it the neighbours and
ask for the missing word). To win the game, the network has to put words that
share neighbours at nearby positions in its internal number-space. Those
positions are the **word embeddings**, and once training is over you keep them
and throw the game away.

## 🟡 Mechanics (Practitioner)
What the book says (book pp. 19–20):
- Word2Vec is "one of the earlier and most popular" methods for making word embeddings.
- It trains a neural network to **predict the context of a word given the target word, or vice versa**.
- The core idea: **words that appear in similar contexts tend to have similar meanings** (the *distributional hypothesis*).
- So, when the embeddings are projected to 2-D, similar words cluster together (Fig. 2.3: duck/eagle/goose; Germany–Berlin, England–London; long/longer/longest).
- You *can* feed pretrained Word2Vec vectors into an ML model, but LLMs usually learn their own embeddings in the input layer, which are tuned to the task and data at hand (book p. 20).

**A formula for the "guess the neighbour" game** (standard skip-gram form; *not
printed in the book*, added here so you can read papers):

$$P(c \mid w) = \frac{\exp(\mathbf{u}_c^{\top}\mathbf{v}_w)}{\sum_{c' \in \mathcal{V}} \exp(\mathbf{u}_{c'}^{\top}\mathbf{v}_w)}$$

Symbol by symbol:
- $P(c \mid w)$ = "the probability that word $c$ shows up as a neighbour, *given* the centre word $w$".
- $\mathbf{v}_w$ = the embedding of the centre word $w$ (this is the vector we keep).
- $\mathbf{u}_c$ = a second "context" vector for the neighbour word $c$.
- $\mathbf{u}_c^{\top}\mathbf{v}_w$ = their dot product ([[inner-product]]): big when the two vectors point the same way.
- $\exp(\cdot)$ and the fraction = a [[softmax]]: turns scores into probabilities that sum to 1.
- $\sum_{c'\in\mathcal{V}}$ = add up over every word $c'$ in the [[vocabulary]] $\mathcal{V}$.

Training maximises the log of these probabilities over all (centre, neighbour)
pairs in a window around each word — i.e. [[maximum-likelihood-estimation]].
The two directions the book mentions are usually called **skip-gram** (word →
context) and **CBOW**, "continuous bag of words" (context → word) — names not
used in the book.

**Tiny example.** In "the duck swims in the pond" with a window of 1, the centre
word "duck" produces the training pairs (duck, the) and (duck, swims). Because
"goose" will produce similar pairs elsewhere in the corpus, its vector $\mathbf{v}_{\text{goose}}$
gets pulled toward $\mathbf{v}_{\text{duck}}$.

## 🔴 Theory (Expert)
- **Static embeddings.** Word2Vec gives one vector per word type, whatever the
  sentence. "Bank" (river) and "bank" (money) share a vector. LLMs fix this with
  contextualised output embeddings from [[self-attention]] (book p. 20; Ch. 3).
- **Separate vs end-to-end.** Word2Vec embeddings are trained for a proxy task
  (neighbour prediction) and then frozen. LLM embeddings are trained jointly with
  the whole network on [[next-token-prediction]], so they are shaped by the final
  objective (book p. 20). Both are learned by gradient-based optimisation
  (background: Word2Vec with [[stochastic-gradient-descent]]; the book's LLM
  uses [[adamw-optimizer|AdamW]]).
- **Why similarity emerges.** With softmax training, two words with the same
  neighbour distribution are pushed toward the same optimum
  $\mathbf{v}$, so geometric closeness ([[norm]] distance or cosine,
  [[angle-between-vectors]]) tracks distributional similarity.
- **Cost of the full softmax.** The denominator sums over the whole vocabulary,
  which is expensive; the original papers used tricks such as negative sampling
  and hierarchical softmax (background, not from the book).
- **Origin** (background): Mikolov et al., Google, 2013. The book cites no paper
  for Word2Vec.
- **Word level only** (background). Word2Vec works on whole words, so it has an
  out-of-vocabulary problem that subword methods like [[byte-pair-encoding]]
  avoid.

## Connections
- Builds on: [[word-embeddings]], [[vector]], [[inner-product]], [[softmax]], [[maximum-likelihood-estimation]]
- Leads to: [[token-embedding]], [[embedding-layer]], [[self-attention]] (contextual embeddings)
- Related: [[embedding-dimension]], [[out-of-vocabulary-words]], [[dimensionality-reduction]]
- Used by: classic [[natural-language-processing]] pipelines
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §2.1, book pp. 19–20 (Word2Vec, distributional idea, Fig. 2.3, pretrained vs learned embeddings).
