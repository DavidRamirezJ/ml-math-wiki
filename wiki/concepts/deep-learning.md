---
title: Deep Learning
type: concept
tags: [deep-learning, neural-networks, machine-learning, ai]
difficulty: beginner
aliases: [DL, deep neural networks]
sources: [llm-ch01-understanding-large-language-models, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Deep Learning

> **One-liner:** Deep learning is machine learning with neural networks that have many layers, so the network can discover useful features by itself instead of a human hand-picking them.

## 🟢 Intuition (Dummy)
Picture an assembly line. Raw material (pixels, sound, words) goes in one end.
Each station (layer) does a small transformation and passes the result on. Early
stations notice simple things; later stations combine those into more complex
ideas. With *many* stations — a **deep** line — you can turn raw input into very
abstract understanding.

The big win: **you don't have to tell it what to look for.** In old-school spam
filters, a human expert chose clues like "counts of the words *prize*, *win*,
*free*", "number of exclamation marks", "ALL-CAPS words", "suspicious links".
A deep network learns its own clues straight from the raw email text.

It was originally inspired by the brain: lots of simple "neurons" connected
together. LLMs are deep learning applied to text.

## 🟡 Mechanics (Practitioner)
**Book definitions.**
- "A subset of machine learning that focuses on utilizing neural networks with
  **three or more layers** (also called deep neural networks) to model complex
  patterns and abstractions in data" (§1.1, p. 3).
- Fig. 1.1 label: "Machine learning with neural networks consisting of many layers."
- App. A: a subcategory of ML focused on training and applying deep neural
  networks; "deep" = multiple **hidden layers** of artificial neurons (nodes),
  which let the network model complex, **nonlinear** relationships (A.1.2, p. 253).

**One layer, in math** (standard notation; the book shows it in code later):

$$\mathbf h = \sigma(W\mathbf x + \mathbf b)$$

- $\mathbf x$ = the input [[vector]] to the layer.
- $W$ = the layer's weight [[matrix]] (learned).
- $\mathbf b$ = a [[bias-vector]] (learned).
- $\sigma$ = a nonlinear [[activation-function]] (e.g. [[relu]], [[gelu]]), applied to each entry.
- $\mathbf h$ = the layer's output, passed to the next layer.

**A deep network = layers composed:**

$$f(\mathbf x) = f_L\big(f_{L-1}(\cdots f_1(\mathbf x)\cdots)\big)$$

- $f_\ell$ = layer $\ell$; $L$ = number of layers ("deep" = $L$ is large, ≥ 3 per the book).

**How it learns.** Define a [[loss-function]], compute its [[gradient]] with
[[backpropagation]] (PyTorch's autograd does this automatically —
[[automatic-differentiation]]), and update the weights with
[[gradient-descent]]. These are the three pillars of [[pytorch]]: a tensor
library, an autograd engine, and deep-learning building blocks (A.1.1, p. 252).

**Traditional ML vs. deep learning (book's spam example, §1.1 p. 4; "simple
pattern recognition" vs. "unstructured data like images, audio, or text" from
A.1.2, p. 253):**

| | Traditional ML | Deep learning |
|---|---|---|
| Features | hand-crafted by experts ([[feature-extraction]]) | learned from raw data |
| Good at | simple pattern recognition | unstructured data: images, audio, text |
| Labels needed? | yes (spam / not spam) | yes, still (for supervised tasks) |

A concrete size: the GPT-2 model the book builds has 12 transformer blocks and
124 million parameters (§4.1, pp. 93–95); [[gpt-3]] has 96 transformer layers and
175 billion parameters (§1.6, p. 13).

## 🔴 Theory (Expert)
- **Why nonlinearity is essential.** Without $\sigma$, a stack of layers collapses:
  $W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2 = (W_2W_1)\mathbf x + (W_2\mathbf
  b_1+\mathbf b_2)$, a single affine map ([[linear-mapping]], [[affine-space]]).
  Depth only adds expressive power because of the nonlinear activations between layers.
- **Gradient computation.** Training relies on the [[chain-rule]] through the
  composition $f_L\circ\cdots\circ f_1$; reverse-mode differentiation
  ([[backpropagation]]) gets all parameter gradients for about the cost of one
  forward pass. Very deep stacks can suffer [[vanishing-and-exploding-gradients]],
  which motivates [[shortcut-connections]] and [[layer-normalization]] (Ch. 4).
- **Representation learning.** The book's key contrast is that deep learning does
  not need manual feature extraction (p. 4). Formally the network learns a
  feature map $\phi_\theta(\mathbf x)$ jointly with the final predictor, rather
  than fixing $\phi$ by hand.
- **Not just supervised.** Deep nets still need labels for supervised tasks (p. 4),
  but LLM pretraining derives labels from the text itself
  ([[self-supervised-learning]], [[next-token-prediction]]).
- **Place in AI.** Deep learning ⊂ [[machine-learning]] ⊂
  [[artificial-intelligence]] (Fig. 1.1; Fig. A.2). LLMs are a specific
  application of deep learning to text (Fig. 1.1 caption).

## Connections
- Builds on: [[machine-learning]], [[neural-network]], [[linear-mapping]], [[matrix]], [[chain-rule]], [[gradient]]
- Leads to: [[backpropagation]], [[multilayer-perceptron]], [[transformer-architecture]], [[large-language-model]], [[generative-ai]]
- Used by: [[pytorch]], [[large-language-model]]
- Related: [[feature-extraction]], [[activation-function]], [[automatic-differentiation]], [[gradient-descent]], [[overfitting-and-underfitting]]
- People: [[geoffrey-hinton]], [[yoshua-bengio]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Ch. 1 intro + §1.1, book pp. 2–4 (PDF pp. 24–26) — definition, ≥3 layers, Fig. 1.1, no manual feature extraction, spam features.
- Same file §1.6, book p. 13 — GPT-3 layer and parameter counts.
- Same file §4.1, book pp. 93–95 — GPT-2 124M configuration (12 blocks).
- Same file §A.1.1–A.1.2, book pp. 252–254 (PDF pp. 274–276) — PyTorch components, deep learning definition, hidden layers, Fig. A.2.
