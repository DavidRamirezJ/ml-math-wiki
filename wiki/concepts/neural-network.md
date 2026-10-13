---
title: Neural Network (Deep Neural Network)
type: concept
tags: [deep-learning, neural-networks, llm, foundations]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Neural Network (Deep Neural Network)

> **One-liner:** A neural network is a big stack of simple "multiply, add, squash" steps with adjustable knobs, and a *deep* one just has many of those steps stacked up.

## 🟢 Intuition (Dummy)
Picture an assembly line in a factory. Raw material (say, the words of an email) goes in
at one end. At each station a worker does one small thing to it, then passes it on. At
the end you get a finished product, for example a verdict: "spam" or "not spam".

A neural network is that assembly line, but every worker has a row of little dials
(the **weights**). At first the dials are set at random, so the output is junk. During
**training** you show the line lots of examples where you already know the right answer,
and you nudge every dial a tiny bit so that the output gets closer to that answer. Repeat
millions of times and the line becomes good at its job.

The idea was first inspired by the brain: lots of simple "neurons" connected to each
other (book p. 253). "Deep" just means there are many stations in the middle, called
**hidden layers**. Raschka puts the bar at "three or more layers" (book p. 3).

Why should you care? A [[large-language-model]] like ChatGPT *is* a deep neural network,
one that is trained on a huge pile of text (book p. 2). Everything in this book (attention,
transformer blocks, fine-tuning) is a way of building or training one of these lines.

One more nice thing: in older machine learning a human expert had to decide what to
measure, for example "count the exclamation marks" or "look for the word *free*".
A deep network learns its own useful measurements from the raw data, so nobody has to
hand-pick those features (book pp. 3–4). See [[feature-extraction]]. This is also why deep
learning is "particularly good at handling unstructured data like images, audio, or text", which
makes it well suited for LLMs (book p. 253). (AI as a whole also includes non-learning approaches
such as rule-based systems, genetic algorithms, expert systems, fuzzy logic and symbolic reasoning,
book p. 4.)

## 🟡 Mechanics (Practitioner)

### One layer
A single fully connected layer takes an input vector and does three things: multiply by
a weight matrix, add a bias, apply a nonlinear function:

$$\mathbf{h} = \sigma(W\mathbf{x} + \mathbf{b})$$

Reading it symbol by symbol:
- $\mathbf{x}$ = the input [[vector]] (the numbers coming in, e.g. 50 features).
- $W$ = the **weight [[matrix]]**: one row per output unit, one column per input. These are trainable parameters.
- $W\mathbf{x}$ = matrix–vector product: each output is a weighted sum (a [[inner-product|dot product]]) of the inputs.
- $\mathbf{b}$ = the **[[bias-vector]]**, one extra adjustable number per output unit.
- $W\mathbf{x}+\mathbf{b}$ = an [[affine-space|affine]] map (a [[linear-mapping]] plus a shift).
- $\sigma$ = an [[activation-function]] such as [[relu]] ($\mathrm{ReLU}(a)=\max(0,a)$), applied to each entry separately.
- $\mathbf{h}$ = the layer's output, the "hidden" representation passed on to the next layer.

### Stacking layers = "deep"
A deep network is a composition of $L$ such layers:

$$f_\theta(\mathbf{x}) = f_L\big(f_{L-1}(\cdots f_1(\mathbf{x})\cdots)\big)$$

- $f_\ell$ = layer number $\ell$ (one "multiply, add, squash" step).
- $\circ$-style nesting = the output of one layer is the input of the next.
- $\theta$ = all [[model-parameters]] together, i.e. every $W_\ell$ and $\mathbf{b}_\ell$.
- $f_\theta(\mathbf{x})$ = the network's final output for input $\mathbf{x}$.

Why the nonlinearity $\sigma$ matters: without it, $W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$ is
again just one affine map, so stacking would buy you nothing. The squash between layers is
what lets the network model "complex, nonlinear relationships in the data" (book p. 253).

### The book's worked example: a small multilayer perceptron (Appendix A)
Raschka's running example is a [[multilayer-perceptron]] with two hidden layers (Listing A.4, book p. 266):

```python
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),   # 1st hidden layer
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),           # 2nd hidden layer
            torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs),  # output layer
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits
```

With `model = NeuralNetwork(50, 3)` the math is

$$\mathbf{h}_1 = \mathrm{ReLU}(W_1\mathbf{x}+\mathbf{b}_1),\quad \mathbf{h}_2 = \mathrm{ReLU}(W_2\mathbf{h}_1+\mathbf{b}_2),\quad \mathbf{z} = W_3\mathbf{h}_2+\mathbf{b}_3$$

- $\mathbf{x}\in\mathbb{R}^{50}$: 50 input features ("$\in\mathbb{R}^{50}$" = "is a list of 50 real numbers").
- $W_1\in\mathbb{R}^{30\times 50}$, $\mathbf{b}_1\in\mathbb{R}^{30}$: first hidden layer, 30 units.
- $W_2\in\mathbb{R}^{20\times 30}$, $\mathbf{b}_2\in\mathbb{R}^{20}$: second hidden layer, 20 units.
- $W_3\in\mathbb{R}^{3\times 20}$, $\mathbf{b}_3\in\mathbb{R}^{3}$: output layer, 3 classes.
- $\mathbf{z}\in\mathbb{R}^3$: the raw output scores, called [[logits]].

**Counting the knobs.** A [[linear-layer]] with $d_{in}$ inputs and $d_{out}$ outputs has
$d_{in}d_{out}+d_{out}$ parameters (weights plus biases). So

$$\underbrace{50\cdot30+30}_{1530}+\underbrace{30\cdot20+20}_{620}+\underbrace{20\cdot3+3}_{63}=2213,$$

which is exactly what the book prints: "Total number of trainable model parameters: 2213"
(book p. 267). The same network with 2 inputs and 2 outputs has
$(2\cdot30+30)+(30\cdot20+20)+(20\cdot2+2)=752$ parameters (Exercise A.3, book p. 311).
Compare: an LLM has "tens or even hundreds of billions of parameters" (book p. 2).

### How it learns
Training is [[supervised-learning]]: show inputs with known answers, measure the error with a
[[loss-function]], compute how each weight affects the error with [[backpropagation]]
(PyTorch does this for you via [[automatic-differentiation]]), then nudge each weight
downhill with [[gradient-descent]]. The "parameters" of an LLM are exactly these
"adjustable weights in the network that are optimized during training" (book p. 2).

## 🔴 Theory (Expert)
- **Function class.** A depth-$L$ network is the parametric family
  $\mathcal{F}=\{f_\theta=\sigma_L\circ A_L\circ\cdots\circ\sigma_1\circ A_1\}$ with affine $A_\ell(\mathbf{x})=W_\ell\mathbf{x}+\mathbf{b}_\ell$.
  Training picks $\theta$ by [[empirical-risk-minimization]]. With a cross-entropy loss this is
  [[maximum-likelihood-estimation]] of a categorical model whose parameters are the softmax of the logits.
- **Collapse without nonlinearity.** If every $\sigma_\ell$ is the identity, then
  $f_\theta(\mathbf{x}) = (W_L\cdots W_1)\mathbf{x} + \mathbf{b}'$, a single affine map with rank at most
  $\min_\ell d_\ell$. Depth only adds expressive power through the nonlinearities.
- **Universal approximation** (standard result, not in the book): one hidden layer with a
  non-polynomial activation and enough units can approximate any continuous function on a
  compact set arbitrarily well. Depth makes many functions exponentially cheaper to represent,
  which is one reason practice favours deep nets.
- **Row-vector convention.** PyTorch stores a batch as rows, so a linear layer computes
  $Y = XW^\top + \mathbf{b}$ with $X\in\mathbb{R}^{n\times d_{in}}$ and $W\in\mathbb{R}^{d_{out}\times d_{in}}$.
  That is why `model.layers[0].weight.shape` prints `torch.Size([30, 50])` (out × in) (book p. 267).
- **Representation learning.** The book's contrast with traditional ML (handcrafted features
  such as trigger words "prize", "win", "free", exclamation-mark counts, all-caps words,
  suspicious links; book p. 4) is the classic argument that deep nets learn the feature map
  $\phi(\mathbf{x})$ jointly with the classifier. Labels are still needed for supervised tasks
  in both settings (book p. 4).
- **Where LLMs sit.** Figure 1.1 (book p. 3) nests AI ⊃ ML ⊃ deep learning ⊃ LLMs, with
  [[generative-ai]] inside deep learning and overlapping LLMs. Figure A.2 (book p. 253)
  shows deep learning ⊂ ML ⊂ AI. An LLM is a [[transformer-architecture|transformer]]-based deep
  network; its building blocks (attention, [[feed-forward-network]], [[layer-normalization]],
  [[shortcut-connections]]) are all differentiable layers in the sense above.
- **Training-dynamics pitfalls** of depth ([[vanishing-and-exploding-gradients]]) motivate
  shortcut connections and normalization layers in the GPT block (Ch. 4).

## Connections
- Builds on: [[vector]], [[matrix]], [[linear-mapping]], [[affine-space]], [[inner-product]], [[activation-function]]
- Leads to: [[multilayer-perceptron]], [[deep-learning]], [[transformer-architecture]], [[large-language-model]], [[gpt-architecture]]
- Trained by: [[supervised-learning]], [[loss-function]], [[backpropagation]], [[gradient-descent]], [[stochastic-gradient-descent]]
- Implemented with: [[pytorch]], [[pytorch-nn-module]], [[linear-layer]], [[relu]], [[logits]]
- Related: [[model-parameters]], [[feature-extraction]], [[machine-learning]], [[artificial-intelligence]], [[parameter-count-and-memory]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1 "What is an LLM?", book pp. 2–4 (LLM = deep neural network; parameters; "three or more layers"; manual feature extraction; Figure 1.1).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.1.2 "Defining deep learning", book pp. 252–254 (hidden layers, nonlinear relationships, Figure A.2).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book pp. 265–267 (Listing A.4 MLP, 2213 parameters, weight shape [30, 50]).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise A.3, book p. 311 (752 parameters).
