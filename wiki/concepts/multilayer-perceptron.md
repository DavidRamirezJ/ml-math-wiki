---
title: Multilayer Perceptron
type: concept
tags: [deep-learning, neural-networks, pytorch, architecture]
difficulty: beginner
sources: [llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Multilayer Perceptron

> **One-liner:** A multilayer perceptron (MLP) is a stack of fully connected layers with a simple nonlinearity in between, and it is the classic, plain-vanilla neural network.

## 🟢 Intuition (Dummy)
Imagine a team of judges arranged in rows. The first row looks at the raw data (say, 50
measurements) and each judge writes down an opinion. The second row doesn't see the raw data at
all; it only reads the first row's opinions and forms its own. The last row turns everything
into a final verdict: one score per possible answer.

That is an MLP:
- every judge in one row listens to **every** judge in the row before ("fully connected"),
- each judge's opinion is just a weighted sum of what they heard, plus a personal bias,
- and then a simple rule like "if negative, say 0" (the [[relu|ReLU]]) is applied. Without that
  little bend, stacking rows would be pointless — many straight lines in a row are still just one
  straight line.

Training tunes all the weights so the final verdicts match the right answers. Inside every GPT
[[transformer-block]] there is a small MLP, the [[feed-forward-network]], so this "boring" network is
a real part of an LLM.

## 🟡 Mechanics (Practitioner)

### The picture (Figure A.9, book p. 265)
Input layer → hidden layer 1 → hidden layer 2 → output layer. Each node is a *unit*, each edge is a
*weight*, and each non-output layer also has a gray **bias unit**. The figure is deliberately tiny
(10 inputs, 6 and 4 hidden units, 3 outputs); the code below uses 50 → 30 → 20 → 3.

### The math of one forward pass
$$\mathbf{h}_1 = \mathrm{ReLU}(W_1\mathbf{x} + \mathbf{b}_1),\qquad
\mathbf{h}_2 = \mathrm{ReLU}(W_2\mathbf{h}_1 + \mathbf{b}_2),\qquad
\mathbf{z} = W_3\mathbf{h}_2 + \mathbf{b}_3$$
Symbol by symbol:
- $\mathbf{x} \in \mathbb{R}^{50}$ = the input [[vector]] of 50 features ("$\in\mathbb{R}^{50}$" = "a list of 50 real numbers").
- $W_1 \in \mathbb{R}^{30\times50}$ = layer 1's weight [[matrix]] (30 rows = output units, 50 columns = inputs); $\mathbf{b}_1 \in \mathbb{R}^{30}$ = its [[bias-vector]].
- $W_1\mathbf{x}$ = [[matrix-multiplication]]: each of the 30 units takes a weighted sum of the 50 inputs. Adding $\mathbf{b}_1$ makes it an affine map.
- $\mathrm{ReLU}(v) = \max(0, v)$ applied entry by entry — the nonlinear [[activation-function]].
- $\mathbf{h}_1 \in \mathbb{R}^{30}$, $\mathbf{h}_2 \in \mathbb{R}^{20}$ = hidden representations.
- $W_2 \in \mathbb{R}^{20\times30}$, $W_3 \in \mathbb{R}^{3\times20}$; $\mathbf{z} \in \mathbb{R}^3$ = the [[logits]], one raw score per class.
- No activation after the last layer: PyTorch's losses apply softmax/sigmoid internally "for numerical efficiency and stability" (book p. 269).

### Listing A.4 — the book's MLP (book p. 266)
```python
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits

model = NeuralNetwork(50, 3)
```
- Subclass [[pytorch-nn-module]]: define layers in `__init__`, the data flow in `forward`. You never write `backward`; autograd does it.
- `torch.nn.Linear(in, out)` is the fully connected (also "feedforward") [[linear-layer]]: "multiplies the inputs with a weight matrix and adds a bias vector" (book p. 267).
- The outputs of one layer must **match** the inputs of the next (30 → 30, 20 → 20).
- `torch.nn.Sequential` just runs the layers in order, so `forward` is one line.

### Counting parameters
Each `Linear(d_in, d_out)` has $d_{\text{in}}\,d_{\text{out}}$ weights plus $d_{\text{out}}$ biases; ReLU has none.
$$\underbrace{50\cdot30+30}_{1530} + \underbrace{30\cdot20+20}_{620} + \underbrace{20\cdot3+3}_{63} = 2213$$
```python
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)   # 2213
model.layers[0].weight.shape                                         # torch.Size([30, 50])
```
**Exercise A.3** (book p. 275, answer in Appendix C p. 311): the 2-input, 2-output version used in the
training loop of §A.7 has $(2\cdot30+30)+(30\cdot20+20)+(20\cdot2+2) = 90+620+42 = 752$ parameters.

### Running it (book pp. 268–269)
```python
torch.manual_seed(123)          # reproducible random init
model = NeuralNetwork(50, 3)
X = torch.rand((1, 50))
out = model(X)                  # tensor([[-0.1262,  0.1080, -0.1792]], grad_fn=<AddmmBackward0>)
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)                      # tensor([[0.3113, 0.3934, 0.2952]])
```
- Weights start as **small random numbers**, which "breaks symmetry": if all units started equal, they
  would compute the same thing and get the same updates in [[backpropagation]] forever (book p. 268).
- The probabilities are nearly equal because the network is untrained.
- After training on the 2-feature toy data of §A.6 (`NeuralNetwork(2, 2)`, SGD with `lr=0.5`, 3 epochs),
  the MLP reaches training and test accuracy `1.0` (book pp. 274–278).

### Saving and loading (§A.8, book pp. 278–279)
```python
torch.save(model.state_dict(), "model.pth")       # dict: layer name -> weight/bias tensors
model = NeuralNetwork(2, 2)                         # must be the SAME architecture
model.load_state_dict(torch.load("model.pth"))
```
The `state_dict` holds only the parameters $\theta = \{W_\ell, \mathbf{b}_\ell\}$; the class definition supplies the
shapes. That is why the architecture must match exactly. See [[model-checkpointing]].

### In LLMs
The GPT model's [[feed-forward-network]] is a two-layer MLP applied to every token on its own:
`Linear(768, 3072)` → [[gelu]] → `Linear(3072, 768)` for GPT-2 124M (4× expansion of `emb_dim` 768).
It is the same idea as Listing A.4, with GELU instead of ReLU. Parameter count by the same rule:
$(768\cdot3072+3072)+(3072\cdot768+768) = 4{,}722{,}432$, which Exercise 4.1 confirms in code and notes is
about twice the 2,360,064 parameters of the attention module (Appendix C, book p. 302).

## 🔴 Theory (Expert)
- **Composition of affine maps and nonlinearities.** An $L$-layer MLP is
  $f_\theta = A_L \circ \phi \circ A_{L-1} \circ \cdots \circ \phi \circ A_1$ with $A_\ell(\mathbf{v}) = W_\ell\mathbf{v} + \mathbf{b}_\ell$.
  If $\phi$ were the identity, $f_\theta$ would collapse to a single affine map ($W_3W_2W_1\mathbf{x} + \dots$), since
  compositions of [[linear-mapping|linear maps]] are linear. The nonlinearity is what adds expressive power.
- **Universal approximation.** With one hidden layer of enough units and a non-polynomial activation
  (Cybenko 1989 for sigmoids; Leshno et al. 1993 in general), MLPs can approximate any continuous function
  on a compact set to any accuracy. Depth buys efficiency: some functions need exponentially many units
  when shallow but few when deep.
- **Batch convention.** PyTorch stores $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ and computes $Y = XW^\top + \mathbf{b}$ with
  rows of $X$ as examples (hence `grad_fn=<AddmmBackward0>`, a fused add + matmul).
- **Symmetry and initialization.** Permuting hidden units gives the same function, so the loss surface has
  many equivalent minima; random [[weight-initialization]] (PyTorch's `Linear` default is a Kaiming-uniform variant that
  draws weights and biases from $\mathcal{U}(-1/\sqrt{d_{\text{in}}},\ 1/\sqrt{d_{\text{in}}})$, consistent with the small values like $0.1174$ printed on p. 267) breaks the tie and keeps activations at a sensible scale.
- **Name.** "Perceptron" comes from Rosenblatt's (1958) single threshold unit; the modern MLP uses
  differentiable activations so it can be trained with gradients. A one-unit, sigmoid-output "MLP" with no
  hidden layer is [[logistic-regression]], which the book calls "a single-layer neural network" (§A.3).
- **Cost.** A forward pass costs about $2\sum_\ell d_{\ell-1}d_\ell$ FLOPs per example; parameters grow
  quadratically with width, which is why the GPT feed-forward block holds about two thirds of each
  transformer block's parameters (4,722,432 vs. 2,360,064 for attention, Exercise 4.1, book p. 302).

## Connections
- Builds on: [[linear-layer]], [[matrix-multiplication]], [[linear-mapping]], [[bias-vector]], [[activation-function]], [[relu]], [[neural-network]]
- Leads to: [[feed-forward-network]], [[transformer-block]], [[deep-learning]]
- Used by: [[training-loop]], [[forward-pass]], [[backpropagation]], [[model-checkpointing]], [[gpt-architecture]]
- Related: [[logistic-regression]], [[logits]], [[softmax]], [[weight-initialization]], [[pytorch-nn-module]], [[parameter-count-and-memory]], [[gelu]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book pp. 265–269 (Figure A.9; Listing A.4; 2213 parameters; weight shape `[30, 50]`; symmetry breaking; forward pass and softmax outputs).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7, book pp. 274–278 (training the `NeuralNetwork(2, 2)`; Exercise A.3).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.8, book pp. 278–279 (saving/loading the MLP `state_dict`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, book p. 311 (Exercise A.3 solution: 752 parameters); p. 302 (Exercise 4.1: feed-forward MLP 4,722,432 vs. attention 2,360,064 parameters).
- Universal approximation, Rosenblatt, initialization details: standard background, not in the book.
