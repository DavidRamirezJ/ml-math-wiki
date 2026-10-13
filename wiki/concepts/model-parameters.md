---
title: Model Parameters (Trainable Weights)
type: concept
tags: [deep-learning, llm, pytorch, optimization, linear-algebra]
difficulty: beginner
sources: [llm-ch01-understanding-large-language-models, llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Model Parameters (Trainable Weights)

> **One-liner:** Parameters are the adjustable numbers inside a model that training tweaks, little by little, until the model makes good predictions.

*Also called:* trainable parameters, weight parameters, weights.

## 🟢 Intuition (Dummy)
Picture a huge mixing desk in a recording studio, covered in knobs. Sound goes in one side, music
comes out the other, and what it sounds like depends on where every knob is set. At the start the
knobs are set at random, so the output is noise. Training is a patient sound engineer who listens,
hears what is wrong, and nudges every knob a tiny bit in the direction that makes it sound better,
millions of times.

Each knob is a **parameter**. A small toy network in the book has 2,213 knobs. GPT-2 "small" has
about 124 million. GPT-3 has 175 billion. When people say a model is "large", the number of knobs is
half of what they mean (the other half is how much text it was trained on) (book p. 2).

The important thing: you never set the knobs by hand. You only pick *how many* knobs there are and
how they are wired (the architecture). Training sets their values.

## 🟡 Mechanics (Practitioner)

**Definition (book p. 93).** "In the context of deep learning and LLMs like GPT, the term
'parameters' refers to the trainable weights of the model. These weights are essentially the internal
variables of the model that are adjusted and optimized during the training process to minimize a
specific [[loss-function]]."

**Where they live: weight matrices and bias vectors.** A [[linear-layer]] computes
$$\mathbf y = W\mathbf x + \mathbf b .$$
Symbol by symbol: $\mathbf x$ = the input vector; $W$ = the **weight matrix** (every entry is a
parameter); $\mathbf b$ = the **[[bias-vector]]** (every entry is a parameter); $\mathbf y$ = the output
vector. With `bias=False` (as in the book's query/key/value layers, `qkv_bias=False`) the layer is just
$\mathbf y = W\mathbf x$.

**Counting them.** A matrix $W\in\mathbb R^{m\times n}$ ($m$ rows, $n$ columns) holds $m\cdot n$
parameters. The book's example: a $2048\times2048$ weight matrix has
$$2048\times2048 = 4{,}194{,}304 \text{ parameters (book p. 94).}$$
A full linear layer with bias has $d_{\text{in}}\,d_{\text{out}} + d_{\text{out}}$ parameters
($d_{\text{in}}$ = input size, $d_{\text{out}}$ = output size).

**Worked example 1: the toy MLP in Appendix A** (Listing A.4: 50 inputs → 30 → 20 → 3 outputs):
$$\underbrace{50\cdot30+30}_{1530}+\underbrace{30\cdot20+20}_{620}+\underbrace{20\cdot3+3}_{63}=2213 .$$
PyTorch prints exactly that (book p. 267):
```python
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)   # 2213
```
`model.parameters()` walks over every parameter tensor, `.numel()` counts its elements, and
`requires_grad=True` marks a tensor as **trainable** (updated by training; it is the default for
`nn.Linear` weights and biases, book p. 267). [[relu|ReLU]] layers have no parameters.

**Worked example 2: GPT-2 "124M"** (`GPT_CONFIG_124M`: vocab 50,257, context 1,024, emb_dim 768,
12 heads, 12 layers). Counting the book's `GPTModel` gives **163,009,536** parameters (book p. 120).
The gap is because original GPT-2 uses [[weight-tying]]: the token-embedding matrix and the output
head are both $50{,}257\times768 = 38{,}597{,}376$ numbers, and GPT-2 shares one copy. Subtracting the
output head gives **124,412,160**, the real GPT-2 size (book p. 121). (The original GPT-2 report said 117 million; this was later corrected to 124 million, book
p. 93. The book later loads pretrained weights for the larger GPT-2 variants with 345, 762, and 1,542 million
parameters, book p. 93.) At 4 bytes per `float32`
parameter, 163M parameters take about **621.83 MB** (book p. 122). See [[parameter-count-and-memory]].

**Other trainable parameters in the book.**
- **Attention projections** $W_q, W_k, W_v$ (§3.4): three matrices that turn each embedded token into a
  query, key, and value vector. They are what makes self-attention *trainable* (book pp. 65–66). See
  [[query-key-value]].
- **[[layer-normalization|LayerNorm]] scale and shift** (§4.2): two trainable vectors $\gamma$
  (`scale`, initialised to ones) and $\beta$ (`shift`, zeros), each of length `emb_dim` (book p. 103).

**⚠️ Two different "weights".** The book has a sidebar on this (book p. 66):
| | Weight **parameters** | [[attention-weights|Attention **weights**]] |
|---|---|---|
| What | entries of $W_q,W_k,W_v$, linear layers, … | softmax-normalised scores $\alpha_{ij}$ |
| Fixed after training? | yes, learned and then frozen at inference | no, recomputed for every input |
| Stored in `state_dict`? | yes | no |

**How training changes them.** One step of [[gradient-descent]]:
$$\theta \leftarrow \theta - \eta\,\nabla_\theta\mathcal L(\theta).$$
Symbol by symbol: $\theta$ (theta) = all parameters stacked together; $\leftarrow$ = "replace with";
$\eta$ (eta) = the [[learning-rate]]; $\nabla_\theta\mathcal L$ = the [[gradient]] of the loss with
respect to every parameter (computed by `loss.backward()`); $\mathcal L$ = the loss. In PyTorch:
`optimizer.zero_grad(); loss.backward(); optimizer.step()` (book pp. 146–147). The book actually uses
[[adamw-optimizer|AdamW]], a smarter per-parameter version of this rule.

**Accessing them in PyTorch** (sidebar, book p. 129). Weights "are stored in linear layers";
after `new_layer = torch.nn.Linear(...)` the weight matrix is `new_layer.weight` (and the bias
`new_layer.bias`), and `model.parameters()` gives all trainable parameters, weights and biases,
at once. In the Appendix A MLP, `model.layers[0].weight.shape` is `torch.Size([30, 50])`:
PyTorch stores a `Linear(50, 30)` weight as (out, in) (book p. 267).

**Saving and loading.** A model's parameters are its `state_dict`, "a Python dictionary that maps
each layer in the model to its trainable parameters (weights and biases)" (book p. 279):
```python
torch.save(model.state_dict(), "model.pth")
model = NeuralNetwork(2, 2)                 # same architecture needed first
model.load_state_dict(torch.load("model.pth"))
```
The architecture must match exactly, otherwise the tensors do not fit (book p. 279).

## 🔴 Theory (Expert)
- **A network is a parametric function family.** The architecture fixes a family
  $\{f_\theta:\theta\in\mathbb R^P\}$; training picks one member by (approximately) solving
  $\theta^*=\arg\min_\theta \frac1N\sum_n \ell(f_\theta(\mathbf x_n),y_n)$, i.e.
  [[empirical-risk-minimization]]. $P$ (the parameter count) is a crude measure of
  [[model-capacity]]: GPT-2 small $P\approx1.24\times10^8$; the largest GPT-2 has
  $P\approx1.5\times10^9$, and GPT-3 is "fundamentally the same in terms of model architecture"
  but scaled up from 1.5 billion to $P\approx1.75\times10^{11}$ and trained on more data
  (book p. 94). Training GPT-3 on a single V100 GPU would take an estimated 355 years (Lambda
  Labs figure quoted on book p. 94).
- **Parameters vs. [[hyperparameter|hyperparameters]].** Parameters are learned by gradients;
  hyperparameters (learning rate, `emb_dim`, number of layers, dropout rate) are chosen by the person
  and fix the *shape* of $\theta$ and the training procedure.
- **Non-identifiability.** Many different $\theta$ give the same function (e.g. permuting hidden units,
  or rescaling across a ReLU). So "the" trained weights are one point in a large equivalent set; this
  is why individual weights are rarely interpretable.
- **Initialization matters.** `nn.Linear` is preferred over `nn.Parameter(torch.rand(...))` because it
  uses an optimized [[weight-initialization]] scheme, which makes training more stable (book pp. 71–72).
  `torch.manual_seed(123)` makes the random starting point reproducible (book p. 268).
- **Weight tying as a parameter constraint.** Tying imposes $W_{\text{out}}=E_{\text{tok}}$, which
  removes 38.6M free parameters. It saves memory, but the author finds separate matrices train better,
  as do modern LLMs (book p. 121).
- **Freezing and low-rank updates.** Setting `requires_grad=False` removes parameters from the
  optimization ([[layer-freezing]]). [[low-rank-adaptation|LoRA]] goes further: keep the pretrained $W$
  frozen and learn only a low-rank update $\Delta W = AB$, so the trainable count drops from $mn$ to
  $r(m+n)$ (Appendix E).

## Connections
- Builds on: [[matrix]], [[vector]], [[linear-mapping]], [[tensor]]
- Leads to: [[gradient-descent]], [[backpropagation]], [[parameter-count-and-memory]], [[weight-tying]],
  [[pretrained-weights]], [[model-checkpointing]]
- Used by: [[linear-layer]], [[query-key-value]], [[layer-normalization]], [[gpt-model-configuration]],
  [[low-rank-adaptation]], [[layer-freezing]]
- Contrast: [[hyperparameter]], [[attention-weights]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §1.1 ("large" = parameters + data), book p. 2 (PDF p. 24).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4 (trainable $W_q,W_k,W_v$; weight-parameters sidebar; `nn.Linear` init), book pp. 64–73 (PDF pp. 86–95).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1 (definition, $2048\times2048$ example, GPT-2/GPT-3 sizes), book pp. 93–94 (PDF pp. 115–116).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.2 (LayerNorm `scale`/`shift`), book p. 103 (PDF p. 125).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.6 (163,009,536 / 124,412,160 / 621.83 MB), book pp. 120–122 (PDF pp. 142–144).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.1 (weight-parameters sidebar), book p. 129 (PDF p. 151); §5.2 (training loop), book pp. 146–147 (PDF pp. 168–169).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (2,213 parameters, `requires_grad`), book pp. 265–268 (PDF pp. 287–290).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.8 (`state_dict`), book pp. 278–279 (PDF pp. 300–301).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.1 (LoRA low-rank update $\Delta W$), book p. 322 (PDF p. 344).
