---
title: PyTorch nn.Module and nn.Sequential
type: concept
tags: [pytorch, deep-learning, neural-network, software, llm-building]
difficulty: beginner
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-appA-introduction-to-pytorch, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# PyTorch nn.Module and nn.Sequential

> **One-liner:** `nn.Module` is PyTorch's "Lego brick" class for a piece of a neural network — it holds the learnable weights and says how input flows through them — and `nn.Sequential` snaps bricks together in a straight line.

## 🟢 Intuition (Dummy)
Imagine a kitchen appliance. Inside it are its parts (blades, a motor) and a recipe for
what happens when you press "start". An `nn.Module` is exactly that for neural networks:
- in its **constructor** (`__init__`) you install the parts — the layers with their
  trainable weights;
- in its **`forward`** method you write the recipe — "first do this layer, then that one".

The nice bonus: PyTorch automatically keeps a list of every trainable number inside,
so the optimizer can find them and [[backpropagation]] can compute their gradients. And
since a module can contain other modules, you can build big things — a whole GPT —
out of small ones: attention modules inside transformer blocks inside the GPT model.

`nn.Sequential` is the "conveyor belt" version: if your layers simply run one after the
other, you list them in order and PyTorch wires them up.

## 🟡 Mechanics (Practitioner)
**The pattern** (Listing A.4, §A.5, book p. 266) — a [[multilayer-perceptron]] with two
hidden layers:
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
What each piece does (book pp. 265–267):
- **subclass `torch.nn.Module`**: gives you parameter tracking and lots of helper methods.
- **`__init__`**: define the layers. `super().__init__()` must run first so the module
  machinery is set up.
- **`forward`**: how the input passes through the layers; together the operations form
  a **[[computational-graph]]**.
- **`backward`**: you normally **don't** write it — autograd derives it (§A.5, book p. 265).
- **`nn.Sequential`**: "not required, but it can make our life easier" when layers run in
  a fixed order; then `forward` just calls `self.layers(x)` (book pp. 266–267).
- Calling `model(X)` runs `forward` automatically (book p. 268).
- The output of the last layer is called the **[[logits]]**; softmax is *not* applied
  inside the model, because PyTorch's losses fuse it in for numerical stability (book p. 269).

**What `nn.Sequential` computes, in math.** It is plain function composition:
$$\mathrm{Sequential}(f_1, f_2, \dots, f_K)(\mathbf x) = f_K\big(\cdots f_2(f_1(\mathbf x))\cdots\big) = (f_K\circ\cdots\circ f_1)(\mathbf x)$$
- $f_1,\dots,f_K$ = the listed sub-modules, in order.
- $\circ$ = "composed with": $(g\circ f)(\mathbf x)$ means "apply $f$ first, then $g$".

For Listing A.4 that is:
$$\mathbf h_1 = \mathrm{ReLU}(W_1\mathbf x + \mathbf b_1),\quad
\mathbf h_2 = \mathrm{ReLU}(W_2\mathbf h_1 + \mathbf b_2),\quad
\mathbf z = W_3\mathbf h_2 + \mathbf b_3$$
- $\mathbf x\in\mathbb R^{50}$ = input features; $\mathbf z\in\mathbb R^{3}$ = the 3 logits.
- $W_1\in\mathbb R^{30\times50},\ W_2\in\mathbb R^{20\times30},\ W_3\in\mathbb R^{3\times20}$ = weight matrices; $\mathbf b_i$ = [[bias-vector|bias vectors]].
- [[relu|ReLU]]$(a)=\max(0,a)$, applied to each entry.
- The output size of one layer must **match** the input size of the next (30→30, 20→20).

**Printing and inspecting** (book pp. 266–267):
```
NeuralNetwork(
  (layers): Sequential(
    (0): Linear(in_features=50, out_features=30, bias=True)
    (1): ReLU()
    (2): Linear(in_features=30, out_features=20, bias=True)
    (3): ReLU()
    (4): Linear(in_features=20, out_features=3, bias=True)
  )
)
```
- Sub-modules are indexable: `model.layers[0].weight` (shape `torch.Size([30, 50])`),
  `model.layers[0].bias`.
- **Counting trainable parameters**:
  ```python
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  # Total number of trainable model parameters: 2213
  ```
  Check by hand: $(50\cdot30+30)+(30\cdot20+20)+(20\cdot3+3) = 1530+620+63 = 2213$.
  ReLU has no parameters. With 2 inputs and 2 outputs (§A.7) it's
  $90+620+42 = 752$ (Exercise A.3, Appendix C).
- `model.named_parameters()` yields `(name, tensor)` pairs like `layers.0.0.weight` —
  used in §4.4's `print_gradients` to print each layer's gradient size (book p. 112).

**Other container modules in the book:**
- `nn.ModuleList([...])`: a list of sub-modules whose parameters are registered, but you
  write the loop in `forward` yourself — handy when you need extra logic, like adding a
  shortcut (Listing 4.5, `ExampleDeepNeuralNetwork`, book pp. 110–111).
- `nn.Parameter(tensor)`: registers a raw tensor as a trainable parameter. `SelfAttention_v1`
  uses `nn.Parameter(torch.rand(d_in, d_out))` for $W_q, W_k, W_v$ (Listing 3.1, book
  pp. 70–71); `SelfAttention_v2` swaps them for `nn.Linear(d_in, d_out, bias=qkv_bias)`
  (Listing 3.2, book pp. 72–73), getting a better [[weight-initialization]] for free.

**Modes and gradients:**
- `model.train()` / `model.eval()` switch layers like [[dropout]] and
  [[batch-normalization]] between training and inference behaviour; for the plain MLP
  they're redundant but best practice (§A.7, book pp. 275–276). See [[train-vs-eval-mode]].
- `with torch.no_grad(): out = model(X)` → same values, but no `grad_fn` and no graph:
  `tensor([[-0.1262, 0.1080, -0.1792]])` (book pp. 268–269).

## 🔴 Theory (Expert)
- **A module is a parameterized function.** Mathematically an `nn.Module` represents a
  map $f_\theta:\mathbb R^{d_{in}}\to\mathbb R^{d_{out}}$ together with its parameter set
  $\theta$. Nesting modules builds a **tree**; `model.parameters()` is a traversal of that
  tree that returns the disjoint union $\theta = \bigsqcup_m \theta_m$ of all sub-module
  parameters — the vector the optimizer updates.
- **`Sequential` = composition, `ModuleList` = a free-form graph.** `Sequential` can
  only express a chain $f_K\circ\cdots\circ f_1$. Anything with a branch or merge — a
  shortcut $\mathbf x + f(\mathbf x)$ (§4.4), multi-head attention's split and
  concatenation — needs a custom `forward`. The GPT model in Ch. 4 still uses
  `nn.Sequential` for its stack of [[transformer-block|transformer blocks]], because each
  block hides its residual logic inside its own `forward`.
- **Composition and the chain rule.** Because the network is a composition, its
  [[jacobian]] factors as $J_{f_K\circ\cdots\circ f_1} = J_{f_K}\cdots J_{f_1}$; autograd
  records each module's operations in `forward` and applies this product in reverse order
  ([[backpropagation]], [[automatic-differentiation]]). That's why you only ever write
  `forward`.
- **Row-vector convention.** PyTorch modules act on batches $X\in\mathbb R^{n\times d_{in}}$
  (one example per row), so a linear layer computes $XW^\top+\mathbf b$; this is why
  $W$ is stored as $d_{out}\times d_{in}$ — the book shows `torch.Size([30, 50])` for
  `Linear(50, 30)` (book p. 267) and hints that `nn.Linear` stores the weight "in transposed
  form" (Exercise 3.1, §3.4, book p. 73); the $XW^\top$ formula is PyTorch's convention, see [[linear-layer]].
- **Parameter count is linear algebra.** A `Linear(d_in, d_out)` adds
  $d_{in}d_{out}+d_{out}$ parameters; summing over modules gives model size and memory
  ([[parameter-count-and-memory]]). Frozen parameters (`requires_grad=False`) are still
  in `parameters()` but excluded by the `if p.requires_grad` filter — exactly how the
  LoRA appendix counts 0 then 2,666,528 trainable parameters (§E.4).

> **From [[llm-appE-parameter-efficient-fine-tuning-with-lora]] — module surgery.** A model is a
> *tree* of modules. `model.named_children()` gives the direct children as `(name, module)`
> pairs, and `setattr(model, name, new_module)` swaps one child for another. The book's
> `replace_linear_with_lora(model, rank, alpha)` (§E.4, book p. 330) walks this tree
> recursively: if a child is an `nn.Linear`, it is replaced by `LinearWithLoRA(child, rank, alpha)`,
> which keeps the old layer and adds a [[low-rank-adaptation|LoRA]] path; otherwise the function
> calls itself on that child. The new `LoRALayer` stores $A$ and $B$ as `nn.Parameter`s, so they
> are registered automatically and are trainable, while the wrapped original weights stay frozen
> because `requires_grad = False` was set *before* the swap (book pp. 331–332). The printed model
> on pp. 332–333 shows every `Linear` (Q/K/V, `out_proj`, both feed-forward layers, `out_head`)
> now nested as `LinearWithLoRA(linear=Linear(...), lora=LoRALayer())`.

## Connections
- Builds on: [[pytorch]], [[tensor]], [[linear-mapping]], [[matrix-multiplication]], [[computational-graph]]
- Leads to: [[linear-layer]], [[multilayer-perceptron]], [[transformer-block]], [[gpt-architecture]], [[training-loop]]
- Used by: [[self-attention]] (`SelfAttention_v1/v2`), [[multi-head-attention]], [[feed-forward-network]], [[low-rank-adaptation]]
- Related: [[model-parameters]], [[weight-initialization]], [[train-vs-eval-mode]], [[forward-pass]], [[backpropagation]], [[logits]], [[shortcut-connections]]
- People: —

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4.2 (Listings 3.1–3.2, `nn.Module`, `nn.Parameter`, `nn.Linear`), book pp. 70–74.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.4 (`nn.ModuleList`, Listing 4.5, `named_parameters`), book pp. 109–113.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5 (Listing A.4, `nn.Sequential`, 2213 parameters, forward pass, `no_grad`), book pp. 265–269.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.7 (`train()`/`eval()`), book pp. 274–278; Appendix C, Exercise A.3 (752 parameters), book p. 311.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4 (Listings E.5–E.6, `replace_linear_with_lora` with `named_children`/`setattr`, printed model), book pp. 329–333.
