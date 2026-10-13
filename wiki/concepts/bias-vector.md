---
title: Bias Vectors in Linear Layers (qkv_bias)
type: concept
tags: [deep-learning, linear-algebra, transformer, llm, pytorch]
difficulty: beginner
sources: [llm-ch03-coding-attention-mechanisms, llm-ch04-implementing-a-gpt-model, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Bias Vectors in Linear Layers (qkv_bias)

> **One-liner:** A bias vector is a list of learned "offset" numbers that a layer adds after its matrix multiplication, so the output doesn't have to pass through zero.

## 🟢 Intuition (Dummy)
Think of a bathroom scale. The *weights* of a layer are like the scale's sensitivity: how much the needle moves per kilo. The *bias* is the little zero-adjust wheel: it shifts every reading up or down by a fixed amount, no matter what you put on the scale.

Without a bias, a layer that receives an all-zero input is forced to output all zeros. With a bias, it can output any baseline it likes. That's a small bit of extra freedom.

In LLMs, there's a fun twist: modern models mostly **switch the bias off** in the attention layers that make queries, keys and values, because it turns out not to help. But the original GPT-2 from OpenAI *did* use it, so when the book loads GPT-2's real weights, it has to turn the bias back on with one setting: `qkv_bias=True`.

## 🟡 Mechanics (Practitioner)
### The formula
A linear layer (`nn.Linear`) with a bias computes
$$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$
and without one it computes just $\mathbf{y} = W\mathbf{x}$.

Reading it symbol by symbol:
- $\mathbf{x}$ = the input vector (for example, one token's 768-number embedding),
- $W$ = the weight matrix (shape out × in), learned during training,
- $W\mathbf{x}$ = the [[matrix-multiplication|matrix-vector product]], a [[linear-mapping]],
- $\mathbf{b}$ = the **bias vector**, one learned number per output dimension,
- $\mathbf{y}$ = the output vector.

The book's appendix describes it exactly like this: a Linear layer "multiplies the inputs with a weight matrix and adds a bias vector" (§A.5, book p. 267); you can read it as `model.layers[0].bias`, and it is trainable (`requires_grad=True`) by default (book p. 267). Figure A.9 (book p. 265) draws the bias as an extra gray "bias unit" node in each input/hidden layer, connected to every unit of the next layer. When you run the forward pass, the output shows `grad_fn=<AddmmBackward0>`: "Addmm" = matrix multiplication (mm) followed by an addition (Add), and the addition is the bias (book p. 269). In PyTorch's batch (row-vector) convention it is $Y = XW^\top + \mathbf{b}$, where $\mathbf{b}$ is [[broadcasting|broadcast]] across every row of $X$.

### The query/key/value projections
In [[self-attention]], three linear layers turn each input $\mathbf{x}$ into a query, key and value ([[query-key-value]]). With `qkv_bias=True` each one is affine:
$$\mathbf{q} = \mathbf{x}W_q + \mathbf{b}_q,\qquad \mathbf{k} = \mathbf{x}W_k + \mathbf{b}_k,\qquad \mathbf{v} = \mathbf{x}W_v + \mathbf{b}_v$$
- $W_q, W_k, W_v$ = the three trainable projection matrices,
- $\mathbf{b}_q, \mathbf{b}_k, \mathbf{b}_v$ = their three bias vectors (only present when `qkv_bias=True`).

In code (from `SelfAttention_v2`, §3.4.2, and reused in `MultiHeadAttention`, §3.6):
```python
self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
```

### Where the setting lives
In the [[gpt-model-configuration|`GPT_CONFIG_124M`]] dictionary (§4.1, book p. 95):
```python
"qkv_bias": False        # Query-Key-Value bias
```
The book switches it off at first, "following the norms of modern LLMs". Then in §5.5 (book p. 164), to load OpenAI's [[pretrained-weights]]:
```python
NEW_CONFIG.update({"qkv_bias": True})
```
The book's reason: bias vectors "are not commonly used in LLMs anymore as they don't improve the modeling performance and are thus unnecessary", but we must match the pretrained weights' settings for consistency.

When the weights are copied in (Listing 5.5, book p. 165), OpenAI stores the three biases fused into one vector `c_attn.b` of length $3 \times 768 = 2304$. The book splits it into three equal parts:
```python
q_b, k_b, v_b = np.split(
    (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
gpt.trf_blocks[b].att.W_query.bias = assign(
    gpt.trf_blocks[b].att.W_query.bias, q_b)
```
(No transpose is needed, because a bias is a 1-D vector.)

### Worked numbers
- **Parameter count of one Linear layer** = $d_\text{in}\,d_\text{out} + d_\text{out}$ (weights + biases). For the book's toy MLP 50 → 30 → 20 → 3 (§A.5, book p. 267): $(50\cdot30+30)+(30\cdot20+20)+(20\cdot3+3) = 1530+620+63 = 2213$ trainable parameters. Of those, $30+20+3 = 53$ are biases.
- **GPT-2 124M** (derived, not printed in the book): `qkv_bias=True` adds $3 \times 768 = 2{,}304$ numbers per transformer block, so $12 \times 2{,}304 = 27{,}648$ in total. That's about 0.02% of the model: tiny, which is part of why dropping it costs nothing.
- Other GPT-2 layers *always* have biases in the original weights: the attention output projection (`c_proj.b`), both feed-forward layers (`c_fc.b`, `c_proj.b`), and the LayerNorm "shift" (`ln_*.b`) (book pp. 165–166). The final output head has **no** bias (`bias=False`, §4.1).

## 🔴 Theory (Expert)
- **Linear vs affine.** $\mathbf{x}\mapsto W\mathbf{x}$ is a [[linear-mapping]]: it maps $\mathbf{0}$ to $\mathbf{0}$ and preserves sums. Adding $\mathbf{b}$ makes it an **affine map** (see [[affine-space]]): a linear map followed by a translation. Affine maps can be written as linear maps in one higher dimension with the "homogeneous coordinate" trick: $\begin{bmatrix}W & \mathbf{b}\end{bmatrix}\begin{bmatrix}\mathbf{x}\\1\end{bmatrix}$.
- **Why Q/K biases are almost useless in attention.** An attention score is $\mathbf{q}_i^\top\mathbf{k}_j$. Expanding with biases:
  $$(\mathbf{x}_iW_q+\mathbf{b}_q)(\mathbf{x}_jW_k+\mathbf{b}_k)^\top = \mathbf{x}_iW_qW_k^\top\mathbf{x}_j^\top + \mathbf{x}_iW_q\mathbf{b}_k^\top + \mathbf{b}_q W_k^\top\mathbf{x}_j^\top + \mathbf{b}_q\mathbf{b}_k^\top.$$
  The 2nd and 4th terms do not depend on $j$, so they are constant across a [[softmax]] row and cancel out (softmax is shift-invariant). Only the 3rd term, a per-key offset, survives. A value bias $\mathbf{b}_v$ also passes straight through, because attention weights sum to 1 ([[convex-combination]]): $\sum_j a_{ij}(\mathbf{v}_j+\mathbf{b}_v)=\sum_j a_{ij}\mathbf{v}_j+\mathbf{b}_v$. That's a constant offset the next linear layer could absorb. (This is a standard argument, not from the book, but it explains the book's empirical claim that the biases "don't improve the modeling performance".)
- **Biases next to normalization are redundant.** A bias added right before a [[layer-normalization]] or [[batch-normalization]] step is partly cancelled by mean subtraction. That's why LayerNorm brings its own learned shift $\beta$, which plays the role of the bias.
- **Initialization.** PyTorch initializes `nn.Linear` biases uniformly in $\pm 1/\sqrt{d_\text{in}}$. Many LLM codebases zero them or drop them (see [[weight-initialization]]).

## Connections
- Builds on: [[linear-mapping]], [[matrix-multiplication]], [[vector]], [[affine-space]], [[linear-layer]]
- Leads to: [[query-key-value]], [[multi-head-attention]], [[gpt-model-configuration]], [[pretrained-weights]]
- Used by: [[self-attention]], [[feed-forward-network]], [[layer-normalization]] (its shift $\beta$), [[multilayer-perceptron]], [[gpt-2]]
- Related: [[model-parameters]], [[parameter-count-and-memory]], [[broadcasting]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.4.2, book pp. 72–73 (`nn.Linear(d_in, d_out, bias=qkv_bias)`); §3.6, book pp. 84–86 (`MultiHeadAttentionWrapper` and `MultiHeadAttention` keep `qkv_bias`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.1, book p. 95 (`"qkv_bias": False` and the explanation).
  > ⚠️ Book cross-reference error: §4.1 says the setting is revisited "in chapter 6" when the GPT-2 weights are loaded. Actually the weights are loaded in §5.5 (book p. 164).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 164–166 (`qkv_bias=True`, splitting `c_attn.b`, the other bias tensors).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book pp. 265–269 (Fig. A.9 bias units p. 265; `bias=True` default and the 2,213-parameter count pp. 266–267; `AddmmBackward0` p. 269).
