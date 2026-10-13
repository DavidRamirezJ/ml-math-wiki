---
title: Output Projection (out_proj)
type: concept
tags: [attention, transformer, linear-layer, multi-head-attention]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-ch05-pretraining-on-unlabeled-data, llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# Output Projection (out_proj)

> **One-liner:** The output projection is one last linear layer that takes the glued-together results of all attention heads and mixes them into a single, tidy output vector.

## 🟢 Intuition (Dummy)
Imagine a panel of 12 experts. Each one reads the same sentence and writes a short
note about what they noticed — one cares about grammar, another about who did what,
another about which word a pronoun points to. If you just staple the 12 notes
together, you get a long report where every expert's part is stuck in its own corner.

The **output projection** is the editor who reads the stapled report and rewrites it
as one blended summary, taking a bit from this expert and a bit from that one. In
[[multi-head-attention]] the heads work separately; `out_proj` is the first place
where their results actually get **combined**.

## 🟡 Mechanics (Practitioner)

### Where it sits
In the book's `MultiHeadAttention` (Listing 3.5, book pp. 86–87):
```python
self.out_proj = nn.Linear(d_out, d_out)        # "Uses a Linear layer to combine head outputs"
...
context_vec = (attn_weights @ values).transpose(1, 2)
context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
context_vec = self.out_proj(context_vec)       # "Adds an optional linear projection"
return context_vec
```
First the heads are glued back together ([[tensor-reshaping-and-transposing|reshaped]] to $(b, T, d_{out})$), then `out_proj` is applied to every token's vector.

### The formula
$$\operatorname{MultiHead}(X) = \big[\,Z_1 \,\|\, Z_2 \,\|\, \cdots \,\|\, Z_h\,\big]\, W_o + \mathbf{b}_o$$
Symbol by symbol:
- $X$ = the input token vectors (one row per token).
- $Z_i$ = the [[context-vector|context vectors]] that head $i$ produced (a $T\times d_h$ matrix: one row per token, $d_h$ = `head_dim` numbers per row).
- $\|$ = "glue side by side" (concatenate along the feature axis). Gluing $h$ blocks of width $d_h$ gives width $h\,d_h = d_{out}$.
- $W_o$ = the output-projection weight matrix, size $d_{out}\times d_{out}$. It's trainable.
- $\mathbf{b}_o$ = a [[bias-vector]] of length $d_{out}$, added to every row. `nn.Linear` has a bias by default, so `out_proj` has one even though the Q/K/V layers use `bias=qkv_bias` (default `False`).
- The whole thing is just a [[linear-layer]] applied to each token separately.

(PyTorch detail: `nn.Linear` stores its weight with shape $(d_{out}, d_{in})$, i.e. it stores $W_o^\top$, and computes `y = x @ weight.T + bias` — the same thing as $\mathbf{x}W_o + \mathbf{b}_o$. See [[linear-layer]].)

### Why it actually mixes heads
Split $W_o$ into horizontal blocks $W_o^{(1)},\dots,W_o^{(h)}$, each $d_h \times d_{out}$. Then
$$[Z_1\|\cdots\|Z_h]\,W_o = \sum_{i=1}^{h} Z_i\, W_o^{(i)}.$$
- $\sum_{i=1}^{h}$ = "add up over all heads".
- $Z_i W_o^{(i)}$ = head $i$'s output, pushed through its own slice of $W_o$.

So every output number is a weighted blend of **all** heads' numbers. Without $W_o$, output column 1 would only ever come from head 1.

### Worked numbers
- **Tiny example (book p. 90):** `MultiHeadAttention(d_in=3, d_out=2, context_length=6, dropout=0.0, num_heads=2)`, so `head_dim = 1`. Here `out_proj` is `Linear(2, 2)`: $2\cdot2 + 2 = 6$ parameters. The output has shape `[2, 6, 2]`, e.g. first row `[0.3190, 0.4858]`. These numbers differ from the wrapper's output (book p. 85) partly because the wrapper has no `out_proj` (our observation).
- **GPT-2 small size** (Exercise 3.3, book p. 91: 12 heads, $d_{in}=d_{out}=768$): `out_proj` has $768^2 + 768 = 590{,}592$ parameters. The Q/K/V matrices together have $3\cdot768^2 = 1{,}769{,}472$. So one attention module has $2{,}360{,}064$ parameters, and `out_proj` is exactly one quarter of the weight matrices. (Arithmetic ours.) Over 12 transformer blocks, `out_proj` alone is $12\times590{,}592 = 7{,}087{,}104$ parameters. → [[parameter-count-and-memory]]

### Is it necessary?
The book is upfront: "This output projection layer is not strictly necessary (see appendix B for more details), but it is commonly used in many LLM architectures, which is why I added it here for completeness" (book pp. 89–90). Appendix B points to He and Hofmann (2023), "Simplifying Transformer Blocks", which found "it's possible to also achieve good performance without the value weight matrix and projection layer" (book p. 292).

### Later in the book
- **Loading GPT-2 weights** (§5.5, book pp. 165–166): OpenAI calls this layer `c_proj`. The book copies `attn.c_proj.w.T` into `att.out_proj.weight` and `attn.c_proj.b` into `att.out_proj.bias`. → [[pretrained-weights]], [[gpt-2]]
- **LoRA** (appendix E, book pp. 330–333): `out_proj` is one of the linear layers that gets wrapped with `LinearWithLoRA`, next to `W_query`, `W_key`, `W_value`. → [[low-rank-adaptation]]

## 🔴 Theory (Expert)
- **Origin.** The original transformer (Vaswani et al. 2017) defines $\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$ with $W^O\in\mathbb{R}^{hd_v\times d_{model}}$. The book's `out_proj` is this $W^O$ (plus a bias). (Background, not cited in §3.6.)
- **Merging with $W_v$.** Head $i$'s contribution is $A_i V_i W_o^{(i)} = A_i X\,(W_v^{(i)} W_o^{(i)})$, where $A_i$ = head $i$'s [[attention-weights]]. So per head only the product $W_{OV}^{(i)} = W_v^{(i)}W_o^{(i)}\in\mathbb{R}^{d_{in}\times d_{out}}$ matters, a matrix of [[basis-and-rank|rank]] at most $d_h$ (a [[low-rank-approximation|low-rank]] map). This "OV circuit" view explains why He & Hofmann can drop $W_v$ and $W_o$: the pair is over-parameterized, and the residual stream plus the following [[feed-forward-network]] can absorb the remaining [[linear-mapping]].
- **Expressivity without it.** If $W_o = I$, the output is a direct sum of head outputs; each output coordinate depends on one head only. A full $W_o$ makes the map head-mixing. Since the next sublayer (after the [[shortcut-connections|residual add]] and [[layer-normalization]]) is itself linear-then-nonlinear, the extra $W_o$ is not strictly needed for expressivity, which is why it's "optional" in principle.
- **Initialization / scaling.** Some GPT-2-style implementations scale the init of residual-path projections (including `c_proj`) by $1/\sqrt{2N}$ for $N$ layers, to keep the residual stream's variance from growing with depth (general knowledge; not in the book). → [[weight-initialization]]
- **In fused implementations** (e.g. PyTorch `nn.MultiheadAttention`, referenced in appendix B, book p. 292) it's called `out_proj` too and is applied after the fused [[scaled-dot-product-attention]] / [[flashattention]] kernel.

## Connections
- Builds on: [[linear-layer]], [[matrix-multiplication]], [[linear-mapping]], [[bias-vector]], [[context-vector]]
- Leads to: [[transformer-block]], [[gpt-architecture]], [[low-rank-adaptation]]
- Used by: [[multi-head-attention]], [[weight-splitting-multi-head-attention]], [[gpt-2]], [[pretrained-weights]]
- Related: [[tensor-reshaping-and-transposing]], [[parameter-count-and-memory]], [[query-key-value]], [[low-rank-approximation]]
- People: [[ashish-vaswani]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.2, book pp. 86–90 (Listing 3.5 `self.out_proj = nn.Linear(d_out, d_out)`; "not strictly necessary … commonly used in many LLM architectures"; output `[2, 6, 2]`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6, Exercise 3.3, book p. 91 (GPT-2 size module; parameter count derived).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` appendix B, Ch. 3 references, book p. 292 (He & Hofmann 2023, "Simplifying Transformer Blocks"; PyTorch `MultiHeadAttention`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §5.5, book pp. 165–166 (`c_proj` → `out_proj`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 330–333 (`replace_linear_with_lora`, p. 330; printed model shows `(out_proj): LinearWithLoRA` with `Linear(768, 768, bias=True)`, p. 333).
- The OV-circuit, Vaswani $W^O$, and init-scaling remarks are background, not from the book.
