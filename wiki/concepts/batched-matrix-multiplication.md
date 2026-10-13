---
title: Batched Matrix Multiplication
type: concept
tags: [linear-algebra, tensors, pytorch, attention, efficiency]
difficulty: intermediate
sources: [llm-ch03-coding-attention-mechanisms, llm-appA-introduction-to-pytorch]
created: 2026-10-13
updated: 2026-10-13
---

# Batched Matrix Multiplication

> **One-liner:** Batched matrix multiplication does many separate matrix multiplications at once — one per "slot" in a stack of matrices — with a single command.

## 🟢 Intuition (Dummy)
Picture a stack of homework sheets. Every sheet has its own little multiplication
problem, and every problem has the same shape. You could solve the sheets one by
one. Or you could hand the whole stack to a machine that solves all of them in
one go, sheet 1 with sheet 1, sheet 2 with sheet 2, and never mixes them up.

That's batched matrix multiplication. In an LLM the "stack" is made of
**texts in a batch** and **attention heads**. Each (text, head) pair has its own
little matrix problem, and the GPU solves all of them in parallel. It's the
trick that makes [[multi-head-attention]] fast instead of slow.

## 🟡 Mechanics (Practitioner)

### The rule: multiply the last two axes, repeat over the rest
For two ordinary matrices, the [[matrix-multiplication|matrix product]] is
$$(\mathbf{A}\mathbf{B})_{ij} = \sum_{k=1}^{n} A_{ik}\,B_{kj}, \qquad \mathbf{A}\in\mathbb{R}^{m\times n},\; \mathbf{B}\in\mathbb{R}^{n\times p}.$$
Reading it symbol by symbol:
- $\mathbf{A}$, $\mathbf{B}$ = the two matrices; $\mathbf{A}$ has $m$ rows and $n$ columns, $\mathbf{B}$ has $n$ rows and $p$ columns.
- $(\mathbf{A}\mathbf{B})_{ij}$ = the entry in row $i$, column $j$ of the answer.
- $\sum_{k=1}^{n}$ = "add up, for $k$ going from 1 to $n$".
- $A_{ik}B_{kj}$ = entry $k$ of row $i$ of $\mathbf{A}$ times entry $k$ of column $j$ of $\mathbf{B}$. So each answer entry is a [[inner-product|dot product]] of a row with a column.

**Batched version.** Now give the tensors extra leading axes, say a batch axis $b$ and a head axis $h$:
$$\mathcal{C}[b, h, :, :] = \mathcal{A}[b, h, :, :]\;\mathcal{B}[b, h, :, :] \quad \text{for every } b, h.$$
- $\mathcal{A}, \mathcal{B}, \mathcal{C}$ = 4D [[tensor|tensors]] (curly letters remind us they have more than two axes).
- $[b, h, :, :]$ = "fix the batch index $b$ and the head index $h$, and take the whole matrix that's left" (the colon `:` means "all of this axis").
- The equation says: the result in slot $(b,h)$ is just the ordinary matrix product of the two matrices sitting in slot $(b,h)$. Nothing crosses between slots.

Shapes: $(B, H, m, n) \;@\; (B, H, n, p) \to (B, H, m, p)$. The leading axes must match (or [[broadcasting|broadcast]]); the last two follow the usual matrix rule.

The book says it plainly: PyTorch's `@` on a 4D tensor carries out the
multiplication "between the two last dimensions (num_tokens, head_dim) and then
repeated for the individual heads" (book p. 89).

### The book's worked example (§3.6.2, book pp. 88–89)
A tensor with shape `(b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)`:
```python
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
print(a @ a.transpose(2, 3))
```
`a.transpose(2, 3)` swaps the last two axes (tokens and head_dim), so each
$3\times4$ matrix becomes $4\times3$ — see [[tensor-reshaping-and-transposing]].
The product has shape $(1, 2, 3, 3)$:
```
head 0: [[1.3208, 1.1631, 1.2879],      head 1: [[0.4391, 0.7003, 0.5903],
         [1.1631, 2.2150, 1.8424],               [0.7003, 1.3737, 1.0620],
         [1.2879, 1.8424, 2.0402]]               [0.5903, 1.0620, 0.9912]]
```
The book then does the same thing **the slow way**, one head at a time, and gets
"exactly the same results" (book p. 89):
```python
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
```
**Check one number by hand.** Entry $(0,0)$ of head 0 is row 1 dotted with itself:
$0.2745^2 + 0.6584^2 + 0.2775^2 + 0.8573^2 = 0.0754 + 0.4335 + 0.0770 + 0.7350 = 1.3208$. ✔

Notice each $3\times3$ result is symmetric. That's because $\mathbf{M}\mathbf{M}^\top$ is a [[gram-matrix]]: entry $(i,j)$ is the dot product of token rows $i$ and $j$.

### Where it's used in the LLM
In the book's `MultiHeadAttention` (Listing 3.5, book pp. 86–87):
```python
attn_scores = queries @ keys.transpose(2, 3)   # (b, h, T, d_h) @ (b, h, d_h, T) -> (b, h, T, T)
context_vec = attn_weights @ values            # (b, h, T, T) @ (b, h, T, d_h) -> (b, h, T, d_h)
```
- The first line gives the [[attention-scores]] $Q_iK_i^\top$ for every text and every head at once.
- The second line mixes the [[query-key-value|values]] with the [[attention-weights]] to make the [[context-vector|context vectors]] of every head.

Already in §3.5 the book used a 3D version: `queries @ keys.transpose(1, 2)` on shape `(2, 6, 2)` keeps the batch axis at position 0 and gives scores of shape `(2, 6, 6)` (book p. 81).

### Why bother?
The chapter summary puts it this way: multi-head attention can be built by
stacking causal-attention modules, but "a more efficient way of creating
multi-head attention modules involves batched matrix multiplications" (book p. 91).
One big kernel launch keeps the GPU busy; a Python loop over heads
(`[head(x) for head in self.heads]`, book p. 85) runs them one after another. → [[gpu-computing]]

The book also names a second, related saving (book p. 90): `MultiHeadAttention` needs **only one
matrix multiplication** to compute the keys, `keys = self.W_key(x)` (same for queries and values),
whereas the wrapper repeats this projection — "computationally one of the most expensive steps" —
for each head. One wide $W_q$ replaces $W_{q1}, W_{q2}, \dots$ (Figure 3.26, book p. 88); the result is
then split into heads by [[tensor-reshaping-and-transposing|reshaping]], and only *then* does batched
matmul take over. → [[weight-splitting-multi-head-attention]]

## 🔴 Theory (Expert)
- **Formal statement.** Let $\mathcal{A}\in\mathbb{R}^{B_1\times\cdots\times B_r\times m\times n}$ and $\mathcal{B}\in\mathbb{R}^{B_1\times\cdots\times B_r\times n\times p}$. Batched matmul is the map
  $$\mathcal{C}_{\beta,i,j} = \sum_{k=1}^{n}\mathcal{A}_{\beta,i,k}\,\mathcal{B}_{\beta,k,j},\qquad \beta\in[B_1]\times\cdots\times[B_r].$$
  In Einstein notation (`torch.einsum("...ik,...kj->...ij", A, B)`), the batch multi-index $\beta$ is a "free" index that appears in both inputs and the output and is **not** summed. Only $k$ is contracted.
- **Block-diagonal view.** Stacking the $N=\prod B_\ell$ slices, batched matmul equals one ordinary product of block-diagonal matrices: $\operatorname{diag}(A_1,\dots,A_N)\,\operatorname{diag}(B_1,\dots,B_N) = \operatorname{diag}(A_1B_1,\dots,A_NB_N)$. So it is a [[linear-mapping]] that acts independently on each block — exactly why heads don't "talk" to each other inside attention. The mixing between heads happens later, in the [[output-projection]].
- **Broadcasting.** PyTorch `torch.matmul` broadcasts the batch axes: a $(T,T)$ mask or a $(1,H,\dots)$ tensor is virtually repeated to fit. The rule is applied only to leading axes; the last two must satisfy the inner-dimension rule $n=n$. → [[broadcasting]]
- **Cost.** $N$ products of size $m\times n$ by $n\times p$ cost $N\cdot m n p$ multiply–adds, the same as the loop. The win is not fewer FLOPs; it's hardware utilization (one fused kernel, better memory reuse, no Python overhead). For attention scores that's $B\cdot H\cdot T^2 d_h = B\,T^2 d_{out}$ — quadratic in [[context-length]], which is the bottleneck [[flashattention]] attacks.
- **Gram structure.** Each slice $\mathcal{A}_\beta\mathcal{A}_\beta^\top$ in the book's demo is symmetric positive semidefinite: $\mathbf{x}^\top(\mathbf{M}\mathbf{M}^\top)\mathbf{x} = \lVert \mathbf{M}^\top\mathbf{x}\rVert^2 \ge 0$. → [[gram-matrix]], [[symmetric-positive-definite-matrix]]
- **Associativity per slice** carries over, so $(\mathcal{A}\mathcal{B})\mathcal{C}=\mathcal{A}(\mathcal{B}\mathcal{C})$ slice-wise; frameworks exploit this when choosing contraction order.

## Connections
- Builds on: [[matrix-multiplication]], [[matrix]], [[inner-product]], [[tensor]], [[matrix-transpose]], [[broadcasting]]
- Leads to: [[multi-head-attention]], [[weight-splitting-multi-head-attention]], [[scaled-dot-product-attention]], [[flashattention]]
- Used by: [[self-attention]], [[causal-attention]], [[attention-scores]], [[context-vector]], [[gpu-computing]]
- Related: [[gram-matrix]], [[tensor-reshaping-and-transposing]], [[linear-mapping]], [[pytorch]]
- People: [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.2, book pp. 86–89 (Listing 3.5; batched-matmul demo tensor `a` and both head results; "carried out between the two last dimensions … repeated for the individual heads").
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.2, book pp. 88, 90 (Figure 3.26; one projection matmul instead of one per head).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.6.1, book p. 85 (sequential head loop in the wrapper) and chapter 3 summary, book p. 91.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §3.5, book p. 81 (3D batched `queries @ keys.transpose(1, 2)`).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.2.3, book p. 261 (`.matmul` and `@` are the same operation).
- The einsum, block-diagonal, and cost remarks are standard linear algebra, not from the book.
