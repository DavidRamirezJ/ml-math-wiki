---
title: Affine Space
type: concept
tags: [linear-algebra, foundations, deep-learning]
difficulty: advanced
sources: [mml-ch02-linear-algebra, llm-ch05-pretraining-on-unlabeled-data, llm-appA-introduction-to-pytorch]
created: 2026-06-04
updated: 2026-10-13
---

# Affine Space

> **One-liner:** An affine space is a vector subspace that's been picked up and shifted off the origin — a flat that no longer passes through zero.

## 🟢 Intuition (Dummy)
A [[vector-space|subspace]] always contains the origin (the zero vector) — lines and planes through
$(0,0,0)$. An **affine** space is what you get when you slide one of those flats away from the
origin: a line or plane that *doesn't* go through zero. It still looks flat and straight, it's just
been *offset*. You describe it with a **support point** (where it sits) plus a **direction space**
(which way it extends). Lines and planes in everyday space that miss the origin are affine spaces.

**In LLMs.** Every "linear" layer inside a neural network or a GPT model is really an *affine*
map: it multiplies by a weight matrix (stretch / rotate / squash) and then **adds a bias vector**
(slide everything over). The multiply alone would keep the origin fixed; the added bias is the
shift that moves the result off the origin, exactly the "picked up and shifted" idea above.

## 🟡 Mechanics (Practitioner)
**Definition (Def 2.25):** given a [[vector-space]] $V$, a point $\mathbf x_0\in V$, and a subspace
$U\subseteq V$,
$$ L = \mathbf x_0 + U := \{\mathbf x_0 + \mathbf u : \mathbf u\in U\} $$
is an **affine subspace** (a.k.a. *linear manifold*). $U$ is the **direction / direction space**;
$\mathbf x_0$ is the **support point**. ⚠️ If $\mathbf x_0\notin U$, then $\mathbf 0\notin L$, so $L$
is **not** a vector subspace.

**Parametric equation:** with $(\mathbf b_1,\dots,\mathbf b_k)$ an ordered basis of $U$,
$$ \mathbf x = \mathbf x_0 + \lambda_1\mathbf b_1 + \dots + \lambda_k\mathbf b_k, \quad \lambda_i\in\mathbb R $$

**The hierarchy in $\mathbb R^n$:**
| Dimension of $L$ | Name |
|---|---|
| 1 | **line** $\;\mathbf y = \mathbf x_0 + \lambda \mathbf b_1$ |
| 2 | **plane** $\;\mathbf y = \mathbf x_0 + \lambda_1\mathbf b_1 + \lambda_2\mathbf b_2$ |
| $n-1$ | **hyperplane** |

(In $\mathbb R^2$ a line is a hyperplane; in $\mathbb R^3$ a plane is a hyperplane.)

⚠️ **ML caution (from the book):** the literature often blurs *linear* vs *affine*, calling affine
maps "linear." A neural net layer $\mathbf x\mapsto W\mathbf x + \mathbf b$ is **affine**, not linear
(the bias $\mathbf b$ is the support point shift).

### In LLMs: `nn.Linear` is an affine map
*(Source: Build a Large Language Model (From Scratch), §A.5 and §5.5.)*

PyTorch's `torch.nn.Linear(d_in, d_out)` computes, for a batch of row vectors,

$$Y = XW^\top + \mathbf b$$

Reading it symbol by symbol:
- $X$ = the input [[matrix]], one example per row, shape $n\times d_{\text{in}}$ ($n$ = batch size);
- $W$ = the weight matrix, stored as shape $d_{\text{out}}\times d_{\text{in}}$ (out × in);
- $W^\top$ = its [[matrix-transpose|transpose]], so $XW^\top$ has shape $n\times d_{\text{out}}$;
- $\mathbf b$ = the [[bias-vector]] of length $d_{\text{out}}$, added to **every** row ([[broadcasting]]);
- $Y$ = the output, shape $n\times d_{\text{out}}$.

The $XW^\top$ part is the [[linear-mapping]] $\Phi$; $+\,\mathbf b$ is the translation. So by
Def 2.26 above, the layer is an **affine mapping**, and its image is an affine subspace with support
point $\mathbf b$.

**Worked numbers (book §A.5).** The multilayer perceptron of Listing A.4,
`NeuralNetwork(50, 3)`, is Linear(50→30) → ReLU → Linear(30→20) → ReLU → Linear(20→3)
(book p. 266). Each affine layer has $d_{\text{in}}d_{\text{out}}$ weights plus $d_{\text{out}}$ biases:

$$(50\cdot30+30) + (30\cdot20+20) + (20\cdot3+3) = 1530 + 620 + 63 = 2213$$

trainable parameters, which matches the book's printed count of 2,213 (book p. 267). The first
weight matrix has `torch.Size([30, 50])` (out × in) (book p. 267). The forward output carries
`grad_fn=<AddmmBackward0>`: **Addmm** = matrix multiply (`mm`) followed by addition (`Add`), i.e.
literally "linear part, then shift" (book pp. 268–269). Figure A.9 even draws the bias as an extra
gray **bias unit** node in each layer (book p. 265).

**Bias on or off in GPT (book §5.5).** To load OpenAI's GPT-2 weights, the book must set
`qkv_bias=True`, because OpenAI used bias vectors in the linear layers that compute queries, keys
and values (book p. 164). Each projection then becomes affine:
$\mathbf q = \mathbf xW_q + \mathbf b_q$, $\mathbf k = \mathbf xW_k + \mathbf b_k$,
$\mathbf v = \mathbf xW_v + \mathbf b_v$ ([[query-key-value]]). The book notes such bias vectors are
"not commonly used in LLMs anymore" because they don't improve modeling performance (book p. 164);
with `qkv_bias=False` the projections are purely *linear* maps (support point $\mathbf 0$).
GPT-2's LayerNorm also ends with an elementwise affine step, $\gamma\odot\hat{\mathbf x}+\beta$ (the
`g`/`b` gain and bias, book p. 166; see [[layer-normalization]]).

## 🔴 Theory (Expert)
- **Affine mapping (Def 2.26):** an affine map is a [[linear-mapping]] composed with a translation,
  $\mathbf x \mapsto \Phi(\mathbf x) + \mathbf a$. Many linear-map properties carry over (e.g.
  compositions of affine maps are affine).
- **Connection to linear systems:** the solution set of an *inhomogeneous* system $A\mathbf x=\mathbf b$
  is **either empty or an affine subspace** of dimension $n-\text{rk}(A)$ — this is exactly the
  "particular $+$ homogeneous" decomposition from [[gaussian-elimination]]. The homogeneous case
  $A\mathbf x=\mathbf 0$ is the special affine space with support point $\mathbf x_0=\mathbf 0$ (a
  genuine subspace).
- **In LLMs: why the nonlinearity is essential.** Compositions of affine maps are affine:
  $W_2(W_1\mathbf x+\mathbf b_1)+\mathbf b_2 = (W_2W_1)\mathbf x + (W_2\mathbf b_1+\mathbf b_2)$.
  Without the ReLU (or [[gelu]]) between layers, the whole 50→30→20→3 network of §A.5 would collapse
  to *one* affine map $\mathbb R^{50}\to\mathbb R^{3}$, no matter how many layers it has. This is why
  the book places nonlinear [[activation-function|activation functions]] between the hidden layers
  (book p. 266).
- **Homogeneous coordinates ("bias trick").** An affine map on $\mathbb R^n$ is a linear map on
  $\mathbb R^{n+1}$: append a constant 1 to the input,
  $\begin{bmatrix}W & \mathbf b\end{bmatrix}\begin{bmatrix}\mathbf x\\1\end{bmatrix} = W\mathbf x+\mathbf b$.
  The gray "bias unit" node in the book's Figure A.9 (book p. 265) is this constant-1 coordinate
  drawn as a neuron. It also explains the parameter count $d_{\text{out}}(d_{\text{in}}+1)$.
- **Linear-vs-affine wording across books.** The MML caution above and the LLM book agree on the
  math; the LLM book (like PyTorch) simply keeps the name "linear layer" for `nn.Linear`, which is
  affine whenever `bias=True`. Not a contradiction, only terminology.
- **Hyperplanes** are the geometric objects that separate classes in Support Vector Machines
  (Ch. 12) — the affine subspace is where the "linear" decision boundary lives.

## Connections
- Builds on: [[vector-space]], [[linear-mapping]], [[basis-and-rank]]
- Leads to: hyperplanes & SVMs (Ch. 12), separating hyperplanes ([[separating-hyperplane]], [[support-vector-machine]]), [[linear-layer]]
- Used by: neural network affine layers ([[linear-layer]], [[bias-vector]], [[query-key-value]], [[layer-normalization]]), decision boundaries, regression offsets
- Related: [[matrix-transpose]], [[multilayer-perceptron]], [[activation-function]], [[pretrained-weights]]
- People: —

## Sources
- `raw/mml-book.pdf` §2.8, Defs 2.25–2.26, printed pp. 61–62 (PDF pp. 66–67). Validated 2026-06-04.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.5, book pp. 265–269 (Figure A.9 bias units, Listing A.4, 2,213 parameters, `[30, 50]` weight shape, `AddmmBackward0`).
- Same PDF, §5.5, book pp. 164–166 (`qkv_bias=True`, bias vectors no longer common, LayerNorm `g`/`b`).
- Homogeneous-coordinates and collapse-of-affine-compositions arguments: standard linear algebra, not stated in either book.
