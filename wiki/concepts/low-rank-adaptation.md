---
title: LoRA (Low-Rank Adaptation)
type: concept
tags: [llm, fine-tuning, parameter-efficient-fine-tuning, linear-algebra, matrix-decomposition]
difficulty: intermediate
sources: [llm-appE-parameter-efficient-fine-tuning-with-lora, llm-ch07-fine-tuning-to-follow-instructions, llm-ch06-fine-tuning-for-classification]
created: 2026-10-13
updated: 2026-10-13
---

# LoRA (Low-Rank Adaptation)

> **One-liner:** Instead of retraining a giant weight matrix, LoRA freezes it and learns a small "correction" built from two skinny matrices, so you fine-tune about 2% of the numbers and still get almost the same result.

## 🟢 Intuition (Dummy)
Picture a huge, expensive painting (the [[pretrained-weights|pretrained model]]). A customer wants
it adjusted a bit: warmer colours, a brighter sky. You could repaint the whole canvas
([[fine-tuning|full fine-tuning]]), but that costs a lot, and if 100 customers each want their own
version, you need 100 full copies of the painting.

LoRA's trick: leave the original painting untouched and put a thin, transparent **overlay** on
top of it. The overlay is cheap to make, because it is only allowed to contain a few simple
"brush-stroke patterns". To show customer #37's version, you lay customer #37's overlay on the
original. You store one big painting plus 100 tiny overlays, not 100 paintings.

Why do a few simple patterns work at all? When you adapt a model to one new task (spotting spam,
following instructions), the change it needs is usually not "change everything in every
direction". It is "nudge things along a few important directions". "Low-rank" is the math word for
"only a few independent directions" (see [[basis-and-rank]]). LoRA bets that the update lives in a
small subspace of the weight space, and it only learns that small part.

The book's result: GPT-2 small has about **124 million** weights. With LoRA, only about **2.7
million** are trained (almost 50× fewer), and the spam classifier still reaches about **97–98% test
accuracy** (Appendix E).

## 🟡 Mechanics (Practitioner)

### 1. Start from the ordinary weight update
Take one [[linear-layer]] with weight [[matrix]] $W$. Normal training ([[backpropagation]] +
[[gradient-descent]]) learns how much to change every weight:

$$W_{\text{updated}} = W + \Delta W$$

Read it symbol by symbol:
- $W$ = the pretrained weight matrix of one layer (for GPT-2 small attention, $768\times768$).
- $\Delta W$ ("delta W") = the **update matrix**: the amount to add to each weight so that the
  [[loss-function]] goes down. It has **exactly the same shape** as $W$, so it is just as big.
- $W_{\text{updated}}$ = the weights after the change.

### 2. LoRA's idea: approximate the update with two small matrices

$$\Delta W \approx AB \qquad\Longrightarrow\qquad W_{\text{updated}} = W + AB$$

- $\approx$ = "is approximately equal to".
- $A$ = a tall, thin matrix of shape $d_{in}\times r$ (in the code: `(in_dim, rank)`).
- $B$ = a short, wide matrix of shape $r\times d_{out}$ (in the code: `(rank, out_dim)`).
- $AB$ = their [[matrix-multiplication|matrix product]]. Its shape is $d_{in}\times d_{out}$, the
  same as $W$, so we can add it to $W$.
- $r$ = the **rank**, the "inner dimension" that $A$ and $B$ share. It is a
  [[hyperparameter]] you choose, and it is much smaller than $d_{in}$ and $d_{out}$ (the book uses
  $r=16$ against $d=768$). See [[lora-rank-and-alpha]].

Because $AB$ is built through a bottleneck of width $r$, it can have **at most $r$ independent
directions**: $\operatorname{rank}(AB)\le r$. That is the "low-rank" in the name.

### 3. Keep the two paths separate (distributive law)
For an input row vector $x$, the [[linear-mapping|linear map]] distributes over the sum:

$$x(W + \Delta W) = xW + x\Delta W \qquad\text{and for LoRA}\qquad x(W + AB) = xW + xAB$$

- $x$ = the layer input (the book multiplies from the left, like PyTorch's `x @ W`).
- $xW$ = the output of the **frozen** pretrained path.
- $xAB$ = the output of the small LoRA path. The code computes it as $(xA)B$: first squeeze $x$
  down to $r$ numbers, then expand back to $d_{out}$. The big $d_{in}\times d_{out}$ matrix $AB$ is
  never built.

So the pretrained weights never change. The LoRA matrices are kept on the side and are added when
you use the model. This is what lets you store one base model plus many small per-customer
"overlays" (book p. 324).

### 4. The full LoRA layer, with scaling
The book's `LoRALayer` also multiplies by a scaling factor $\alpha/r$:

$$\text{LoRA}(x) = \frac{\alpha}{r}\,\big(x\,A\,B\big),\qquad
y = \underbrace{xW + b}_{\text{frozen Linear}} + \underbrace{\tfrac{\alpha}{r}\,xAB}_{\text{LoRA}}
  = x\Big(W + \tfrac{\alpha}{r}AB\Big) + b$$

- $\alpha$ ("alpha") = a scaling factor that sets how strongly the LoRA path affects the output.
- $\frac{\alpha}{r}$ = the actual multiplier. With the book's defaults $\alpha = r = 16$ it is exactly 1.
- $b$ = the frozen [[bias-vector]] of the original Linear layer.
- $\frac{\alpha}{r}AB$ = the learned stand-in for $\Delta W$.

### 5. Initialization: start as a no-op
- $A$ gets random values: `kaiming_uniform_(A, a=math.sqrt(5))`, "the same initialization used
  for Linear layers in PyTorch" ([[weight-initialization]]).
- $B$ starts as **all zeros**.

So at the start $AB = 0$ (anything times a zero matrix is zero), and $W + 0 = W$. The model behaves
exactly like the pretrained model. The book checks this: before LoRA training the accuracies are
**46.25% / 45.00% / 48.75%** (train / val / test), identical to chapter 6 (book p. 334).

### 6. Tiny worked example (my numbers, $d_{in}=d_{out}=3$, $r=1$)

$$A=\begin{bmatrix}1\\2\\0\end{bmatrix},\quad B=\begin{bmatrix}1 & -1 & 3\end{bmatrix},\quad
AB=\begin{bmatrix}1&-1&3\\2&-2&6\\0&0&0\end{bmatrix}$$

- $AB$ has 9 entries but was built from only $3+3=6$ numbers.
- Every row of $AB$ is a multiple of $[1,-1,3]$, so it has rank 1.
- With $x=[1,1,1]$: $xA = 1+2+0 = 3$, then $(xA)B = [3,-3,9]$. Building $AB$ first gives
  $x(AB) = [1+2+0,\ -1-2+0,\ 3+6+0] = [3,-3,9]$. Same answer (associativity), but the LoRA way never
  forms the big matrix.

### 7. Counting parameters
For one adapted layer:

$$\#\text{params}_{\text{LoRA}} = d_{in}\,r + r\,d_{out} = r\,(d_{in}+d_{out})
\qquad\text{vs.}\qquad \#\text{params}_{\Delta W} = d_{in}\,d_{out}$$

- $d_{in}\,r$ = entries of $A$; $r\,d_{out}$ = entries of $B$.
- One $768\times768$ attention matrix: full $\Delta W$ = 589,824 numbers; LoRA with $r=16$ =
  $16\cdot1536 = 24{,}576$ numbers (24× fewer).

The whole GPT-2 small spam classifier, with `rank=16, alpha=16` on every Linear layer
(book p. 332; table re-derived and checked against the book's total):

| Layer (per transformer block) | $d_{in}\to d_{out}$ | LoRA params |
|---|---|---|
| `W_query`, `W_key`, `W_value`, `out_proj` (4×) | 768→768 | 4 × 16·1536 = 98,304 |
| `ff.layers[0]` | 768→3072 | 16·3840 = 61,440 |
| `ff.layers[2]` | 3072→768 | 16·3840 = 61,440 |
| **per block** | | **221,184** |
| 12 blocks | | 2,654,208 |
| `out_head` | 768→2 | 16·770 = 12,320 |
| **Total** | | **2,666,528** |

Trainable parameters: **124,441,346** before freezing → **0** after freezing → **2,666,528** after
adding LoRA. That is about 2.1% of the model ("almost 50×" fewer; exactly ≈ 46.7×). Stored in
float32 that is ≈ 10.7 MB per task instead of ≈ 498 MB for a full copy (my arithmetic, 4 bytes per
number; see [[parameter-count-and-memory]]).

### 8. The book's code (Listings E.5, E.6)

```python
import math
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank
    def forward(self, x):
        x = (self.alpha / self.rank) * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)
```

Workflow: freeze everything first (`param.requires_grad = False`, see [[layer-freezing]]), *then*
call `replace_linear_with_lora(model, rank=16, alpha=16)`. The new $A$ and $B$ are created after the
freeze, so they are the only trainable tensors. LoRA wraps every Linear layer in
[[multi-head-attention]], the [[feed-forward-network]], and the output head.

### 9. Training and results (Listing E.7, spam classification)
- [[adamw-optimizer|AdamW]], `lr=8e-4`, `weight_decay=0.1`, `num_epochs=5`, `eval_freq=50`,
  `eval_iter=5`, using chapter 6's `train_classifier_simple` (book p. 334).
- Loss falls from 3.820 / 3.462 (train / val) at step 0 to 0.019 / 0.252 at step 600; "Training
  completed in 12.10 minutes" (book pp. 334–335). The book expects ~15 min on an M3 MacBook Air and
  under half a minute on a V100/A100.
- Final full-dataset accuracy: **train 100.00%, validation 96.64%, test 98.00%** (book p. 336).
  The small train-vs-val gap hints at a little [[overfitting]].
- In this small example LoRA was *slower* than chapter 6's plain fine-tuning, because the extra
  LoRA path costs forward-pass compute. For bigger models, where backpropagation dominates, LoRA
  usually makes training faster (book p. 335).
- **Exercise 7.4** applies the same three helpers to [[instruction-fine-tuning]] of GPT-2 medium.
  On an Nvidia L4: 1.30 min with LoRA vs 1.80 min without (≈ 28% faster), with a Llama 3 judge
  score of about 50, "in the same ballpark" as full fine-tuning (book p. 310, Appendix C).

> ⚠️ Book inconsistencies (flagged, not resolved by the book):
> - Book p. 336 prints test accuracy **98.00%**, but the paragraph below says "96.64% and **97.33%**".
> - Figure E.5's caption says "over six epochs"; the code uses `num_epochs = 5`.
> - Figure E.3's caption says "approximate the weight update matrix (W)"; it should be ΔW (the
>   annotation inside the figure is correct).

> ⚠️ Notation: the original paper (Hu et al. 2021) writes the layer column-style, $h = W_0x + BAx$,
> with $B\in\mathbb R^{d\times r}$ set to zero and $A\in\mathbb R^{r\times k}$ random. The book writes it
> row-style, $xW + xAB$, with $A$ random and $B$ zero. This is not a disagreement: transpose one form
> and you get the other with the letters swapped. In both, **the factor next to the output starts at
> zero**. Many other texts quote the paper form $\Delta W = BA$; read it as the same thing.

## 🔴 Theory (Expert)
- **The function class.** LoRA restricts the update to
  $\mathcal M_r=\{\Delta W\in\mathbb R^{d_{in}\times d_{out}}:\operatorname{rank}(\Delta W)\le r\}$.
  Every matrix of rank $\le r$ factors as $AB$ with inner dimension $r$ (take a rank factorization,
  or a truncated [[singular-value-decomposition|SVD]] $U_r\Sigma_rV_r^\top$ and split $\Sigma_r$). So
  the parametrization $AB$ reaches *exactly* $\mathcal M_r$. Note $\mathcal M_r$ is not a vector
  subspace (the sum of two rank-1 matrices can have rank 2); it is a cone/variety of dimension
  $r(d_{in}+d_{out}-r)$. The book's phrase "a smaller dimensional subspace" is the intuitive picture.
- **Link to Eckart–Young.** If you already knew the full-fine-tuning update $\Delta W^\star$, the
  best rank-$r$ choice would be its truncated SVD ([[low-rank-approximation]], error
  $\sigma_{r+1}$ in spectral norm). LoRA never computes $\Delta W^\star$. It learns $A,B$ directly by
  gradient descent on the task loss, which is cheaper and is all you need if the useful update is
  approximately low-rank. Hu et al. report that this "intrinsic rank" is often very small (a
  paper claim, not stated in the book).
- **Non-uniqueness / gauge freedom.** For any invertible $M\in\mathbb R^{r\times r}$,
  $(AM)(M^{-1}B)=AB$. Many $(A,B)$ pairs give the same update, so the loss surface in $(A,B)$ has
  flat directions. This is harmless for optimization in practice.
- **Gradients and the zero init.** Let $G=\partial L/\partial y$ for $y = xW+b+\frac{\alpha}{r}xAB$
  (batch of rows $X$). Then
  $$\frac{\partial L}{\partial B}=\frac{\alpha}{r}(XA)^\top G,\qquad
    \frac{\partial L}{\partial A}=\frac{\alpha}{r}X^\top G\,B^\top,\qquad
    \frac{\partial L}{\partial W}\ \text{not computed (frozen)}.$$
  At step 0, $B=0$, so $\partial L/\partial A = 0$, but $\partial L/\partial B\neq0$ because $A$ is
  random. So $B$ moves first and then $A$ starts to learn. If *both* were zero, both gradients would
  be zero forever (a saddle point). That is why exactly one factor is zero-initialized.
- **Compute.** The LoRA path costs $O(r(d_{in}+d_{out}))$ multiply-adds per token, versus
  $O(d_{in}d_{out})$ for the frozen path. The memory win is mostly in the optimizer: AdamW keeps two
  moment estimates per *trainable* parameter, so only $A,B$ need them.
- **Merging for inference** (general knowledge, not in the book). After training you can compute
  $W' = W + \frac{\alpha}{r}AB$ once and drop the extra path, so inference has no added latency.
  Keeping them separate (as the book does) instead lets you swap adapters per customer.
- **Relatives.** [[galore]] also uses low rank, but on the *gradient* during (pre)training while
  the full weights are updated. QLoRA, DoRA and other variants exist (general knowledge). LoRA is the
  flagship of [[parameter-efficient-fine-tuning]].

## Connections
- Builds on: [[matrix-multiplication]], [[basis-and-rank]], [[low-rank-approximation]], [[linear-layer]], [[linear-mapping]], [[backpropagation]], [[gradient-descent]], [[weight-initialization]], [[layer-freezing]], [[fine-tuning]]
- Leads to: [[lora-rank-and-alpha]], [[parameter-efficient-fine-tuning]], [[galore]]
- Used by: [[classification-fine-tuning]] (Appendix E spam classifier), [[instruction-fine-tuning]] (Exercise 7.4), [[gpt-2]] adaptation
- Related math: [[singular-value-decomposition]], [[vector-space]], [[parameter-count-and-memory]], [[adamw-optimizer]], [[overfitting]]
- People: [[edward-j-hu]], [[sebastian-raschka]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix E intro and §E.1, book pp. 322–324 (PDF pp. 344–346): definition, $W+\Delta W$, $\Delta W\approx AB$, distributive law, Figure E.1, Hu et al. citation (p. 323).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` end of §E.1 on book p. 324 (the page shared with the start of §E.2): $x(W+AB)=xW+xAB$ and the storage/scalability benefit; §E.2, book pp. 324–325: the spam dataset setup.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix B, ch. 5 references, book p. 294: GaLore (Zhao et al. 2024, arXiv 2403.03507), cited only as "replace AdamW with GaLoreAdamW"; its link to LoRA is this wiki's note.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 328–336 (PDF pp. 350–358): Listings E.5–E.7, Figures E.2–E.5, parameter counts (p. 332), accuracies (pp. 334, 336), timing (pp. 334–335).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §7.8, Exercise 7.4, book p. 247; Appendix C, Exercise 7.4 solution, book p. 310.
- Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models", arXiv 2106.09685 (cited on book p. 323). The paper's notation, intrinsic-rank claim, and merging trick are from the paper / general knowledge, not the book. The worked $3\times3$ example, gradient formulas, and memory figures are this wiki's own derivations.
