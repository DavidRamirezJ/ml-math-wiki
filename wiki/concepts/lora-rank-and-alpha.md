---
title: LoRA Rank r and Scaling Factor alpha
type: concept
tags: [llm, fine-tuning, hyperparameters, linear-algebra]
difficulty: intermediate
sources: [llm-appE-parameter-efficient-fine-tuning-with-lora]
created: 2026-10-13
updated: 2026-10-13
---

# LoRA Rank r and Scaling Factor alpha

> **One-liner:** In LoRA, the rank $r$ decides how big (and how flexible) the add-on matrices are, and alpha decides how loudly that add-on speaks compared with the frozen model.

## 🟢 Intuition (Dummy)
Think of [[low-rank-adaptation|LoRA]] as adding a small "correction channel" next to each frozen
layer.

- **Rank $r$** = how many lanes that channel has. More lanes means it can express more kinds of
  corrections, but it costs more trainable numbers. Fewer lanes is cheaper but less flexible. The
  book calls this the balance between **adaptability** and **efficiency** (book p. 329).
- **Alpha $\alpha$** = the volume knob on the channel's output. Turn it up and the correction has a
  bigger effect on the final output; turn it down and it whispers (book p. 329).

The book's rule of thumb: **$r = 16$ and $\alpha = 16$ are good defaults**. People often raise the
rank, and **alpha is usually half, double, or equal to the rank** (book p. 332).

## 🟡 Mechanics (Practitioner)

### Where $r$ and $\alpha$ appear
The LoRA layer's forward pass (Listing E.5, book p. 329):

$$\text{LoRA}(x) = \frac{\alpha}{r}\,\big(x\,A\,B\big),\qquad A\in\mathbb R^{d_{in}\times r},\quad B\in\mathbb R^{r\times d_{out}}$$

Symbol by symbol:
- $x$ = the layer input (last dimension $d_{in}$).
- $A$ = trainable matrix with $d_{in}$ rows and $r$ columns; $B$ = trainable matrix with $r$ rows and
  $d_{out}$ columns. $\mathbb R^{m\times n}$ means "an $m$-by-$n$ table of real numbers".
- $r$ = the **rank** = the shared inner dimension of $A$ and $B$. The input is squeezed to $r$ numbers
  ($xA$) and then expanded back to $d_{out}$ ($\cdot B$).
- $\alpha$ = the **scaling factor**.
- $\frac{\alpha}{r}$ = the multiplier actually applied to the LoRA output.

In code: `x = (self.alpha / self.rank) * (x @ self.A @ self.B)`.

### What $r$ controls: size and maximum rank
1. **Parameter count** for one layer is $r(d_{in}+d_{out})$, so it grows **linearly** in $r$.
2. **Expressiveness**: $\operatorname{rank}(AB)\le r$, so the update can use at most $r$ independent
   directions ([[basis-and-rank]]).

For the book's GPT-2 small spam classifier, every Linear layer gets LoRA, and
$\sum(d_{in}+d_{out})$ over all of them is $166{,}658$. So the total is $166{,}658\cdot r$.
The book prints the $r=16$ row (book p. 332); the other rows are my arithmetic:

| $r$ | LoRA params | share of 124,441,346 |
|---|---|---|
| 4 | 666,632 | 0.54% |
| 8 | 1,333,264 | 1.07% |
| **16 (book)** | **2,666,528** | **2.14%** |
| 32 | 5,333,056 | 4.29% |
| 64 | 10,666,112 | 8.57% |

Check of the 166,658: per block $4\cdot1536 + 2\cdot3840 = 13{,}824$; $\times12 = 165{,}888$; plus
`out_head` $768+2 = 770$; total $166{,}658$.

### What $\alpha$ controls: the strength of the update
The effective weight of a LoRA-wrapped layer is

$$W_{\text{eff}} = W + \frac{\alpha}{r}\,AB$$

- $W$ = the frozen pretrained weight.
- $\frac{\alpha}{r}AB$ = the learned update, playing the role of $\Delta W$.

| $\alpha$ with $r=16$ | multiplier $\alpha/r$ | rule from the book |
|---|---|---|
| 8 | 0.5 | "half" the rank |
| 16 | 1.0 | "equal" (book default) |
| 32 | 2.0 | "double" the rank |

### Interaction with the zero init
$B$ starts at zero, so $\frac{\alpha}{r}AB = 0$ at the start **for any** $r$ and $\alpha$. The choice of
these two numbers does not change the starting model. The book confirms the untrained accuracies
(46.25% / 45.00% / 48.75%) are unchanged after inserting LoRA (book p. 334). The two hyperparameters
only matter once training starts moving $B$ (and then $A$).

## 🔴 Theory (Expert)
- **Why divide by $r$?** (Hu et al. 2021, not explained in the book.) Without the $1/r$, a larger rank
  sums more terms in $xAB = \sum_{k=1}^{r}(xA)_k B_{k,:}$, so the size of the update would grow with $r$,
  and you would need a new [[learning-rate]] every time you changed $r$. Dividing by $r$ keeps the
  update scale roughly comparable, so you can tune $\alpha$ once and then try different ranks.
- **$\alpha$ vs. learning rate.** With an adaptive optimizer like [[adamw-optimizer|AdamW]], the step
  on each entry of $A$ and $B$ is about $\eta$ in size no matter how big the gradient is. The change in
  $W_{\text{eff}}$ is then scaled by $\alpha/r$. So tuning $\alpha$ behaves roughly like tuning the
  learning rate of the LoRA path (the LoRA paper's remark). This is why the book's guideline ties
  $\alpha$ to $r$ rather than giving a free range.
- **Rank-stabilized scaling** (general knowledge): later work (rsLoRA, Kalajdzievski 2023) argues
  that $\alpha/\sqrt r$ keeps learning stable at large ranks better than $\alpha/r$. With $\alpha/r$, very
  high ranks can learn too slowly.
- **Choosing $r$ from a matrix-approximation view.** If the "ideal" full update $\Delta W^\star$ has
  singular values $\sigma_1\ge\sigma_2\ge\dots$, the best rank-$r$ error is $\sigma_{r+1}$
  ([[low-rank-approximation]], Eckart–Young). If the spectrum decays fast, a small $r$ loses little.
  The book does not measure this. It just reports that $r=16$ is enough for 96.64% validation accuracy.
- **Capacity and overfitting.** Larger $r$ means more [[model-capacity]] and more risk of
  [[overfitting]] on small data like the 1,045-example spam training set. Smaller $r$ acts as stronger
  [[regularization]].

## Connections
- Builds on: [[low-rank-adaptation]], [[basis-and-rank]], [[matrix-multiplication]], [[hyperparameter]]
- Leads to: [[parameter-efficient-fine-tuning]] design choices, [[parameter-count-and-memory]]
- Used by: [[classification-fine-tuning]] (Appendix E), [[instruction-fine-tuning]] (Exercise 7.4, also $r=\alpha=16$)
- Related: [[low-rank-approximation]], [[learning-rate]], [[adamw-optimizer]], [[model-capacity]], [[regularization]]
- People: [[edward-j-hu]]

## Sources
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §E.4, book pp. 328–332 (PDF pp. 350–354): Figure E.2 ("The inner dimension r is a hyperparameter", p. 328), Listing E.5 and the meaning of rank and alpha (p. 329), defaults $r=\alpha=16$ and "alpha half, double, or equal to the rank" (p. 332), 2,666,528 parameters (p. 332); zero-init sanity check (p. 334).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` Appendix C, Exercise 7.4 solution, book p. 310 (`rank=16, alpha=16`).
- The parameter-count table beyond $r=16$, the $1/r$ rationale, the $\alpha$-vs-learning-rate remark (Hu et al. 2021), and rsLoRA are this wiki's derivations or general knowledge, not from the book.
