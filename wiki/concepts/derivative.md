---
title: Derivative
type: concept
tags: [calculus, differentiation]
difficulty: beginner
sources: [mml-ch05-vector-calculus, llm-appA-introduction-to-pytorch, llm-ch04-implementing-a-gpt-model]
created: 2026-06-04
updated: 2026-10-13
---

# Derivative

> **One-liner:** The derivative tells you how fast a function is changing at a point — the steepness of its graph right there.

## 🟢 Intuition (Dummy)
Imagine driving up a hill. At any instant your speedometer-for-steepness would
say "the road is rising this fast right now." That instantaneous steepness is
the **derivative**. If you only knew where you started and where you ended you'd
get the *average* slope (rise over run). Zoom in closer and closer on a single
spot and the average slope settles down to one number — the slope of the line
that just kisses the curve there (the **tangent line**). That number is the
derivative. It points in the direction the function is climbing fastest.

**In LLMs.** Training a language model is a long hike downhill on the "loss landscape". At
every step the model asks, for each weight, "which way is downhill, and how steep?" — that is a
derivative (or, with many weights, a [[partial-derivative]]). The good news from the LLM book:
you do not need to remember your calculus in detail, because PyTorch's autograd computes
the derivatives for you; a high-level picture is enough (§A.4, book p. 263).

## 🟡 Mechanics (Practitioner)
Start with the **difference quotient** (Def 5.1), the slope of the secant line
between two points:
$$\frac{\delta y}{\delta x} = \frac{f(x+\delta x)-f(x)}{\delta x}.$$
Shrinking $\delta x$ to zero gives the **derivative** (Def 5.2):
$$\frac{df}{dx} = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}.$$

**Worked example (power rule).** For $f(x)=x^n$, expanding $(x+h)^n$ with the
binomial theorem and taking $h\to0$ leaves a single surviving term:
$$\frac{d}{dx}x^n = nx^{n-1}.$$

**The four rules you actually use** (5.29–5.32), with $f' = df/dx$:
$$
\begin{aligned}
\text{Product:}\quad &(fg)' = f'g + fg' \\
\text{Quotient:}\quad &\left(\tfrac{f}{g}\right)' = \frac{f'g - fg'}{g^2} \\
\text{Sum:}\quad &(f+g)' = f' + g' \\
\text{Chain:}\quad &\big(g(f(x))\big)' = g'(f(x))\,f'(x)
\end{aligned}
$$

**Chain-rule example.** $h(x)=(2x+1)^4$ with $f(x)=2x+1,\ g(f)=f^4$:
$$h'(x)=g'(f)f'(x)=4(2x+1)^3\cdot 2 = 8(2x+1)^3.$$

**ML usage:** every gradient-based learner needs derivatives. The derivative is
the 1-D building block; stacking it gives the [[gradient]] and [[jacobian]]
that drive gradient descent.

### In LLMs: the derivatives inside one neuron (LLM book §A.4, pp. 263–264)
Appendix A of the LLM book trains a one-input logistic-regression "neuron"
$u = w_1x_1,\ z = u + b,\ a = \sigma(z),\ L = L(a,y)$ with $x_1=1.1$, $w_1=2.2$, $b=0$, $y=1$
(Figure A.8). Three single-variable derivatives do the work:

1. **Sigmoid derivative** (the edge labelled $\frac{da}{dz}$):
   $$\sigma(z) = \frac{1}{1+e^{-z}},\qquad \frac{da}{dz} = \sigma(z)\,\big(1-\sigma(z)\big)$$
   - $\sigma$ = the [[sigmoid-function]]; $e^{-z}$ = "e to the minus $z$"; $\frac{da}{dz}$ = "how fast the output $a$ changes as $z$ changes".
   - Derivation with the quotient/chain rule: $\frac{d}{dz}(1+e^{-z})^{-1} = (1+e^{-z})^{-2}e^{-z} = \sigma(z)\cdot\frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$.
   - Numbers: $z = 2.42$, $a = \sigma(2.42) \approx 0.9183$, so $\frac{da}{dz} \approx 0.9183\times0.0817 \approx 0.0750$.
2. **Log derivative** inside [[binary-cross-entropy]]: with $y=1$, $L = -\ln a$, and $\frac{d}{da}(-\ln a) = -\frac{1}{a} \approx -1.0890$.
3. **Linear pieces:** $\frac{d}{dw_1}(w_1x_1) = x_1 = 1.1$, and in $z = u + b$ both $\frac{\partial z}{\partial u}$ and $\frac{\partial z}{\partial b}$ equal $1$.

Chaining them (the **chain rule** from the list above) gives $\frac{\partial L}{\partial b} \approx -1.0890\times0.0750 \approx -0.0817$
and $\frac{\partial L}{\partial w_1} \approx 1.1\times(-0.0817) \approx -0.0898$. PyTorch's `torch.autograd.grad`
prints exactly `tensor([-0.0898])` and `tensor([-0.0817])` (Listing A.3). Intermediate factors are computed
by hand; the book shows only the final gradients.

**Sanity check with a finite difference** (the derivative's definition, used as a test):
$\frac{L(w_1+h)-L(w_1)}{h}$ with $h = 10^{-4}$ reproduces $\approx -0.0898$ — the gradient-checking trick in the Theory tier.

## 🔴 Theory (Expert)
- $f$ is **differentiable** at $x$ iff the limit in Def 5.2 exists; this is
  strictly stronger than continuity. The book assumes differentiability
  throughout, noting that with sub-differentials many results extend to
  functions that are continuous but non-differentiable at isolated points
  (e.g. ReLU at $0$).
- The derivative is the best **local linear approximation**: $f(x)\approx
  f(x_0)+f'(x_0)(x-x_0)$, the degree-1 [[taylor-series|Taylor polynomial]].
- **Gradient checking.** Because the derivative is *defined* as a limit of
  difference quotients, you can verify a hand-coded analytic gradient against a
  finite-difference estimate with small $h$ (e.g. $10^{-4}$); the book flags a
  relative-error test $\sqrt{\tfrac{\sum_i(dh_i-df_i)^2}{\sum_i(dh_i+df_i)^2}}<10^{-6}$.
- The chain rule (5.32) is the seed of all of vector calculus: its multivariate
  form ([[chain-rule]]) becomes a product of Jacobians and ultimately
  [[backpropagation]].
- **In LLMs: derivatives of activations matter.** The sigmoid's derivative peaks at $\tfrac14$ (at $z=0$)
  and decays to $0$ as $|z|$ grows; multiplied across many layers this causes
  [[vanishing-and-exploding-gradients]]. Modern LLMs use [[relu]]/[[gelu]] (derivative $\approx 1$ for
  large positive inputs) plus [[shortcut-connections]] to keep derivatives from shrinking. ReLU's kink at
  $0$ is handled with a sub-derivative, as noted above.
- **Autograd does not do symbolic calculus.** PyTorch never forms a formula like $\sigma(z)(1-\sigma(z))$
  for the whole network; each primitive op stores its local derivative rule, and reverse-mode
  [[automatic-differentiation]] multiplies the numbers along the [[computational-graph]].


> **From [[llm-ch04-implementing-a-gpt-model]]:** the book compares activation functions by their derivatives (book p. 106). [[relu]] has derivative 0 for every $x<0$ and no derivative at the corner $x=0$. [[gelu]] is smooth, and its derivative is non-zero for almost all negative inputs; the only zero is at its minimum, near $x \approx -0.75$. So neurons with negative input still pass a small gradient.

## Connections
- Builds on: [[vector-space|functions and limits]]
- Leads to: [[partial-derivative]], [[gradient]], [[taylor-series]], [[chain-rule]]
- Used by: gradient descent (Ch. 7), [[backpropagation]], [[gradient-descent]], [[automatic-differentiation]]
- Related: [[sigmoid-function]], [[binary-cross-entropy]], [[logistic-regression]], [[computational-graph]], [[vanishing-and-exploding-gradients]]
- People: —

## Sources
- `raw/mml-book.pdf` §5.1 (Defs 5.1–5.2, rules 5.29–5.32), printed pp. 141–146 (PDF pp. 147–152).
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §A.4, book pp. 263–264 (Figure A.8 local derivatives incl. $da/dz$; sidebar "Partial derivatives and gradients"; Listing A.3 gradients). Sigmoid-derivative derivation and finite-difference check are standard, not in the book.
- `raw/Build_a_Large_Language_Model_(From_Scrat.pdf` §4.3, book p. 106 (GELU has a non-zero gradient for almost all negative inputs, except near $x \approx -0.75$; ReLU has a sharp corner at 0).
