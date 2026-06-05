---
title: Derivative
type: concept
tags: [calculus, differentiation]
difficulty: beginner
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
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

## Connections
- Builds on: [[vector-space|functions and limits]]
- Leads to: [[partial-derivative]], [[gradient]], [[taylor-series]], [[chain-rule]]
- Used by: gradient descent (Ch. 7), [[backpropagation]]
- People: —

## Sources
- `raw/mml-book.pdf` §5.1 (Defs 5.1–5.2, rules 5.29–5.32), printed pp. 141–146 (PDF pp. 147–152).
