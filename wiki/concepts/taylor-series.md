---
title: Taylor Series
type: concept
tags: [calculus, approximation, differentiation]
difficulty: intermediate
sources: [mml-ch05-vector-calculus]
created: 2026-06-04
updated: 2026-06-04
---

# Taylor Series

> **One-liner:** A Taylor series rebuilds a smooth function as an infinite sum of polynomial pieces, using the function's slopes (derivatives) at one point.

## 🟢 Intuition (Dummy)
You know a function's value at one spot, plus how steep it is there, plus how its
steepness is changing, and so on. With just that local information you can
*predict* the function nearby: start at the known value, add the slope times how
far you walk, then a curvature correction, then finer corrections. A **Taylor
series** is that recipe. Keep a few terms and you get a good approximation close
to your starting point; keep more terms and it stays accurate over a wider range.
It is how calculators and ML approximations replace a hard function with an easy
polynomial.

## 🟡 Mechanics (Practitioner)
**Taylor polynomial of degree $n$ (Def 5.3):** for $f:\mathbb R\to\mathbb R$ at $x_0$,
$$T_n(x)=\sum_{k=0}^{n}\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k.$$
**Taylor series (Def 5.4):** let $n\to\infty$ for a smooth $f\in C^\infty$:
$$T_\infty(x)=\sum_{k=0}^{\infty}\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k.$$
At $x_0=0$ this is the **Maclaurin series**. If $f=T_\infty$, $f$ is **analytic**.

**Worked example.** $f(x)=\sin x+\cos x$ at $x_0=0$ has derivatives cycling
$1,1,-1,-1,\dots$, giving
$$T_\infty(x)=1+x-\tfrac{1}{2!}x^2-\tfrac{1}{3!}x^3+\tfrac{1}{4!}x^4+\cdots=\cos x+\sin x.$$

**Multivariate Taylor series (Def 5.7).** With $\boldsymbol\delta:=\mathbf x-\mathbf x_0$,
$$f(\mathbf x)=\sum_{k=0}^{\infty}\frac{D_{\mathbf x}^k f(\mathbf x_0)}{k!}\,\boldsymbol\delta^k,$$
where $\boldsymbol\delta^k$ is a $k$-fold outer product (a $k$-th order tensor).
The low-order terms are the ones used in practice:
$$
\begin{aligned}
k=0:&\quad f(\mathbf x_0)\\
k=1:&\quad \nabla_{\mathbf x}f(\mathbf x_0)\,\boldsymbol\delta\\
k=2:&\quad \tfrac{1}{2}\,\boldsymbol\delta^\top H(\mathbf x_0)\,\boldsymbol\delta
\end{aligned}
$$
with $H$ the [[hessian]]. The first two give the **linearization** (5.148):
$$f(\mathbf x)\approx f(\mathbf x_0)+(\nabla_{\mathbf x}f)(\mathbf x_0)(\mathbf x-\mathbf x_0).$$

**ML usage:** first-order Taylor = local linear model behind gradient methods and
the extended Kalman filter; second-order Taylor (gradient + Hessian) underlies
Newton's method and the **Laplace approximation** of a posterior near its mode
(§5.9, Ch. 6).

## 🔴 Theory (Expert)
- A degree-$n$ Taylor polynomial exactly reproduces any polynomial of degree
  $\le n$ (all higher derivatives vanish). Example 5.15 expands
  $f(x,y)=x^2+2xy+y^3$ about $(1,2)$ and recovers it exactly at order 3.
- A Taylor series is a special **power series** $\sum a_k(x-c)^k$; convergence and
  whether $f=T_\infty$ (analyticity) are nontrivial — $C^\infty$ does not imply
  analytic (e.g. $e^{-1/x^2}$).
- The tensor $\boldsymbol\delta^k$ and $D^k_{\mathbf x}f$ make the multivariate
  series notationally heavy; in coordinates,
  $D^k_{\mathbf x}f(\mathbf x_0)\boldsymbol\delta^k=\sum_{i_1\dots i_k}D^k_{\mathbf x}f[i_1,\dots,i_k]\,\delta[i_1]\cdots\delta[i_k]$.
- Taylor expansion is the standard way to approximate intractable expectations
  $\mathbb E_{\mathbf x}[f(\mathbf x)]=\int f(\mathbf x)p(\mathbf x)\,d\mathbf x$
  when $p$ is Gaussian (§5.9).

## Connections
- Builds on: [[derivative]], [[gradient]], [[hessian]], [[partial-derivative]]
- Leads to: Newton's method & Laplace approximation (Ch. 7, Ch. 6), linearization
- Used by: extended Kalman filter, function approximation
- People: [[brook-taylor]]

## Sources
- `raw/mml-book.pdf` §5.1.1 (Defs 5.3–5.4), §5.8 (Defs 5.7–5.8, Example 5.15), §5.9, printed pp. 142–145, 165–170 (PDF pp. 148–151, 171–176).
