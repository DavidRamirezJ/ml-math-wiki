# Understanding: Vector Space
_Tutor checklist — items get ✅ only when the human has explained them back correctly._

## 1. The problem
- [x] What "vector space" is trying to capture — why we need a word beyond "vector"
- [x] Why it existed — closure gives certainty: add/scale → same kind of object → lets you CHAIN operations (W₂·W₁·x)
- [x] The alternative we're rejecting: one theory (add+scale+closure) covers arrows, lists, polynomials, functions, images alike

## 2. The solution
- [x] What a vector space IS: a set + two operations (add, scale); the concrete test = add any two / scale any one → still inside?
- [x] CLOSURE — the single most important property ("you can't operate your way out"); property of the SET, not a single vector
- [x] The axioms at a working level (identity 0, inverse, distributivity) — make add/scale "behave like arithmetic"; closure is the load-bearing one
- [x] SUBSPACE: U⊆V closed on its own; closure FORCES 0 (λ=0→0·v=0), so missing 0 = not closed. Cause→effect direction nailed.
- [x] SPAN: span{a}={λ·a:λ∈ℝ}=a line (scaling can't turn); span{a,b}=plane. Can READ set-builder + ∈ notation.

## 3. Broader context
- [x] Why this matters for ML — a data point with n features is a VECTOR in ℝⁿ; the space is the stage
- [x] Connection back to MATRICES: a matrix maps space→space, ℝⁿ → ℝᵐ (n=in/columns, m=out/rows); columns span the reachable output
- [x] What breaks if closure fails — chained ops (W₂·W₁·x) need every output to be a valid input for the next machine
- [x] Leads to: linear-independence, basis-and-rank, dimension (next lessons)

## ✅ SESSION COMPLETE — all items mastered via teach-back 2026-06-30
