# Understanding: Linear Independence
_Tutor checklist — items get ✅ only when the human has explained them back correctly._

## 1. The problem
- [ ] The intuitive question: does a new arrow add a genuinely NEW direction, or is it redundant?
- [ ] Why "redundant" needs a precise definition (can't eyeball it in ℝⁿ)
- [ ] Connection back to span: redundant vector = adds nothing to the span

## 2. The solution (the definition + the test)
- [ ] Linear combination: v = Σ λᵢ·xᵢ (read the notation)
- [ ] The formal definition (Def 2.12): independent ⇔ ONLY the trivial solution (all λ=0) gives Σλᵢxᵢ = 0
- [ ] Why "= 0 forces all λ = 0" captures "no redundancy" (the logical link)
- [ ] Dependent = some non-trivial combo hits 0 ⇒ one vector is a combo of the others
- [ ] Quick facts: contains 0 → dependent; multiple of another → dependent; m > k → dependent
- [ ] The practical test: stack as columns, row-reduce, every column a pivot ⇔ independent

## 3. Broader context
- [ ] Independence is the gatekeeper for BASIS (minimal independent spanning set)
- [ ] RANK = number of independent columns = dimension of reachable output (ties to matrix session)
- [ ] ML payoff: redundant features, dimensionality reduction (PCA), why collinearity is a problem
