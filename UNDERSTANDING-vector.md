# Understanding: Vector
_Tutor checklist — items get ✅ only when the human has explained them back correctly._

## 1. The problem
- [x] What a vector *is* by the membership test, not by appearance (add + scale → same kind of thing) ✅ stated it himself
- [x] Why we need such an abstract definition — the root force (so one toolkit works for arrows, lists, polynomials, images…) ✅ "a tool to manipulate many kinds of data"
- [x] What the alternative views were (vector = arrow only? vector = list of numbers only?) and why they're too narrow ✅ rejected arrow def via the person example

## 2. The solution
- [x] The mechanics: a vector in R^n, and the two operations done component-wise (addition, scalar multiplication) ✅ quiz Q1 correct
- [x] Why component-wise / why closure matters — the design decision (result must stay in the same space) ✅ defined closure as "can't escape the set"
- [x] Edge cases: the zero vector, scaling by 0 and by negatives; "objects that look different but are the same vector" ✅ proved 0·a=0 forces the zero vector

## 3. Broader context
- [x] Why this matters for ML — data points, images, embeddings are all vectors in R^n ✅ averaging images = add + scale-by-0.5, closure keeps it an image
- [x] What it leads to / unlocks downstream (vector-space, linear-independence, basis, all of ML) ✅ subspace (y=x) discovered live
- [x] What breaks if you forget the rules (e.g. broadcasting silently doing the wrong operation) ✅ saw the broadcasting trap
