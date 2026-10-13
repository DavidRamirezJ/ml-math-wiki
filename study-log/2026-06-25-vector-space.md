# Study Session — Vector Space

- **Date:** 2026-06-25 (completed across sittings through 2026-06-30)
- **Skill:** /teach
- **Source material:** `vector-space` (wiki concept page), `vector`, `matrix`, MML Ch.2 §2.4 (Definition 2.9)
- **Goal:** Master the third atomic concept — the *stage* every vector and matrix lives on. From "what a single vector is" to "the whole space of vectors, why it's closed, and how matrices move between spaces."

## What we covered
- **Vector space = a SET of vectors closed under add & scale.** The concrete test: take any two, add them; take any one, scale it — do the results stay inside? If yes, it's a space.
- **Closure** is THE big idea: "you can't operate your way out." It's a property of the *set*, not of a single vector. It gives the certainty that add/scale always returns the same *kind* of object.
- **Why closure matters:** it lets you CHAIN operations. A deep net is W₃·W₂·W₁·x — every output must be a valid vector the next machine can eat. Closure guarantees that.
- **Span** = the set of all *vectors* reachable from starting arrows (NOT a set of numbers — the numbers λ,i,j are just the dials). `span{a} = { λ·a : λ ∈ ℝ }` = a line. `span{a,b}` = a plane. In ℝ³: 1 arrow→line, 2→plane, 3 independent→whole room. The "dimension" = how many genuinely-different directions.
- **Subspace** = a subset that is itself closed under add & scale → therefore must contain 0. Key chain (cause→effect): closure lets you scale by λ=0, and 0·v = 0, so 0 is forced into the set. A set missing 0 was never closed. Shifted-off-origin sets ("affine") are NOT subspaces — and are NOT spans.
- **Every span IS a subspace** (set all coefficients to 0 → 0 is always reachable → contains origin + closed).
- **ML payoff / connection to matrices:** a data point with n features is a VECTOR in ℝⁿ (the space = the stage). A matrix is a machine **W : ℝⁿ → ℝᵐ** (n in = columns, m out = rows). The span of its columns = its reachable output.
- **Notation fluency (new goal this session):** learned to READ and decode ∈ ("belongs to"), ℝ / ℝⁿ, λ (scalar), ∀ ("for all"), ⊆ ("subset of"), and set-builder `{ recipe : condition }` = "the set of all … such that …".

## Key takeaways (in the human's words)
- "no because it can reach all the plane" — discovered closure unprompted.
- "closure gives the certainty that when we add/scale a vector, the output is a vector of the same kind."
- "it is a big line, doesn't change direction, this happens only with one vector" — span of one vector = a line, because scaling can't turn.
- "including 0, since 0·v = 0, the set is forced to contain 0" — subspace origin rule, correct cause→effect direction.
- "you plug in 0 and land on 0" — why every span contains the origin.
- "n to m" — matrix maps ℝⁿ → ℝᵐ (after correcting a flipped direction).

## Mastery check
- [x] What a vector space is + the concrete add/scale test
- [x] Closure — can't operate your way out; property of the SET
- [x] Span = set of reachable VECTORS; line vs plane; dimension ladder
- [x] Subspace — closed on its own, forced to contain 0 (closure → 0, right direction)
- [x] Every span is a subspace
- [x] Reading notation: ∈, ℝⁿ, λ, ∀, ⊆, set-builder
- [x] ML payoff: data = vector in ℝⁿ; matrix W : ℝⁿ → ℝᵐ; columns' span = output reach

## Notes for the tutor (next session)
- **Recurring trap:** "span = the numbers" instead of "span = the reachable vectors." Resurfaced in the final teach-back. Re-drill cold next time.
- **Word precision under load:** swapped "subspace" for "vector"/"space" several times in free recall, and flipped the matrix direction (said "m to n"). Open-recall exposed these; MCQ would have hidden them. Keep preferring open recall.
- **The matrix direction** ℝⁿ → ℝᵐ needed a concrete [height,weight,age]→[2 scores] example before it clicked. Use concrete numbers when letters get slippery.
- New standing preference (saved to skill + memory): SHOW formal notation and decode it symbol-by-symbol — the learner is building syntax-reading fluency.

## Open questions / revisit next
- **Linear independence** — the exact rule for "does this new arrow add a genuinely new direction, or is it redundant?" (the "make arrow 3 dependent" toggle in the 3D sim).
- **Basis & rank** — a minimal set of independent arrows that spans the space; rank = # independent columns of a matrix = dimension of its reachable output. This makes "collision / invertible" from the matrix session exact and countable.
- The bias term: a layer is W·x + b (affine, not purely linear) — still flagged from the matrix session.

## Spaced review (test yourself later — don't re-read, recall)
- [ ] What's the concrete two-action test for "is this set a vector space?"
- [ ] Why must every subspace contain 0? Walk the chain: closure → λ=0 → therefore 0. (Direction matters!)
- [ ] Is span a set of numbers or a set of vectors? Read `span{a} = { λ·a : λ ∈ ℝ }` aloud.
- [ ] In ℝ³ I give you 3 arrows but the span is only 2-D. What happened to the third arrow?
- [ ] A matrix is W : ℝ? → ℝ?. Which letter is the input (columns), which is output (rows)?

## Artifacts
- `teach-board.html` (live animated board — 5 scenes: add+scale, closure, span-two, span-one, subspace)
- `span-3d-sim.html` (interactive rotatable 3D: 1/2/3 arrows → line/plane/space, + "shift off origin" and "dependent arrow" toggles)
- `UNDERSTANDING-vector-space.md` (tutor checklist — all items ✅)
