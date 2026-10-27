# Study Session — Backpropagation

- **Date:** 2026-10-15
- **Skill:** /explain (interactive HTML), no formal /teach
- **Source material:** `backpropagation`, `chain-rule`, `relu`, `gradient-descent`, `multilayer-perceptron`; follows `study-log/2026-10-13-mlp-cnn-activation.md`
- **Goal:** Understand backprop — where ∂L/∂w comes from (open item from 13 Oct).

## What we covered
1. Backprop is only step 3 of the training loop: it computes gradients, it does not change weights.
2. Intuition: factory blame passed backward; gears = chain rule (multiply local rates).
3. Tiny net x → z = w1·x + b1 → h = ReLU(z) → ŷ = w2·h → L = (ŷ − t)². Worked numbers (x=2, t=3, w1=0.5, b1=0, w2=1): z=1, h=1, ŷ=1, L=4; ∂L/∂ŷ=−4, ∂L/∂w2=−4, ∂L/∂h=−4, ∂L/∂z=−4, ∂L/∂w1=−8, ∂L/∂b1=−4. Update with η=0.05 → L = 0.36.
4. Nudge test: Δw1 = 0.01 ripples to ΔL ≈ −0.08, so ΔL/Δw1 ≈ −8.
5. Reuse trick: ∂L/∂z is shared by w1 and b1 → all gradients for ~one extra pass.
6. ReLU slope as a 0/1 gate (dead ReLU when z < 0).
7. Matrix form: ∂L/∂W = δ·xᵀ, ∂L/∂b = δ, δ_prev = (Wᵀδ) ⊙ σ′(z).

## Key takeaways (in the human's words)
- (Por llenar.)

## Mastery check
- [ ] Explain in one sentence what backprop computes and what it does NOT do
- [ ] Do the 5 backward steps of the tiny net by hand
- [ ] Explain why ReLU with z < 0 gives zero gradient
- [ ] Explain the reuse trick (why backward, not forward)
- [ ] Read the matrix form δ_prev = (Wᵀδ) ⊙ σ′(z)

## Open questions / revisit next
- Step 6 of the foundations: overfitting.
- Backprop in a CNN (shared filter sums blame from every position).
- Vanishing gradients and shortcut connections (wiki `backpropagation` already has the LLM-book numbers).

## Spaced review
1. Redo the tiny net with x = 1, t = 2, w1 = 1, b1 = 0, w2 = 1. What is ∂L/∂w1?
2. Why does training need more memory than inference?
3. ∂L/∂w = +3, η = 0.1. Does w go up or down, and by how much?

## Artifacts
- `explainers/2026-10-15_backpropagation.html` (English, vocab helper): step-through forward/backward animation with sliders, nudge ripple, live training chart, matrix form, 5-question quiz.
