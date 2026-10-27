# Storyboard — Chain rule (ML Math Atoms · Video 9 of 16)

Source: `raw/mml-book.pdf` §5.1 (5.32), §5.2 (5.48), §5.3 (Def. 5.6); LLM book §A.4 (p. 263); wiki `chain-rule`, `jacobian`.
Axiom: derived from **C — Local linearity**, applied twice.

## Chapters
| # | Scene | Learning goal | Main visual |
|---|---|---|---|
| 0 | C0_Intro | a network is a chain; slopes multiply | layer pipeline |
| 1 | C1_Composition | g(f(x)); 3x+1 then u²: 1 → 4 → 16; order matters; gears 3×2=6; dy/dx = dy/du·du/dx | machines, rotating gears |
| 2 | C2_Derive | local linearity twice → g'(f(x))·f'(x) | multiplier pipeline |
| 3 | C3_Example | (3x+1)² at 1: 8×3 = 24; mistake → 6; check 18x+6, numeric | pipeline, plot with tangent |
| 4 | C4_Long | a=2x, b=a², c=5b: 2×4×5 = 40; check 20x² | 3-box pipeline |
| 5 | C5_Jacobian | J = matrix of partials; F=(x1², x1x2) → [[2,0],[2,1]]; J of Ax = A; J_G J_F = [10, 4] checked | matrices |
| 6 | C6_ML | backprop preview: multiply from the end 1 → 5 → 20 → 40; reuse | backward arrows |
| 7 | C7_Practice | (2x−1)³ at 1 → 6; b=3a+1, a=x² at 2 → 12 | |
| 8 | C7_Code | numeric checks, cumprod backward, J_G @ J_F | code + real terminal |
| 9 | C8_Summary / C9_Credits | next: exp & log | |

## Terms
Owner: composition, chain rule, Jacobian. Reminders: local linear approximation (07), partial derivative (08), matrix multiplication (02).

## Colour ledger
input x → BLUE · inner function / inner slope → ORANGE · outer function / slope → GREEN · result → YELLOW · layers → PURPLE · backward pass / mistakes → RED.
