# Storyboard — Gradient (ML Math Atoms · Video 8 of 16)

Source: `raw/mml-book.pdf` §5.2 (Def. 5.5, eq. 5.40); LLM book §A.4; wiki `partial-derivative`, `gradient`.
Axiom: **C — Local linearity** in many dimensions, f(x+h) ≈ f(x) + ∇f·h (C written with the inner product, B).

## Chapters
| # | Scene | Learning goal | Main visual |
|---|---|---|---|
| 0 | C0_Intro | one slope per weight → ∇L | weights → partials → ∇L |
| 1 | C1_Map | two inputs = heights; contour map of x²+y² (levels 5…25) | pipeline, side view, contour circles |
| 2 | C2_Partial | freeze the others: x²+3y at (1,2) → 2 and 3 | parabola contours, slice plot |
| 3 | C3_Gradient | ∇f = (2,3); bowl ∇f(3,4) = (6,8), length 10; arrow field | contour map + arrows (¼ length) |
| 4 | C4_Linear | f(x+h) ≈ f(x)+∇f·h; 7.5 vs 7.51; 24.8 vs 24.82 | formulas + step on map |
| 5 | C5_Steepest | slope along u = ‖∇f‖cosθ; max along ∇f, 0 along contour, −∇f downhill; GD preview | rotating direction u |
| 6 | C6_ML | ∇L has one entry per weight (GPT-2 small ≈124M); .grad same shape; toy step 25 → 16 | grids, step on bowl |
| 7 | C7_Practice | ∇(3x+y²) at (2,1) = (3,2); bowl at (0,5): (0,10) | |
| 8 | C7_Code | numeric gradient vs exact | code + real terminal |
| 9 | C8_Summary / C9_Credits | next: chain rule | |

## Terms
Owner: partial derivative, gradient. Reminders: derivative, local linear approximation (video 7), dot product (video 5), norm (video 6).

## Colour ledger
∂/∂x → ORANGE · ∂/∂y → GREEN · ∇f / highlight → YELLOW · step h, contours → BLUE · weights, slopes → PURPLE · warnings/downhill → RED.
