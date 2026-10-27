# Storyboard — Derivative (ML Math Atoms · Video 7 of 16)

Source: `raw/mml-book.pdf` §5.1 (Defs 5.1–5.2, rules 5.29–5.32), §5.8 (Taylor / linearization); wiki `derivative`, `taylor-series`.
Axiom: **C — Local linearity**: f(x+h) ≈ f(x) + f'(x)·h.

## Hook
A network nudges each weight; it needs to know how much the error changes → the derivative.

## Prerequisites
School algebra (functions, graphs, x²).

## Chapters
| # | Scene | Learning goal | Main visual |
|---|---|---|---|
| 0 | C0_Intro | why derivatives | weight → error → df/dx |
| 1 | C1_Average | rise/run; secant on x² from 3 to 4 = 7; shrink h → tangent | live secant (ValueTracker h) |
| 2 | C2_Limit | definition as a limit; 7, 6.1, 6.01 → 6; algebra gives 2x; never put h = 0 | table, algebra steps |
| 3 | C3_Local | zoom until straight; f(x+h) ≈ f(x)+f'(x)h; 9.6 vs 9.61; x³: 9.2 vs 9.261; step ↦ f'h is linear | live zoom window, tangent + error bar |
| 4 | C4_Rules | power, constant factor, sum; 3x²+5x → 6x+5, slope 11 at 1; chain rule teaser (video 9) | plot with tangent |
| 5 | C5_Flat | f' = 0 at min/max; x³ caution | sliding tangent coloured by sign |
| 6 | C6_ML | toy loss (2w−6)²: slope −16 at w=1, nudge 0.01 → −0.16 (true −0.1596); best w=3 | loss curve + tangent |
| 7 | C7_Practice | (x²)' at 5 = 10, 5.1² ≈ 26; (4x³−2x)' = 12x²−2 → 10 | |
| 8 | C7_Code | numeric ratio vs exact | code + real terminal |
| 9 | C8_Summary / C9_Credits | next: gradient | |

## Terms (owner = this video)
derivative, limit, local linear approximation (term cards). Reminder: linear mapping (video 4).

## Colour ledger
curve/input → BLUE · result/tangent/highlight → YELLOW · slope f'(x), weights → PURPLE · step h, secant, guide lines → ORANGE · error/negative slope → RED.
