# Storyboard — Norm (ML Math Atoms · Video 6 of 16)

Source: `raw/mml-book.pdf` §3.1 (Def. 3.1), §3.3 (Def. 3.6); LLM book §D.3 (gradient clipping); wiki `norm`, `metric`.
Axiom: **B — Inner product** (the L2 norm is sqrt(<x,x>)); L1 and L-infinity are norms not induced by any inner product.

## Hook
How big is an error? How far apart are two words? Is a training step too large? One number: the norm.

## Prerequisites
Video 1 (vector), video 5 (inner product).

## Chapters
| # | Scene | Learning goal | Main visual |
|---|---|---|---|
| 0 | C0_Intro | why size/distance matter | questions → ‖x‖ |
| 1 | C1_Length | ‖x‖ = sqrt(x·x); (3,4)→5; (2,3,6)→7 | right triangle, squares 4+9+36=49 |
| 2 | C2_OtherNorms | L1 (taxi 7), L2 (5), L∞ (4); unit circles | taxi path, circle/diamond/square |
| 3 | C3_Rules | positive, ‖cx‖=|c|‖x‖, triangle inequality (√50≈7.07≤10) | live slider, tip-to-tail triangle |
| 4 | C4_Unit | unit vectors, normalize (3,4)→(0.6,0.8); zero vector; cosine similarity | rotating unit arrow |
| 5 | C5_Distance | d(x,y)=‖x−y‖; (1,1)-(4,5)=5; √22≈4.69; metric rules | difference arrow, triangle |
| 6 | C6_ML | L2/L1 regularization, gradient clipping (6,8)→(3,4), nearest neighbour | diamond corners, clip arc, kNN plane |
| 7 | C7_Practice | (−5,12): L2 13, L1 17; distance (2,7)-(5,3)=5, normalized (−0.6,0.8) | |
| 8 | C7_Code | np.linalg.norm, ord=1/2/inf | code + real terminal |
| 9 | C8_Summary / C9_Credits | summary, next: derivative | |

## Terms (owner = this video)
- norm, unit vector, distance (term cards). Reminders: dot product, cosine similarity (video 5).

## Colour ledger
x, u → BLUE · v → GREEN · result/highlight → YELLOW · scalars c, λ, weights → PURPLE · components/guide lines → ORANGE ·
L1 = ORANGE (taxi path), L2 = BLUE, L∞ = GREEN · warning → RED.
