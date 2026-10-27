# Notes — 04 Linear mapping
- Duration: 12.3 min, 1080p30, subtitles + chapters. Narration 1,789 words / 36 blocks (measure.py OK, 11.9 min).
- Coverage: wiki `linear-mapping` (MML §2.7, Defs 2.15, 2.19), `affine-space` (MML §2.8, Def 2.26); Raschka §4.3 p. 108
  (768 → 3072 → 768 block), App. A (Linear = weight matrix + bias). Definition + two-rule form, f(0)=0; grid picture
  (generic map, stretch, rotation, shear, projection); non-examples x+1 (affine, f(0)=1, 6≠7) and x² (4≠2, 36≠18) + graphs;
  theorem: columns = f(e1), f(e2), worked (3,2) → (4,5) and rotation; composition = product (P·S, (1,2) → (3,0));
  kernel / image of the projection, dim ker + dim image = dim input; ML: Wx + b is affine, stacked linear layers collapse,
  need activation (video 16); axiom A.
- New terms (owner): linear mapping, kernel (null space), image (range). Reminder: basis.
- Demo: NumPy matrix from images of e1, e2; linear rule check; shift/square counter-examples; composition; kernel; W2·W1. Real output.
- QA log: draft → 1 overlap warning (activation box vs source label) + visual fixes (image card covering heading,
  code panel too small, label positions). Final: 0 layout warnings, no AUDIO PROBLEM.
- Note: "collapse into one single map" for 768→3072→768 without activation is the wiki's own argument (not stated in the book).
