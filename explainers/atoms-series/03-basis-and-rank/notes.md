# Notes — 03 Basis & rank
- Duration: 12.8 min, 1080p30, subtitles + chapters. Narration 1,810 words / 38 blocks (measure.py OK, 12.4 min).
- Coverage: wiki `linear-independence` (MML §2.5, Def 2.12), `basis-and-rank` (MML §2.6, Defs 2.13–2.14; Raschka App. E pp. 322–323).
  Hidden redundancy c = a + b; independence definition + zero test; worked: (2,1),(1,2),(3,3) dependent with weights 1,1,−1;
  (2,1),(1,2) independent by solving; basis; coordinates of p=(1,3) in e-basis (1,3) and in b1=(1,1), b2=(−1,1) → (2,1);
  dimension (+ "dimension ≠ number of entries"); rank with 3×3 rank-1 (line) and rank-2 (plane) examples (pseudo-3-D drawing),
  identity rank 3, full rank, row rank = column rank; toy m/cm/kg table rank 2; LoRA mention (768×768 vs rank ≤ 16).
- New terms (owner): linear independence, basis, dimension, rank. Reminder: span. "column space" said in words only (image = video 4).
- Demo: NumPy matrix_rank + solve — real output.
- QA log: draft 1 → 1 layout warning (off-frame text in basis6) fixed; table headers overlapping (h_buff) fixed. Final: 0 warnings,
  no AUDIO PROBLEM. Simplification: 3-D column spaces drawn with a fixed oblique 2-D projection (not a 3-D scene).
