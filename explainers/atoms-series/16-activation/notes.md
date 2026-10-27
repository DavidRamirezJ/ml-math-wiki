# Notes — 16 Activation function
- Duration: 11.9 min (final.mp4, 1080p30). Narration 1641 words / 33 blocks (measure.py 11.59 min, OK).
- Coverage: two linear layers collapse, real 2×2 numbers (2,0) both ways (C1); nonlinearity, activation,
  ReLU = max(0, z) with 5 worked values; ReLU between layers gives (2,1) ≠ (2,0) (C2); sigmoid (0,1),
  tanh (−1,1), GELU = zΦ(z), GELU(1)≈0.84, GELU(−1)≈−0.16, GPT-2 uses GELU (C3); x² on [−2,2] from
  4 + (−3)ReLU(x+2) + 2ReLU(x+1) + 2ReLU(x) + 2ReLU(x−1), max gap 0.25 → 0.0625 with 8 pieces (C4);
  ReLU slope 0/1, PyTorch uses 0 at the corner, dead unit (w=1, b=−10, x∈[0,5]) (C5); MLP from
  Raschka §A.5, GPT-2 FFN Linear→GELU→Linear, 0 parameters, axiom A + a bend (C6).
- New terms: activation function, ReLU, nonlinearity. Reminders: matrix multiplication, derivative.
- QA: 2 draft layout warnings fixed (tanh label overlap, off-frame axis label); contact sheets → moved
  sigmoid formula off the heading, GELU labels off the axis, clipped partial ReLU sums to the plot,
  tightened 7-bullet summary. 0 layout warnings in final. No AUDIO PROBLEM.
- Simplified/uncertain: universal approximation stated qualitatively ("enough ReLU units, any continuous
  curve on an interval"); "ReLU replaced sigmoid/tanh" given as one reason (gradient flow), per wiki.
