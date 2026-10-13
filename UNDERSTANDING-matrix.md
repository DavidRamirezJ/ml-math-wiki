# Understanding: matrix
_Tutor checklist — items get ✅ only when the human has explained them back correctly._

## 1. The problem
- [x] A vector *stores* data — but learning means *transforming* data; matrix = the transformer
- [x] Why one object: REUSE (one rule, all vectors) + rule is plain numbers → storable & LEARNABLE
- [ ] What "linear" buys us (and what it forbids — no bending/warping)

## 2. The solution
- [x] A matrix as a machine: A·x produces a new vector — COLUMNS = input dim, ROWS = output dim
- [x] The columns view: A·x = scale each column by x's entries, then add them (only scale+add!)
- [x] Matrix × matrix = chaining two machines; INNER must match, OUTER survive ([a×b]·[c×d] needs b=c → [a×d])
- [x] Why AB ≠ BA: matrices are ACTIONS, order matters (rotate-then-flip ≠ flip-then-rotate)
- [x] Identity = do-nothing (I·x=x); Inverse = undo machine; NO inverse when inputs collide (info lost)
- [x] The dual role: same grid is a transformation (apply: x→Ax) AND a system of equations (solve: x=A⁻¹b)
- [x] COLLISION skill: inverse exists ⇔ no two inputs share an output (proven cold on P, Q, R)

## 3. Broader context
- [x] Why this matters in ML: a layer = a weight matrix W; W is the box of learnable numbers
- [x] Deep net = W₃·W₂·W₁·x = chaining machines; each layer's output must match next layer's input
- [x] Training = nudging the numbers inside every W (gradient = scale/step); shape mismatch breaks the chain
