# Linear Independence — the redundancy question

> Does a new arrow add a genuinely NEW direction — or is it redundant?

## Redundant = already inside the span of the others

    a = East  = [1, 0]
    b = North = [0, 1]        a, b span the whole floor (R²)

    c = NE    = [1, 1] = 1·a + 1·b     ← built from a, b (scale, then add)

    Add c to the set  →  span STAYS the same  →  c adds nothing  →  REDUNDANT


## The contrast: DEPENDENT vs INDEPENDENT

    DEPENDENT set          {a, b, c}   — c is a leftover; span didn't grow
                           at least one arrow is buildable from the others

    INDEPENDENT set        {a, b}      — remove either one and the span shrinks
                           NObody is buildable from the others; all pull their weight


## Building toward the exact rule
   "redundant" said in words   =   "one arrow is a combination of the others"
   ...but we need this as an EQUATION we can test. Next.
