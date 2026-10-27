"""Narration for: Chain rule — slopes multiply (ML Math Atoms · Video 9 of 16).
One block <= ~75 words. Source: MML book §5.1 (5.32), §5.2 (5.48), §5.3 (Def. 5.6); LLM book §A.4."""

N = {
"intro1": """A neural network is a long chain of functions. The input goes into the first
layer, its output goes into the second layer, and so on, until the loss at the end. To train
the network, we need the slope of the loss with respect to a weight deep inside that chain.
The tool for this is the chain rule. Its message is short: slopes multiply.""",

"intro2": """In this video you will learn to see a function inside a function as a pipeline.
You will see why the slopes multiply, using local linearity twice. You will work through
examples with numbers, meet the Jacobian, the matrix version, and preview backpropagation.
You need video seven on the derivative, and video eight on the gradient.""",

# ── C1: composition ──
"comp1": """Let's start with a pipeline of two machines. The first machine, f, takes x and
gives u. The second machine, g, takes u and gives y. Putting one function after another
like this is called composition. We write it as g of f of x.""",

"comp2": """Here is our example. The inner function is f of x equals three x plus one. The
outer function is g of u equals u squared. Put in x equals one. The inner machine gives
three plus one, which is four. The outer machine gives four squared, which is sixteen. So g
of f of one is sixteen.""",

"comp3": """Order matters. If we apply g first and then f, we get three times x squared,
plus one. At x equals one, that is four, not sixteen. So g of f and f of g are different
functions. Always read a composition from the inside out.""",

# ── C2: gears ──
"gear1": """Now, how fast does the output change? Think of three gears. When the first gear
turns once, the second gear turns three times. When the second gear turns once, the third
gear turns two times. How fast does the third gear turn compared with the first?""",

"gear2": """One turn of the first gear makes three turns of the second. Each of those makes
two turns of the third. So the third gear turns three times two, which is six times as fast.
The rates multiply. They do not add.""",

"gear3": """Functions work the same way. The slope of y with respect to x equals the slope of
y with respect to u, times the slope of u with respect to x. This is called the chain rule.
It looks like fractions cancelling, and that is a good way to remember it.""",

# ── C3: derive it ──
"der1": """Why is it true? We can derive it from local linearity, axiom C, used twice. Step
one: nudge x by a small step h. The inner function changes by about f prime of x times h.
Call that small change k.""",

"der2": """Step two: the outer function sees its input u change by k. Local linearity again:
g changes by about g prime of u times k. Now put in k. The output changes by g prime of u,
times f prime of x, times h.""",

"der3": """So the slope of the whole pipeline is g prime of u times f prime of x, where u is
f of x. That is the chain rule. Each machine, close up, just multiplies small changes by its
slope. Two multiplications in a row are one multiplication by the product.""",

# ── C4: worked example ──
"ex1": """Let's compute the slope of three x plus one, squared, at x equals one. The inner
slope: the derivative of three x plus one is three. The outer slope: the derivative of u
squared is two u. Now the important part. We evaluate it at the inner value, u equals four.
So the outer slope is eight.""",

"ex2": """Multiply: eight times three is twenty four. So the slope of the composition at x
equals one is twenty four. Watch out for a common mistake. Do not evaluate the outer slope
at x equals one. That would give two, and a wrong answer of six.""",

"ex3": """Let's check in two ways. First, expand the square: nine x squared, plus six x, plus
one. Its derivative is eighteen x plus six. At x equals one, that is twenty four. Second, a
numeric check with a tiny step also gives twenty four. All three answers agree.""",

# ── C5: longer chains ──
"long1": """Chains can be longer. Take three steps. First, a equals two x. Then b equals a
squared. Then c equals five b. Each step has its own local slope. At x equals one, a is two,
and the slopes are two, then two a, which is four, and then five.""",

"long2": """The chain rule simply multiplies all of them: two times four times five equals
forty. Check it directly: c equals five times two x, squared, which is twenty x squared. Its
derivative is forty x, and at x equals one that is forty. Correct again.""",

"long3": """It does not matter how long the chain is. Ten steps or a hundred steps, you
multiply the local slopes along the way. This is exactly what happens inside a deep neural
network with many layers.""",

# ── C6: Jacobian ──
"jac1": """What if a machine has several inputs and several outputs? Then one slope is not
enough. We need the partial derivative of every output with respect to every input. Put them
in a grid: one row for each output, one column for each input. This matrix is called the
Jacobian.""",

"jac2": """Example: F takes x one and x two, and outputs x one squared, and x one times x two.
The first row holds the partial derivatives of x one squared: two x one, and zero. The
second row holds those of x one times x two: x two, and x one. At the point one, two, the
Jacobian is two, zero, two, one.""",

"jac3": """A nice fact: for a linear mapping, x goes to A times x, the Jacobian is just the
matrix A. So the Jacobian is the local linear mapping of the function. Close up, every
smooth function acts like a matrix.""",

"jac4": """Now the chain rule for many dimensions. Chained functions have chained Jacobians,
and the Jacobians multiply, as matrices. Let G take u one and u two and give u one plus u
two squared. Its Jacobian at F of one, two, which is one, two, is the row one, four.""",

"jac5": """Multiply the row one, four by the Jacobian of F. The result is the row ten, four.
Check directly. G of F is x one squared plus x one squared times x two squared. Its partial
derivatives at one, two are ten and four. The matrix product gives the right answer.""",

# ── C7: in ML ──
"ml1": """Now machine learning. A network is a chain: input, layer one, layer two, and so on,
then the loss. The book on building large language models describes backpropagation as an
implementation of the chain rule for neural networks. Backpropagation is a molecule built
from this atom, so here we only preview it.""",

"ml2": """The key trick is the direction. Start at the end, where the slope of the loss with
respect to itself is one. Walk backward, and multiply by each local slope. In our three step
chain: one, then times five is five, then times four is twenty, then times two is forty.""",

"ml3": """Each product is computed once and reused for everything further back. That makes
the backward pass fast, even with millions of weights. Libraries like PyTorch do this
automatically. You only need to remember the idea: slopes multiply, from the end
backward.""",

# ── C8: practice ──
"prac1": """Now it is your turn. Pause the video and try this. What is the slope of two x
minus one, cubed, at x equals one? Find the inner slope, then the outer slope at the inner
value, and multiply.""",

"prac2": """Here is the answer. The inner function two x minus one has slope two. At x equals
one, the inner value is one. The outer function u cubed has slope three u squared, which is
three at u equals one. So the slope is three times two, which is six.""",

"prac3": """One more question. Let a equal x squared, and b equal three a plus one. What is
the slope of b with respect to x at x equals two? Pause and try.""",

"prac4": """The slope of a with respect to x is two x, which is four at x equals two. The
slope of b with respect to a is three. Multiply: three times four is twelve. Check: b is
three x squared plus one, with derivative six x, which is twelve at two.""",

# ── C8: code ──
"code1": """Let's check this in Python. We write a numeric slope function. Then we compare the
chain rule with a numeric slope for three x plus one, squared. We multiply the local slopes
of the three step chain from the end backward. And we multiply two Jacobians with NumPy.""",

"code2": """This output is real; the script ran while the video was made. Eight times three
is twenty four, and the numeric slope agrees. The backward products are five, twenty, and
forty. And the product of the Jacobians is ten, four, the same as the numeric check.""",

# ── C9: summary ──
"sum1": """In summary. A composition puts one function after another, g of f of x. The chain
rule says the slopes multiply: the outer slope, at the inner value, times the inner slope. It
comes from local linearity used twice. Longer chains multiply more slopes.""",

"sum2": """With many inputs and outputs, the slopes form a Jacobian, and chained Jacobians
multiply as matrices. Backpropagation applies the chain rule from the end of the network
backward. Next, in video ten: the exponential and the logarithm, which turn products into
sums. See you there.""",
}
