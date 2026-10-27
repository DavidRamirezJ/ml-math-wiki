"""Narration for: Activation function — the bend that makes networks deep (ML Math Atoms · Video 16 of 16).
One block <= ~75 words. Source: wiki activation-function, relu (Raschka §4.2–4.3, §A.5)."""

N = {
"intro1": """A neural network is a stack of layers. Each layer multiplies its input by a matrix.
So here is a strange fact. If that is all the layers do, then a network with one hundred layers
is no smarter than a network with one layer. All that depth is wasted. In this video we find
out why, and we fix it with one small bend.""",

"intro2": """That bend is called an activation function. You will see why two linear layers
collapse into one, how the ReLU function works, and how a few bends can draw a curve. This is
the last atom of the series. You need matrix multiplication from video two, and the derivative
from video seven.""",

# ── C1: collapse ──
"col1": """Let's test the claim with real numbers. Here is a toy input x, equal to three, minus
one. The first layer is the matrix W one, with rows one, two and zero, one. Multiply: one times
three plus two times minus one is one. Zero times three plus one times minus one is minus one.
So the first layer gives one, minus one.""",

"col2": """The second layer is the matrix W two, with rows two, zero and one, one. Multiply the
result one, minus one. Two times one plus zero is two. One times one plus one times minus one is
zero. So two layers in a row give two, zero.""",

"col3": """Now multiply the two matrices first. W two times W one is the matrix with rows two,
four and one, three. Apply this one matrix to x. Two times three plus four times minus one is
two. One times three plus three times minus one is zero. Again two, zero. One matrix did the job
of two layers.""",

"col4": """This is not luck. Matrix multiplication is associative: W two times W one x equals W
two W one, times x. So any stack of linear layers is just one linear layer in disguise. Linear
maps can only stretch, turn, flip and squash space. Stacking them never creates a curve.""",

# ── C2: ReLU ──
"relu1": """The fix is to put something that is not linear between the layers. A function that is
not linear is called a nonlinearity. In a neural network, the nonlinearity applied after a layer
is called an activation function. It acts on every number separately.""",

"relu2": """The simplest and most popular one is ReLU, the rectified linear unit. ReLU of z is the
maximum of zero and z. In words: if z is positive, keep it. If z is negative, replace it with
zero. The graph is flat at zero on the left, and a straight diagonal line on the right.""",

"relu3": """Let's compute some values. ReLU of minus two is zero. ReLU of minus zero point five is
also zero. ReLU of zero is zero. ReLU of one point five is one point five. And ReLU of three is
three. Negative numbers are switched off. Positive numbers pass through unchanged.""",

"relu4": """Now put ReLU between our two layers. The first layer gave one, minus one. ReLU turns
that into one, zero. Then W two gives two, one. Without ReLU we got two, zero. The answer
changed, so the two layers no longer collapse into one matrix. The bend broke the shortcut.""",

# ── C3: other shapes ──
"shape1": """ReLU is not the only bend. The sigmoid function squashes any number into the range
from zero to one. It is smooth, and sigmoid of zero is exactly one half. That makes it useful
when the output should look like a probability.""",

"shape2": """The tanh function, the hyperbolic tangent, has a similar S shape, but its range is
from minus one to one. Tanh of zero is zero. It is smooth as well. Both sigmoid and tanh become
very flat for large positive or large negative inputs.""",

"shape3": """GELU is a smooth version of ReLU. It multiplies z by the probability that a standard
normal random value is below z. For large positive z it is almost z. For large negative z it is
almost zero. GELU of one is about zero point eight four.""",

"shape4": """Look closely at the left side. GELU dips slightly below zero. GELU of minus one is about
minus zero point one six. So negative inputs still give a small output. GPT-2 uses GELU inside
every transformer block.""",

# ── C4: building curves ──
"bend1": """How can simple bends build complex shapes? Let's try to draw the curve y equals x
squared, between minus two and two. We will use only ReLU pieces. Each piece is a shifted ReLU,
times a weight. Each piece adds one corner to the graph.""",

"bend2": """Start with the constant four, the height of the curve at minus two. Add minus three
times ReLU of x plus two. Now the line goes down with slope minus three. Add two times ReLU of x
plus one. At minus one, the slope changes from minus three to minus one.""",

"bend3": """Add two times ReLU of x. At zero, the slope becomes plus one. Add two times ReLU of x
minus one. At one, the slope becomes plus three. Four pieces, and we have a bent line that
touches the curve at every whole number from minus two to two.""",

"bend4": """How close is it? Our demo measured the largest gap between the bent line and the
curve. With four pieces the gap is zero point two five. With eight pieces it is only zero point
zero six two five. More pieces, smaller error. With enough ReLU units, a network can approximate
any continuous curve on an interval as closely as you like.""",

# ── C5: slope ──
"slope1": """Training needs slopes, as we saw in video fifteen. So what is the derivative of ReLU?
On the right side the graph is a line with slope one. On the left side it is flat, so the slope
is zero. At exactly zero there is a corner. Libraries such as PyTorch simply use zero there.""",

"slope2": """Our demo checked this numerically. At minus two the slope is zero. At three the slope
is one. A slope of one is good news: the signal passes back through the unit without shrinking.
This is one reason ReLU replaced sigmoid and tanh, whose slopes become tiny in their flat
tails.""",

"slope3": """But a slope of zero has a dark side. Imagine a unit with weight one and bias minus ten,
and inputs between zero and five. Its input to ReLU is always negative. So its output is always
zero, and its slope is always zero. The demo found zero inputs, out of eleven, with slope one.""",

"slope4": """Such a unit gets no learning signal, so it never changes. It is called a dead ReLU.
This is one reason for smooth cousins like GELU. Their slope for negative inputs is small, but it
is not exactly zero, so the unit can still learn.""",

# ── C6: ML ──
"ml1": """Now let's see where activations live. A multilayer perceptron, or MLP, alternates
layers and bends: linear, ReLU, linear, ReLU, linear. The book Build a Large Language Model uses
exactly this pattern, with thirty and then twenty hidden units. There is one activation after
every linear layer, except the last.""",

"ml2": """In a transformer like GPT-2, each block has a small feed-forward network: linear, GELU,
linear. The GELU is the only nonlinear piece in that sub-network. An activation like ReLU or GELU
has no parameters at all. It adds zero to the parameter count, but it is what makes depth
useful.""",

"ml3": """Where does this atom come from? The layers come from axiom A, linearity. The activation
is the one piece that is deliberately not linear. Linearity alone gives one matrix. Linearity plus
a bend gives deep learning.""",

# ── practice ──
"prac1": """Now it is your turn. Pause the video and try this. Apply ReLU to the vector minus three,
zero point five, two. Then give the slope of ReLU at each of the three entries. Take a moment.""",

"prac2": """Here is the answer. ReLU of minus three is zero. ReLU of zero point five is zero point
five. ReLU of two is two. So the result is zero, zero point five, two. The slopes are zero, one
and one, because only the first entry is negative.""",

"prac3": """One more question. Define g of x as ReLU of x plus ReLU of minus x. What is g of minus
two? What is g of three? Can you name the function g? Pause and try.""",

"prac4": """For minus two: ReLU of minus two is zero, and ReLU of two is two. So g of minus two is
two. For three: ReLU of three is three, and ReLU of minus three is zero. So g of three is three.
The function g is the absolute value. Two ReLU pieces made a corner at zero.""",

# ── code ──
"code1": """Let's check everything in Python with NumPy. ReLU is one line: the maximum of zero and z.
We compare two layers with one combined matrix, and then with ReLU in between. Then we build the
curve x squared from shifted ReLUs and measure the error. Finally, we check the slopes.""",

"code2": """This output is real; the script ran while the video was made. Two layers and one matrix
both give two, zero. With ReLU in between we get two, one. The four ReLU weights are minus three,
two, two and two, with a largest error of zero point two five. The slopes are zero and one, and
the dead unit has slope one for zero inputs.""",

# ── summary ──
"sum1": """In summary. A stack of linear layers is just one linear layer, because matrix
multiplication is associative. An activation function puts a nonlinearity between the layers.
ReLU keeps positive numbers and turns negative numbers into zero. Sigmoid, tanh and GELU are
smooth alternatives.""",

"sum2": """Sums of shifted ReLUs can draw curves. The slope of ReLU is zero or one, and units that
are always negative can die. MLPs use ReLU, and GPT-2 uses GELU. That completes the atoms. Next,
the molecules map shows how these atoms combine into gradient descent, attention and more. Thank
you for learning with me.""",
}
