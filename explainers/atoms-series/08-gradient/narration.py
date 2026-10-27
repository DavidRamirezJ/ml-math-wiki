"""Narration for: Gradient — the slope in many directions (ML Math Atoms · Video 8 of 16).
One block <= ~75 words. Source: MML book §5.2 (Def. 5.5, eq. 5.40); LLM book §A.4."""

N = {
"intro1": """In the last video, one weight had one slope. But a real model has millions of
weights, and its error depends on all of them at the same time. So we need a slope for
each weight. Bundle all those slopes into one vector, and you get the gradient. It is the
arrow that tells a model which way is uphill.""",

"intro2": """In this video you will learn to read a function of two inputs as a map of
heights. You will compute partial derivatives, build the gradient, and see why it points
straight uphill. You need video five on the inner product, video six on the norm, and
video seven on the derivative.""",

# ── C1: functions of two inputs ──
"map1": """So far, our functions had one input and one output, so we could draw them as a
curve. Now take a function with two inputs, x and y, and one output. For example, f of x
and y equals x squared plus y squared. At the point three, four, the output is nine plus
sixteen, which is twenty five.""",

"map2": """Think of the output as a height above the ground. Each point of the flat x y
plane gets a height. Together, the heights form a surface, like a landscape. This one is a
bowl. Its lowest point is the origin, where the height is zero.""",

"map3": """A three dimensional picture is hard to read. Hikers use a better trick: a
contour map. A contour line joins all the points that have the same height. For our bowl,
the points with height twenty five form a circle of radius five. Lower heights give
smaller circles. Here we draw the heights five, ten, fifteen, twenty, and twenty five.""",

"map4": """Read the map like this. Where the contour lines are close together, the ground
is steep. Where they are far apart, the ground is gentle. We will draw the slope of this
landscape as arrows on the map, without any heavy three dimensional drawing.""",

# ── C2: partial derivatives ──
"par1": """Here is the main question. At one point, how steep is the landscape? The answer
depends on the direction you walk. So start with the two simplest directions. Walk only in
the x direction, and keep y frozen. Or walk only in the y direction, and keep x frozen.""",

"par2": """Take a new example: f of x and y equals x squared plus three y, at the point one,
two. Freeze y at two. Then f becomes x squared plus six, a function of x alone. Its
derivative is two x. At x equals one, that is two. So the slope in the x direction is
two.""",

"par3": """Now freeze x at one instead. Then f becomes one plus three y, a function of y
alone. Its derivative is three, everywhere. So the slope in the y direction is three. Each
of these slopes is called a partial derivative, because it uses only part of the
inputs.""",

"par4": """We write a partial derivative with a curly d. Read it as partial f, partial x:
the slope of f when only x moves. The rule is simple. Treat every other input as a
constant number, and use the normal derivative rules from video seven.""",

# ── C3: the gradient ──
"grad1": """Now bundle the partial derivatives into one vector. This vector is called the
gradient. We write it with an upside down triangle, called nabla. For x squared plus three
y at the point one, two, the gradient is two, three. It is an arrow on the map.""",

"grad2": """Let's do the bowl, x squared plus y squared. The partial with respect to x is
two x. The partial with respect to y is two y. So the gradient is two x, two y. At the point
three, four, it is six, eight. Its length is ten.""",

"grad3": """Draw the gradient at many points of the map. Every arrow points away from the
bottom of the bowl, straight up the hill. The arrows are long where the bowl is steep, far
from the center. They are short near the bottom, where the bowl is almost flat.""",

"grad4": """A small note on writing. Some books, including the Mathematics for Machine
Learning book, write the gradient as a row of numbers. Here we draw it as an arrow. It is
the same information. In machine learning code, the gradient usually has exactly the same
shape as the weights.""",

# ── C4: local linearity in many dimensions ──
"lin1": """In video seven, we learned the key idea: close up, a function looks linear. The
new value is about the old value, plus the slope times the step. With many inputs, the step
h is a vector. And the slope times the step becomes the dot product of the gradient with
h.""",

"lin2": """So the formula is: f of x plus h is approximately f of x, plus the gradient dot
h. The dot product multiplies each partial slope by the step in that direction, and adds the
results. This is axiom C, local linearity, written with the inner product of video five.""",

"lin3": """Let's check it. For x squared plus three y at one, two, the value is seven. Step
by h equal to zero point one, zero point one. The gradient dot h is two times zero point one,
plus three times zero point one. That is zero point five. So the guess is seven point five.
The true value is seven point five one.""",

"lin4": """A second check, on the bowl at three, four. Step by zero point one, minus zero
point one. The gradient dot h is six times zero point one, minus eight times zero point one.
That is minus zero point two. So the guess is twenty four point eight. The true value is
twenty four point eight two. Very close again.""",

# ── C5: steepest uphill ──
"st1": """Now the big question. Of all directions, which one goes uphill fastest? Take a
step of length one in some direction u. The change in height is about the gradient dot u.
From video five, a dot product equals the two lengths times the cosine of the angle between
them.""",

"st2": """The length of u is one, so the change is the length of the gradient, times the
cosine of the angle. Watch the direction turn around the point. When u points along the
gradient, the angle is zero, the cosine is one, and the change is largest: ten. So the
gradient points in the steepest uphill direction.""",

"st3": """At an angle of ninety degrees, the cosine is zero, and the height does not change.
That direction runs along the contour line. So the gradient is always perpendicular to the
contour lines. And at one hundred eighty degrees, the change is minus ten. Minus the
gradient points steepest downhill.""",

"st4": """That last fact is the seed of gradient descent, the main training method. Take a
small step against the gradient, again and again, and you walk down the hill. Gradient
descent is a molecule built from these atoms, so here we only preview it.""",

# ── C6: in ML ──
"ml1": """In machine learning, the landscape is the loss: the error score of the model,
which we study in video fifteen. Its inputs are all the weights. The small GPT two model has
about one hundred twenty four million weights. So its gradient has one hundred twenty four
million partial derivatives, one for each weight.""",

"ml2": """Each partial derivative answers the question from the start of this video. If I
nudge only this weight, how fast does the loss change? The gradient collects all the
answers in one vector. Libraries like PyTorch compute it for us, and store a gradient with
the same shape as each weight matrix.""",

"ml3": """Here is one toy step on our bowl. Start at three, four, where the height is
twenty five. The gradient is six, eight. Step against it, with a step size of zero point
one. The new point is two point four, three point two. The height drops from twenty five to
sixteen. One step, and we are much lower.""",

# ── C7: practice ──
"prac1": """Now it is your turn. Pause the video and try this. Take f of x and y equals
three x plus y squared. What are the two partial derivatives, and what is the gradient at
the point two, one? Remember: freeze the other input.""",

"prac2": """Here are the answers. Freeze y, and three x plus a constant has slope three.
Freeze x, and y squared has slope two y. At the point two, one, two y is two. So the
gradient is three, two.""",

"prac3": """One more question, on the bowl x squared plus y squared at the point zero, five.
What is the gradient? In which direction is the steepest way up? And in which direction
does the height not change at all? Pause and think.""",

"prac4": """The gradient is two x, two y, which is zero, ten. So the steepest way up is
straight in the y direction, with slope ten. The height does not change if you walk in the
x direction, along the circle of radius five. That direction is perpendicular to the
gradient.""",

# ── C7: code ──
"code1": """Let's check this in Python. We write a numeric gradient. For each input, we
nudge only that input up and down by a tiny h, freeze the others, and divide the change by
two h. Then we compare with the exact gradients, test the linear guess, and take one step
downhill.""",

"code2": """This output is real; the script ran while the video was made. The numeric
gradients are two, three and six, eight, equal to the exact ones. The linear guesses are
very close to the true values. The slope along the gradient is ten, and at ninety degrees it
is zero. One step takes the height from twenty five to sixteen.""",

# ── C8: summary ──
"sum1": """In summary. A function of two inputs is a landscape, and a contour map shows it
flat. A partial derivative is the slope when only one input moves and the others are frozen.
The gradient bundles all the partial derivatives into one vector. Close up, f of x plus h is
about f of x, plus the gradient dot h.""",

"sum2": """The gradient points steepest uphill, and it is perpendicular to the contour
lines. Minus the gradient points downhill. In machine learning, the gradient of the loss has
one entry per weight. Next, in video nine: the chain rule, where slopes multiply. See you
there.""",
}
