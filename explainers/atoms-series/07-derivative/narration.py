"""Narration for: Derivative — the slope right here (ML Math Atoms · Video 7 of 16).
One block <= ~75 words. Source: MML book §5.1 (Defs 5.1-5.2, rules 5.29-5.32), §5.8 (Taylor)."""

N = {
"intro1": """A neural network learns by changing its weights a tiny bit at a time. Before each
change, it needs to answer one question for every weight. If I nudge this weight a little,
how much does the error change, and in which direction? The number that answers this
question is called the derivative. It is the topic of this video.""",

"intro2": """You will learn what a slope at a single point means, how to compute it as a
limit, and the key idea behind it: close up, every smooth curve looks like a straight line.
Then you will use a few simple rules, and find the lowest point of a curve. You only need
school algebra: functions, graphs, and powers like x squared.""",

# ── C1: from average slope to tangent ──
"avg1": """Let's start with a straight line. Its slope tells you how much it rises for each
step to the right. Slope is rise divided by run. This line rises two for every step of one,
so its slope is two. A straight line has the same slope everywhere.""",

"avg2": """Now take a curve: f of x equals x squared. Its steepness changes from place to
place. Near zero it is flat. Far to the right it is very steep. So what is the slope of
this curve at the single point x equals three? With only one point, rise over run makes no
sense yet.""",

"avg3": """Here is the trick. Take a second point, a small step h to the right. Draw the
straight line through both points. Its slope is the average slope between them. With h
equal to one, we go from three to four. The rise is sixteen minus nine, which is seven. The
run is one. So the average slope is seven.""",

"avg4": """Now shrink the step h. Watch the second point slide toward the first one. The
line through both points turns, and it settles on one special line. This line just touches
the curve at x equals three. It is called the tangent line. Its slope is the slope of the
curve right here.""",

# ── C2: the limit ──
"lim1": """Let's write this as a formula. The average slope between x and x plus h is f of
x plus h, minus f of x, all divided by h. The top is the rise. The bottom is the run. Now
let h get closer and closer to zero. The value this ratio approaches is called a limit.""",

"lim2": """The limit of this ratio, as h goes to zero, is the derivative of f at x. We write
it as f prime of x, or as d f over d x. Read the formula like this: the change in the
output, divided by the change in the input, for a change so small that it shrinks to
nothing.""",

"lim3": """Let's compute it for x squared at x equals three. With h equal to one, the ratio
is seven. With h equal to zero point one, three point one squared is nine point six one.
Subtract nine, divide by zero point one: six point one. With h equal to zero point zero one,
we get six point zero one.""",

"lim4": """Seven, six point one, six point zero one. The numbers are walking toward six. So
the derivative of x squared at three is six. Notice the common mistake. We never put h
equal to zero directly. Then we would get zero divided by zero, which means nothing. We
only ask where the ratio is heading.""",

"lim5": """Algebra shows why the answer is six. Expand x plus h, squared: x squared, plus
two x h, plus h squared. Subtract x squared, and divide by h. What is left is two x plus h.
As h goes to zero, this becomes two x. At x equals three, two x is six. The derivative of x
squared is two x.""",

# ── C3: local linearity ──
"loc1": """Here is the key idea of the whole calculus part of this series. Zoom in on the
curve near x equals three. Zoom in more. And more. The curve looks less and less curved.
Close enough, it looks like a straight line, and that line is the tangent. Every smooth
curve is almost straight when you look closely.""",

"loc2": """So near x, we can replace the curve with its tangent line. In a formula: f of x
plus h is approximately f of x, plus f prime of x times h. This is called the local linear
approximation. The new value is the old value, plus the slope times the step. In this
series, this is axiom C: local linearity.""",

"loc3": """Let's check it with numbers. Take f of x equals x squared, x equals three, and h
equal to zero point one. The formula says: nine, plus six times zero point one, which is
nine point six. The true value of three point one squared is nine point six one. The guess
is wrong by only zero point zero one.""",

"loc4": """A second example: x cubed at x equals two. Its derivative there is twelve; we will see why in a moment. The
formula guesses that two point one cubed is eight, plus twelve times zero point one, which
is nine point two. The true value is nine point two six one. Again very close. For smaller
steps, the guess gets even better.""",

"loc5": """Notice what the step h does in the formula. It is multiplied by one fixed number,
the slope. Double the step, and the change doubles. So, near one point, the change in the
output is approximately a linear mapping of the step. That is the bridge between calculus and linear
algebra.""",

# ── C4: rules ──
"rul1": """Computing limits every time would be slow. Luckily, a few rules do most of the
work. The first is the power rule. The derivative of x to the power n is n times x to the
power n minus one. For x squared, that gives two x, just as we found. For x cubed, it gives
three x squared.""",

"rul2": """The second rule is about a constant factor. If you multiply a function by a
number c, its derivative is multiplied by c too. The third rule is about sums. The
derivative of a sum is the sum of the derivatives. You can take each piece on its own.""",

"rul3": """Let's use all three. Take f of x equals three x squared plus five x. The
derivative of x squared is two x, so three x squared gives six x. The derivative of five x
is five. So f prime of x is six x plus five. At x equals one, the slope is eleven.""",

"rul4": """One more rule is so important that it gets its own video: the rule for a
function inside another function. We will study it in video nine. For now, the power rule,
the constant rule, and the sum rule are enough for many curves.""",

# ── C5: flat at the bottom ──
"flat1": """Now a very useful fact. Here is the curve x minus two, squared, plus one. Watch
the tangent line move along it. On the left, the curve goes down, and the slope is
negative. On the right, the curve goes up, and the slope is positive. At the bottom, the
tangent is flat.""",

"flat2": """So at a minimum, the derivative is zero. Let's check. The derivative is two
times x minus two. It is zero when x equals two, and that is exactly the lowest point. The
same is true at a maximum, the top of a hill: there the tangent is also flat.""",

"flat3": """Be careful with the opposite direction. A zero derivative does not always mean a
minimum or a maximum. For x cubed, the slope at zero is zero, but the curve keeps going up
on both sides. A zero slope only tells you where to look.""",

# ── C6: in ML ──
"ml1": """Now let's connect this to machine learning. A loss is an error score: smaller is
better. We study it in video fifteen. Here is a toy model with one weight w. Its loss is
two w minus six, squared. Expanded, that is four w squared, minus twenty four w, plus thirty
six.""",

"ml2": """With our rules, the derivative is eight w minus twenty four. At w equals one, the
slope is minus sixteen. What does that mean? If we nudge w up by zero point zero one, the
loss changes by about minus sixteen times zero point zero one. That is minus zero point one
six. The true change is minus zero point one five nine six.""",

"ml3": """The sign tells the direction. A negative slope says: increase w, and the loss goes
down. The size says how fast. And at w equals three, the slope is zero. That is the bottom,
the best weight. Real models have millions of weights. In video eight we take one slope
for each weight, and bundle them together.""",

# ── C7: practice ──
"prac1": """Now it is your turn. Pause the video and try this. First, what is the derivative
of x squared at x equals five? Second, use the local linear approximation to guess five
point one squared, without a calculator.""",

"prac2": """Here are the answers. The derivative of x squared is two x, so at five it is
ten. The guess for five point one squared is twenty five, plus ten times zero point one,
which is twenty six. The true value is twenty six point zero one. Very close.""",

"prac3": """One more question. Take f of x equals four x cubed, minus two x. What is f prime
of x, and what is the slope at x equals one? Use the power rule, the constant rule, and the
sum rule. Pause and try.""",

"prac4": """The power rule turns x cubed into three x squared. Times four, that is twelve x
squared. The derivative of minus two x is minus two. So f prime of x is twelve x squared
minus two. At x equals one, the slope is twelve minus two, which is ten.""",

# ── C7: code ──
"code1": """Let's check all of this in Python. We write the average slope as a small
function, called ratio. Then we compute it for x squared at three, with smaller and smaller
steps h. We also test the local linear guess, the x cubed example, and the slope of our toy
loss.""",

"code2": """This output is real; the script ran while the video was made. The ratios are
seven, six point one, six point zero one, and six point zero zero one. They approach the
exact value six. The linear guess nine point six is close to nine point six one. And the
toy loss has slope minus sixteen at w equals one.""",

# ── C8: summary ──
"sum1": """In summary. The derivative is the slope of a curve at one point. It is the limit
of the average slope, f of x plus h minus f of x, over h, as h goes to zero. For x squared,
it is two x. Close up, every smooth curve looks straight, so f of x plus h is about f of x
plus f prime of x times h.""",

"sum2": """The power, constant, and sum rules make derivatives quick. At a minimum or a
maximum, the derivative is zero. In machine learning, the derivative says how the loss
changes when you nudge one weight. Next, in video eight: the gradient, the slope in many
directions at once. See you there.""",
}
