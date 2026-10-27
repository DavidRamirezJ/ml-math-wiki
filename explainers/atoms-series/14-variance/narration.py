"""Narration for: Variance & covariance — spread and moving together (ML Math Atoms · Video 14 of 16).
One block <= ~75 words. Source: MML book §6.4.1–6.4.4; wiki covariance-and-variance."""

N = {
"intro1": """Two classes take the same quiz. Both classes have an average score of five out of
ten. Are the two classes the same? Not at all. In one class, almost everybody scored five. In
the other class, some students scored two and some scored eight. The average hides this
difference. We need a second number that measures spread.""",

"intro2": """Machine learning needs that number all the time. We often rescale every input so
that its spread is one. Inside every GPT block, layer normalization computes a spread for each
token. In this video you will compute variance, standard deviation and covariance by hand. You
need the expected value from video thirteen.""",

# ── C1: spread ──
"spread1": """Here are the two classes as dot plots. Each dot is one student, placed at its
score. This is a toy example with five students per class. Class A has the scores four, five,
five, five and six. Class B has two, four, five, six and eight. Both lists add up to twenty-five,
so both means are five.""",

"spread2": """Remember the mean. It is the expected value: each value times its probability,
added up. With five equally likely students, every probability is one fifth. So the mean is
the balance point of the dots. For both classes, the balance point sits at five. The center is
the same. The spread is not.""",

"spread3": """How can we measure spread? A natural idea: measure how far each dot is from the
mean. That distance, with its sign, is called a deviation. For class B, the deviations are minus
three, minus one, zero, plus one and plus three. Dots on the left get negative deviations. Dots
on the right get positive ones.""",

"spread4": """Now average the deviations. Minus three, minus one, zero, one and three add up to
zero. So the average deviation is zero. And this is not bad luck. The mean is the balance point,
so the deviations always cancel. The plain average of deviations is always zero. It tells us
nothing about spread.""",

# ── C2: variance ──
"var1": """The fix is to square each deviation before averaging. A square is never negative, so
nothing can cancel. Minus three squared is nine. Three squared is also nine. The left side and
the right side now both add to the total. Big deviations count a lot, small ones count a
little.""",

"var2": """This gives the definition. The variance of X is the expected value of X minus mu, all
squared. Read it from the inside. X is the random value. Mu is its mean. X minus mu is the
deviation. We square it. Then E, the expected value, averages the squares with their
probabilities.""",

"var3": """Let's compute it for class B. The deviations are minus three, minus one, zero, one and
three. The squares are nine, one, zero, one and nine. They add up to twenty. Each student has
probability one fifth, so we divide by five. Twenty divided by five is four. The variance of
class B is four.""",

"var4": """Now class A. The deviations are minus one, zero, zero, zero and one. The squares are
one, zero, zero, zero and one. They add up to two. Two divided by five is zero point four. So
class A has variance zero point four, and class B has variance four. Ten times more spread, in
one number.""",

"var5": """Two facts follow. A variance is never negative, because it averages squares. It is
zero only when every value equals the mean. And a common mistake is to forget the square: then
you always get zero.""",

# ── C3: standard deviation and shortcut ──
"sd1": """There is one problem with variance: its units. The scores are in points. But we
squared the deviations, so the variance of four is in points squared. A points squared is hard
to imagine. To go back to the original units, we take the square root.""",

"sd2": """The square root of the variance is called the standard deviation. Its symbol is sigma.
For class B, the square root of four is two. So the standard deviation is two points. You can
read it as a typical distance from the mean. For class A, the square root of zero point four is
about zero point six three.""",

"sd3": """There is also a shortcut for variance. The variance equals the expected value of X
squared, minus mu squared. In words: the mean of the squares, minus the square of the mean.
Let's check it on class B, with the same numbers.""",

"sd4": """The squares of the scores are four, sixteen, twenty-five, thirty-six and sixty-four.
They add up to one hundred forty-five. Divided by five, that is twenty-nine. The mean is five,
and five squared is twenty-five. Twenty-nine minus twenty-five is four. The same variance as
before.""",


# ── C4: covariance ──
"cov1": """Now take two variables at the same time, for example hours studied, called x, and
quiz score, called y. Each pair becomes one dot in a scatter plot. Look at three toy clouds. In
the first, y goes up when x goes up. In the second, y goes down. In the third, there is no
pattern.""",

"cov2": """We want one number that tells these three apart. The idea is to multiply the two
deviations. If a dot is right of the mean of x and above the mean of y, both deviations are
positive, and the product is positive. Left and below gives minus times minus, which is also
positive. The other two corners give negative products.""",

"cov3": """Average these products, and you get the covariance. It is the expected value of X
minus its mean, times Y minus its mean. A positive covariance means the variables move
together. A negative covariance means they move in opposite directions. A covariance near zero
means no straight-line pattern. Our three clouds give zero point five nine, minus zero point
six five, and zero point one.""",

"cov4": """Let's compute one by hand. Four students studied one, three, five and seven hours.
Their scores were two, three, six and five. The mean of x is four. The mean of y is also four.
So the x deviations are minus three, minus one, one and three. The y deviations are minus two,
minus one, two and one.""",

"cov5": """Multiply the pairs. Minus three times minus two is six. Minus one times minus one is
one. One times two is two. Three times one is three. The products add up to twelve. Divide by
four students, and the covariance is three. It is positive, so more hours go with higher scores
in this toy data.""",

"cov6": """One warning. Covariance only sees straight-line patterns. Take x from minus two up to
two, and y equal to x squared. The covariance is zero, even though y depends completely on x.
Zero covariance does not mean unrelated.""",

# ── C5: covariance matrix ──
"mat1": """With two variables we have three numbers: the variance of x, the variance of y, and
their covariance. We can store them in a two by two matrix, called the covariance matrix. The
variances go on the diagonal. The covariance goes in both off-diagonal places.""",

"mat2": """For our four students, the variance of x is five, and the variance of y is two point
five. The covariance is three. So the matrix is five, three, three, two point five. The matrix is
symmetric, because the covariance of x and y equals the covariance of y and x. With ten features,
the matrix would be ten by ten.""",

"mat3": """Where does all this come from? The mean is a weighted sum, from axiom A, linearity,
with weights that follow axiom D, the probability rules. Now put the x deviations in one vector
and the y deviations in another. Their dot product is six plus one plus two plus three, which is
twelve. Divide by four, and you get three.""",

"mat4": """So covariance is an average dot product of deviation vectors. That is axiom B, the
inner product. Variance is the same thing with a vector and itself: an average squared length.
So variance and covariance are built from axioms A, B and D.""",

"mat5": """One honest detail. We divided by n, the number of data points. For a sample from a
larger population, many tools divide by n minus one instead. This is called Bessel's
correction. NumPy's var divides by n by default, but NumPy's cov divides by n minus one.""",

# ── C6: ML ──
"ml1": """Now let's use variance in machine learning. The first use is standardization. For each
value, subtract the mean and divide by the standard deviation. The result is called a z-score.
For class B, the scores two, four, five, six and eight become minus one point five, minus zero
point five, zero, zero point five and one point five.""",

"ml2": """After standardization, the mean is zero and the variance is one. Features in meters,
dollars or years now share one scale, and many learning methods train more stably.""",

"ml3": """The second use is layer normalization, inside every GPT block. For one token, the model
takes its list of features, computes their mean and variance, and standardizes them. GPT-2 small
has seven hundred sixty-eight features per token. A tiny number, epsilon, is added to the
variance, so the model never divides by zero.""",

"ml4": """The third use is principal component analysis, or PCA. It looks for the directions in
which the data spreads the most. Those directions come from the covariance matrix. Layer
normalization and PCA are molecules: they combine this atom with others. We will meet them in
the molecules map.""",

# ── C7: practice ──
"prac1": """Now it is your turn. Pause the video and try this. The data set is one, one, five and
five. What is its variance, and what is its standard deviation? Start with the mean, then the
deviations, then the squares.""",

"prac2": """Here is the answer. The mean is three. The deviations are minus two, minus two, two
and two. Every square is four. Four plus four plus four plus four is sixteen. Divide by four,
and the variance is four. The square root of four is two, so the standard deviation is two.""",

"prac3": """One more question. Three points: one, six, then two, four, then three, two. Is the
covariance of x and y positive or negative? Can you compute it? Pause and try.""",

"prac4": """The mean of x is two, and the mean of y is four. The deviation pairs are minus one
and two, then zero and zero, then one and minus two. The products are minus two, zero and minus
two. They add up to minus four. Divide by three, and the covariance is minus four thirds. It is
negative: when x goes up, y goes down.""",

# ── code ──
"code1": """Let's check everything in Python with NumPy. We make the two classes, compute the
deviations, and average their squares. We check the shortcut. Then we build the covariance
matrix of hours and scores, once with n and once with n minus one. Finally, we standardize
class B.""",

"code2": """This output is real; the script ran while the video was made. Both means are five,
and the deviations average to zero. The variances are zero point four and four. The shortcut
also gives four, and the covariance is three. With n minus one, the matrix grows. The z-scores
have variance one.""",

# ── C8: summary ──
"sum1": """In summary. Variance is the expected squared distance from the mean. We square the
deviations because plain deviations always cancel to zero. The standard deviation is the square
root of the variance, and it is in the original units. The shortcut says: the mean of the
squares minus the square of the mean.""",

"sum2": """Covariance is the average product of deviations. Its sign says whether two variables
move together or in opposite directions. The covariance matrix collects all variances and
covariances. Next, in video fifteen, we meet the loss function: one number that says how wrong a
model is. See you there.""",
}
