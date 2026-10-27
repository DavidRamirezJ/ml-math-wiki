"""Narration for: Probability distribution — the rules of chance (ML Math Atoms · Video 12 of 16).
One block <= ~75 words. Source: wiki probability-distribution, statistical-independence (MML §6.2–6.4)."""

N = {
"intro1": """You show a photo to an image classifier. It does not just answer cat. It answers:
cat with zero point nine, dog with zero point zero eight, fox with zero point zero two. A
language model does the same for the next word, over tens of thousands of words. These lists
of chances are probability distributions, and they follow a few strict rules.""",

"intro2": """In this video you will build a distribution by counting, see how continuous
chances become areas, and learn the two rules of probability: the sum rule and the product
rule. At the end, you will derive Bayes' theorem in three lines. You need video eleven,
random variables. This is axiom D, the probability rules.""",

# ── C1: table ──
"tab1": """Let's go back to two fair dice and the random variable X, their sum. There are
thirty six equally likely outcomes. Count how many outcomes give each sum. A sum of two has
one outcome. A sum of three has two. The counts grow up to seven, which has six, and then
they shrink back to one for twelve.""",

"tab2": """Divide every count by thirty six, and you get a probability for every value. The
probability that X equals two is one over thirty six, about zero point zero two eight. The
probability that X equals seven is six over thirty six, about zero point one six seven. This
complete list of values and their probabilities is called the probability distribution of
X.""",

"tab3": """Every distribution follows two rules. First, no probability is negative. Second,
all the probabilities add up to one. Check it here: one plus two plus three, and so on up
to six and back down to one, gives thirty six. Thirty six over thirty six is one. Something
must happen, so the total chance is one.""",

"tab4": """Compare this with the simulation from video eleven. There, the sum seven appeared
one thousand six hundred eighty times in ten thousand rolls: a share of zero point one six
eight. The distribution says zero point one six seven. The histogram of samples is an
estimate. The distribution is the exact rule behind it.""",

# ── C2: density ──
"den1": """Now a continuous random variable: the waiting time for a bus that comes every ten
minutes. Every time between zero and ten is equally likely. We cannot list the values, so
instead we draw a curve called a density. Here it is a flat line at height zero point one
over the ten minutes.""",

"den2": """For a density, probability is area under the curve. What is the chance of waiting
between two and five minutes? The area is a rectangle: width three times height zero point
one, which is zero point three. And the area under the whole curve is ten times zero point
one, which is one. The total is one again.""",

"den3": """Many real quantities, like human height, have a bell-shaped density. The idea is
the same: the probability of a range is the area of that slice. A single exact value, like
exactly one hundred seventy point zero zero zero centimeters, has no width. So its
probability is zero. We always ask about ranges.""",

# ── C3: joint ──
"joint1": """Machine learning almost always has more than one random variable. So let's look
at two at once. This is a toy example. Each morning we note the weather, rain or no rain,
and whether a person carries an umbrella. A joint table gives the probability of every pair
of values.""",

"joint2": """Read the table. The probability of rain and umbrella together is zero point two
five. Rain and no umbrella is zero point zero five. No rain and umbrella is zero point one.
No rain and no umbrella is zero point six. The four cells cover every possible morning, so
they add up to one.""",

# ── C4: rules ──
"rule1": """First rule. What is the probability of rain, no matter what happens with the
umbrella? Add across the rain row: zero point two five plus zero point zero five gives zero
point three. This is the sum rule: to forget a variable, add over all its values. The result
is called a marginal.""",

"rule2": """The other row gives the probability of no rain: zero point one plus zero point
six is zero point seven. Adding down the columns gives the umbrella: zero point three five,
and no umbrella: zero point six five. Each marginal is again a distribution, because zero
point three plus zero point seven is one.""",

"rule3": """Now a new question. If we know that it rains, how likely is the umbrella? We keep
only the rain row and rescale it so it adds up to one. Zero point two five divided by zero
point three is about zero point eight three three. This is a conditional probability,
written p of umbrella given rain. The bar means given.""",

"rule4": """Turn that division around, and you get the second rule. The joint probability
equals the conditional probability times the marginal: p of x and y equals p of y given x,
times p of x. Check it: zero point eight three three times zero point three is zero point
two five. This is the product rule.""",

# ── C5: independence and Bayes ──
"ind1": """Sometimes one variable tells us nothing about the other. Then p of y given x is
just p of y, and the product rule becomes p of x and y equals p of x times p of y. This is
called independence. Flip a coin and roll a die. The chance of heads and a six is one half
times one sixth, one twelfth.""",

"ind2": """Are rain and umbrella independent? Multiply the marginals: zero point three times
zero point three five is zero point one zero five. But the table says zero point two five.
They are not equal, so the variables are not independent. Of course: seeing an umbrella
tells you something about the weather.""",

"bay1": """Now the payoff. The product rule works in both directions. p of x and y equals p
of y given x times p of x. It also equals p of x given y times p of y. Both right sides are
equal. Divide by p of y, and you get p of x given y equals p of y given x times p of x, over
p of y.""",

"bay2": """That is Bayes' theorem, derived in three lines from the product rule. It turns a
conditional around. We know p of umbrella given rain. Bayes gives p of rain given umbrella:
zero point eight three three times zero point three, over zero point three five, about zero
point seven one four. Bayes is a molecule, built from this atom.""",

# ── C6: ML ──
"ml1": """In machine learning, these ideas are everywhere. A classifier outputs a
distribution over classes. The probabilities are never negative and they add up to one,
exactly like the dice table. Often it is a conditional distribution: the probability of
each class, given the input photo.""",

"ml2": """A language model outputs a distribution over the next token, given all the words
so far. To write a whole text, it chains these conditionals with the product rule: the
probability of a sentence is a product of next word probabilities. That is why the logs from
video ten are so useful.""",

"ml3": """And independence is the usual assumption about training data. We assume the
examples are independent and come from the same distribution. Then the probability of the
whole data set is a product of the probabilities of the examples. Without this assumption,
most loss functions would be much harder to write.""",

# ── C7: practice ──
"prac1": """Your turn. Use the rain and umbrella table. What is the probability of no
umbrella, given rain? Pause the video and try. Hint: keep only the rain row.""",

"prac2": """The rain row is zero point two five and zero point zero five, and it adds up to
zero point three. Divide zero point zero five by zero point three. You get about zero point
one six seven. Check: zero point eight three three plus zero point one six seven is one, as
a conditional distribution must be.""",

"prac3": """Second question. Two fair dice. What is the probability that the sum is at least
eleven? Use the distribution table. Pause and try.""",

"prac4": """A sum of eleven has probability two over thirty six. A sum of twelve has one over
thirty six. These events cannot happen together, so we add them: three over thirty six,
which is one twelfth, about zero point zero eight three.""",

# ── C7: code ──
"code1": """Let's check everything in NumPy. We build all thirty six dice outcomes with add
dot outer, and count each sum. Then we write the joint table as an array. A sum along each
row gives the marginals. Dividing the rain row by its total gives the conditional. And the
outer product of the marginals shows what independence would look like.""",

"code2": """This output is real; the script ran while the video was made. The dice counts are
one to six and back to one, and the probabilities add up to one. The marginals are zero
point three and zero point seven, and zero point three five and zero point six five. The
product rule gives back zero point two five. Independence would give zero point one zero
five. And Bayes gives zero point seven one four.""",

# ── C8: summary ──
"sum1": """In summary. A probability distribution lists every value of a random variable
with its probability. The probabilities are never negative and add up to one. For a
continuous variable, a density gives probability as area. With two variables, the joint
table holds everything.""",

"sum2": """The sum rule gives marginals, the product rule links joint and conditional
probabilities, and independence means p of x and y equals p of x times p of y. Bayes'
theorem follows in three lines. Next, in video thirteen: the expected value, the long-run
average of a random variable.""",
}
