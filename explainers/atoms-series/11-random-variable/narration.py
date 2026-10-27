"""Narration for: Random variable — turning chance into numbers (ML Math Atoms · Video 11 of 16).
One block <= ~75 words. Source: wiki random-variable (MML §6.1–6.2)."""

N = {
"intro1": """Will this customer click on the ad? Is this email spam? What word will a language
model write next? Machine learning lives with uncertainty. We cannot know the answer before
it happens. But to compute with uncertainty, we first need numbers. The tool that turns
chance into numbers is called a random variable.""",

"intro2": """In this video you will learn what a random variable really is: not something
mysterious, but a simple rule. You will build random variables from coins and dice, ask
questions about them, and watch them in a computer simulation. You only need counting and
the idea of a function. This is the first video about axiom D, the rules of probability.""",

# ── C1: experiments and outcomes ──
"exp1": """Start with an experiment. In probability, an experiment is anything with an
uncertain result: flipping a coin, rolling a die, picking a random email. Each possible
result is called an outcome. A coin has two outcomes: heads and tails.""",

"exp2": """Now roll two dice, a blue one and a green one. The blue die shows one to six. The
green die also shows one to six. Every pair is a different outcome. Six times six gives
thirty six outcomes. Here they are in a grid. Each cell is one outcome, for example blue
three and green four.""",

"exp3": """Notice something. Outcomes are not always numbers. Heads is a word. An email is a
text. A photo is a picture. Even for dice, an outcome is a pair, not one number. But
mathematics and computers want numbers. So we need a way to go from outcomes to numbers.""",

# ── C2: the rule ──
"rv1": """Here is the key idea. A random variable is a rule that gives each outcome a number.
That is all. Take the coin. Let Y be one if the outcome is heads, and zero if it is tails.
Each outcome gets exactly one number. Y is a random variable.""",

"rv2": """Back to the two dice. Let X be the sum of the two dice. The rule reads the
outcome and adds the two faces. Blue three and green four give seven. Blue six and green six
give twelve. Fill in the whole grid, and every outcome now has a number from two to
twelve.""",

"rv3": """A word about notation. We write the rule with a capital letter, X. One value it
can take gets a small letter, x. So X of blue three, green four equals seven. And a fun
fact: a random variable is neither random nor a variable. It is a function. The randomness
lives in the outcome, not in the rule.""",

"rv4": """One experiment can carry many random variables. Take the outcome blue two, green
five. The sum is seven. The largest face is five. The difference, green minus blue, is
three. Three different rules, three different numbers, from the very same roll. We choose
the rule that measures what we care about.""",

# ── C3: events ──
"ev1": """Now we can ask questions. An event is a set of outcomes, such as the outcomes where
X is at least ten. Which cells are they? A sum of ten has three cells, eleven has two, and
twelve has one. So the event X at least ten has six outcomes out of thirty six.""",

"ev2": """Another question: when is X equal to seven? Look at the grid. The sevens lie on
one diagonal: one and six, two and five, all the way to six and one. That is six outcomes
too. No other sum has that many. So seven is the most common sum of two dice.""",

"ev3": """Some events are extreme. X equal to one is impossible: the smallest sum is two.
That event is empty, it has no outcomes. And X at most twelve always happens: all thirty six
outcomes are in it. Counting outcomes like this tells us how likely each value is.""",

"ev4": """With fair dice, every outcome has the same chance, one in thirty six. So the
chance of an event is its number of outcomes divided by thirty six. The chance that X is at
least ten is six over thirty six, about zero point one six seven. The full table of chances
for every value is the topic of video twelve.""",

# ── C4: discrete and continuous ──
"dc1": """Random variables come in two kinds. A discrete random variable takes separate
values that you can list: zero or one for the coin, two to twelve for the dice, or the
number of emails you get in a day. You can count them, one by one.""",

"dc2": """A continuous random variable can take any value in a range. Think of the height of
a random person, like one hundred seventy two point three centimeters. Or the time you wait
for the next bus: two minutes, two point five, two point five one. Between any two values
there are always more values.""",

"dc3": """For continuous variables, we ask about ranges instead of exact values. For example:
what is the chance that the waiting time is between two and five minutes? The rule idea does
not change. An outcome comes in, a number comes out. Only the set of possible numbers is
different.""",

# ── C5: samples ──
"sm1": """We can also see a random variable by running the experiment many times. Each run
gives one outcome, and the rule gives one number. That number is called a sample of X. Here
are the first six runs of a computer simulation: the sums are ten, eleven, nine, eight,
three, and eight.""",

"sm2": """Now run it ten thousand times and count how often each sum appears. This bar chart
is called a histogram. It has a triangle shape. The middle value, seven, appears most often,
about one thousand six hundred eighty times. Two and twelve are rare, because each has only
one outcome.""",

"sm3": """Compare the simulation with our counting. The event X at least ten happened in
zero point one six three seven of the runs. Counting gave zero point one six seven. They are
close, but not equal, because samples are random. With more runs, the share usually gets
closer to the counted value.""",

"sm4": """This is exactly the situation in machine learning. We almost never know the rule
and all the outcomes. We only see samples, and those samples are our data. A histogram of the
data is our window into the random variable. The more samples we have, the clearer the
picture becomes.""",

# ── C6: ML ──
"ml1": """In machine learning, random variables are everywhere. Pick a random email from your
inbox. The label Y is one if it is spam and zero if it is not. That is a random variable,
exactly like the coin. A classifier tries to predict Y from the email.""",

"ml2": """Measurements are random variables too. A thermometer shows the true temperature
plus a small error, called noise. If the true value is twenty degrees, the reading might be
twenty point three, or nineteen point eight. The noise is a continuous random variable, and
so is the reading.""",

"ml3": """And a data set is a collection of samples. Each row is one sample of a random
vector: a list of random variables measured together, like age, income, and the label. When
a language model writes, the next word is also a random variable. Each time it generates, we
see one sample.""",

"ml4": """Here is a real example from the book Build a Large Language Model from Scratch.
After the words every effort moves you, a small model gave the next word forward a
probability of about zero point five seven, toward about zero point three six, and closer
about zero point zero six. In one thousand samples, forward appeared five hundred eighty two
times. Each sample is one value of the same random variable.""",

# ── C7: practice ──
"prac1": """Your turn. Flip two coins. Let Z be the number of heads. Pause the video. Write
down all the outcomes, and give each outcome its value of Z. Which outcomes give Z equal to
one?""",

"prac2": """There are four outcomes. Heads heads gives two. Heads tails gives one. Tails
heads gives one. Tails tails gives zero. So Z equals one for two outcomes: heads tails and
tails heads. Notice that two different outcomes can give the same number. That is
allowed.""",

"prac3": """Second question. Roll two fair dice, and let X be their sum. How many outcomes
are in the event X at most three? And what is its chance? Pause and try.""",

"prac4": """A sum of two needs one and one. A sum of three needs one and two, or two and one.
That is three outcomes. The chance is three over thirty six, which is one twelfth, about
zero point zero eight three.""",

# ── C7: code ──
"code1": """Let's simulate it in Python with NumPy. We create a random number generator with a
fixed seed, so the run repeats exactly. We roll two dice ten thousand times. The random
variable X is the sum of each row. Then we count each sum, and the share of runs where X is
at least ten.""",

"code2": """This output is real; the script ran while the video was made. The first rolls give
sums ten, eleven, nine, eight, three, and eight. Seven appeared one thousand six hundred
eighty times, the most of all. The share of X at least ten is zero point one six three
seven, close to the counted zero point one six seven. The coin flips give seven ones and
three zeros.""",

# ── C8: summary ──
"sum1": """In summary. An experiment has outcomes. A random variable is a rule that gives each
outcome a number. One experiment can have many random variables. An event is a set of
outcomes, like X at least ten, and with equally likely outcomes, its chance comes from
counting.""",

"sum2": """Random variables can be discrete or continuous, and we see them through samples.
Labels, noise, and data sets are all random variables. Next, in video twelve, we write down
the full rules of chance: the probability distribution. See you there.""",
}
