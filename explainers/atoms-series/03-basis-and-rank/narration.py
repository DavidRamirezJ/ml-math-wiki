"""Narration for: Basis & rank — how many directions are really there (ML Math Atoms · Video 3 of 16).
One block <= ~75 words. Source: MML book §2.5, §2.6; Raschka App. E."""

N = {
"intro1": """Imagine a table of data with one hundred columns. It looks like one hundred
different directions. But some columns may only repeat others. A length in meters and the
same length in centimeters say the same thing twice. So how many directions are really
there? That number has a name, the rank, and today we learn to count it.""",

"intro2": """You will learn when vectors are redundant, what a basis is, how to describe one
point with two different bases, and how to find the rank of a matrix. You need video one,
about vectors and span, and video two, about matrices.""",

# ── C1: redundancy ──
"red1": """In video one we met a warning. The vectors two, one and four, two point along the
same line. Four, two is just two times two, one. Together they only reach a line, not the
whole plane. The second vector adds nothing new. We say it is redundant.""",

"red2": """Redundancy can also hide. Take three vectors in the plane. The vector a is two,
one. The vector b is one, two. The vector c is three, three. No two of them are parallel. So
is each one really needed?""",

"red3": """Look closer. Walk along a, and then along b. You arrive at three, three, which is
exactly c. So c equals a plus b. Anything you can build with c, you can already build with a
and b. The vector c brings no new direction.""",

"red4": """So the span of a, b and c is the same as the span of a and b alone: the whole
plane. We have three vectors, but only two real directions. We need a test that finds this
kind of hidden redundancy, even in a hundred dimensions.""",

# ── C2: linear independence ──
"ind1": """Here is the test. A set of vectors is called linearly independent if no vector in
the set is a linear combination of the others. Then each vector adds a new direction. If at
least one vector can be built from the others, the set is linearly dependent.""",

"ind2": """There is an equivalent way to say it, and it is easier to check. Look for weights
that make the linear combination equal to the zero vector. All-zero weights always work. The
vectors are independent when that is the only way. If other weights also give zero, they are
dependent.""",

"ind3": """Read the formula symbol by symbol. Lambda is a Greek letter. Here, lambda i is a
weight, a scalar. And x i is the i-th vector. The big sum adds up the weighted vectors. The
arrow says: if this sum is the zero vector, then every lambda must be zero.""",

"ind4": """Example one: a, b and c from before. Take the weights one, one, and minus one. The
first components give two plus one minus three, which is zero. The second components give one
plus two minus three, which is zero. Walk along a, then b, then back along c, and you return
to the start. Non-zero weights give zero, so a, b and c are dependent.""",

"ind5": """Example two: only a and b. Suppose lambda one times two, one plus lambda two times
one, two is the zero vector. The first components give two lambda one plus lambda two equals
zero. So lambda two equals minus two lambda one.""",

"ind6": """Put that into the second components: lambda one plus two lambda two equals zero.
That becomes lambda one minus four lambda one, which is minus three lambda one. So lambda one
is zero, and then lambda two is zero too. Only the zero weights work, so a and b are
independent.""",

"ind7": """Two quick facts. Any set that contains the zero vector is dependent, because one
times the zero vector is already zero. And a set where one vector is a multiple of another,
like two, one and four, two, is dependent too.""",

# ── C3: basis & coordinates ──
"basis1": """Now we can say exactly what a good set of building blocks is. A basis of a vector
space is a set of vectors that is linearly independent and spans the whole space. It is a
smallest spanning set: remove any vector and you lose part of the space.""",

"basis2": """The most familiar basis of the plane is e one, equal to one, zero, and e two, equal
to zero, one. They are independent, and every point is a combination of them. The point one,
three is one times e one plus three times e two. We say its coordinates are one and three.""",

"basis3": """But a basis is not unique. Here is another one. The vector b one is one, one, and b
two is minus one, one. They are not parallel, so they are independent, and together they
span the plane. The new grid is turned and stretched, but it still covers everything.""",

"basis4": """Now describe the same point, one, three, with the new basis. We need weights c one
and c two, so that c one times b one plus c two times b two equals one, three. The first
components say: c one minus c two equals one. The second components say: c one plus c two
equals three.""",

"basis5": """Add the two equations. Two c one equals four, so c one is two. Then c two is one.
Check it. Two times one, one is two, two. Add minus one, one, and you get one, three. So in
the new basis, the same point has the coordinates two and one.""",

"basis6": """Same point, two different descriptions. The coordinates depend on the basis you
choose. And because basis vectors are independent, each point has exactly one set of
coordinates in a given basis. With redundant vectors, one point could have many
descriptions.""",

"dim1": """Here is a deep fact. Every basis of a space has the same number of vectors. This
number is called the dimension. The plane has dimension two. Ordinary space, with length,
width and height, has dimension three. That is also why three vectors in the plane are always
dependent.""",

"dim2": """Careful with a common mistake. Dimension is not the number of entries in each vector.
The span of the single vector zero, one is a line. Its vectors have two entries, but the line
has dimension one. In the same way, a flat plane through the origin in three-dimensional space
has dimension two.""",

# ── C4: rank ──
"rank1": """Now back to matrices. The rank of a matrix is the number of linearly independent
columns. It answers our question: how many real directions do the columns contain? All the
combinations of the columns form the column space, and its dimension is the rank.""",

"rank2": """Example one. This three by three matrix has the columns one, two, zero, then minus
one, minus two, zero, then three, six, zero. The second column is minus one times the first.
The third column is three times the first. So only one column is independent. The rank is
one, and the column space is just a line.""",

"rank3": """Example two. The columns are one, zero, one, then zero, one, one, then one, one, two.
The first two columns are not parallel, so they are independent. But the third column is the
first plus the second. So the rank is two, and the column space is a flat plane through the
origin.""",

"rank4": """And the three by three identity matrix has three independent columns, so its rank is
three. Its columns reach all of three-dimensional space. The rank can never be larger than
the number of rows or the number of columns. When it is as large as possible, we say the
matrix has full rank.""",

"rank5": """One more surprising fact from the book. The number of independent rows always equals
the number of independent columns. So you can count the rank with rows or with columns. In
practice, a computer finds it for you, as we will see in the Python demo.""",

# ── C5: ML ──
"ml1": """Why does this matter in machine learning? Look at this toy table of three objects.
The first column is the length in meters. The second column is the same length in
centimeters, which is one hundred times the first column. The third column is the weight in
kilograms.""",

"ml2": """There are three columns, but the second one is a multiple of the first. So the rank
is two, not three. The data has only two real directions. A redundant feature like this adds
no new information, and finding the rank helps us detect it.""",

"ml3": """Rank is also behind a popular method for fine-tuning large language models, called
LoRA, which stands for low-rank adaptation. LoRA lets each weight matrix change only along a
few directions, for example sixteen out of seven hundred sixty-eight. Fewer directions means
far fewer numbers to train. We only mention it here.""",

"axiom": """Where does all this come from? Independence, basis, dimension and rank are all
questions about linear combinations: which points can we reach by adding and scaling? So this
whole video rests on axiom A, linearity.""",

# ── C7: practice ──
"prac1": """Your turn. Pause the video and try this. Are the vectors one, two and three, six
linearly independent? If not, find weights, not all zero, that combine them into the zero
vector.""",

"prac2": """Three, six is three times one, two. So they are dependent. One choice of weights is
three and minus one. Three times one, two is three, six, and minus three, six cancels it.
Three times the first vector minus the second gives zero, zero.""",

"prac3": """Second question. What is the rank of the two by three matrix with rows one, two,
three and two, four, six? Count the independent columns, or the independent rows. Pause and
try.""",

"prac4": """The second row is two times the first row. So there is only one independent row,
and the rank is one. The columns agree: one, two, then two, four, then three, six. They are
all multiples of one, two. One independent column, so the rank is one.""",

# ── C7: code ──
"code1": """Let's check it in NumPy. The function matrix rank counts the independent columns
for us. We test a, b and c, solve for the coordinates of one, three in the new basis, and find
the rank of our three examples and of the toy table.""",

"code2": """This output is real; the script ran while the video was made. A plus b minus c is
the zero vector, and the rank of a, b and c is two. The coordinates in the new basis are two
and one. The ranks of the examples are one, two and three. And the toy table has rank two.""",

# ── C8: summary ──
"sum1": """In summary. Vectors are linearly independent when the only way to combine them into
zero is with all-zero weights. A basis is an independent set that spans the space. Every point
has unique coordinates in a basis, and the coordinates change when the basis changes.""",

"sum2": """Every basis has the same size: the dimension. The rank of a matrix is the number of
independent columns, which equals the number of independent rows. Next, in video four: linear
mappings, the functions that respect adding and scaling. See you there.""",
}
