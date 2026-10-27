"""Narration for: Matrix — a grid of numbers that moves vectors (ML Math Atoms · Video 2 of 16).
One block <= ~75 words. Source: MML book §2.2; Raschka §A.2.3, §4.1."""

N = {
"intro1": """In video one, we saw that data becomes vectors. But a model almost never looks at
just one vector. It looks at a whole table of them at once. And every layer of a neural
network keeps what it has learned in a grid of numbers. That grid has a name. It is called a
matrix.""",

"intro2": """In this video you will learn the two faces of a matrix. It is a table that stores
data, and it is a machine that moves vectors. You will multiply a matrix by a vector, and two
matrices together, by hand. You only need video one: vectors, adding, scaling, and linear
combinations.""",

# ── C1: table ──
"table1": """Let's start with the first face: a table. Here are three houses. For each house we
write three numbers: the area in square meters, the number of bedrooms, and the age in years.
Each house is a vector, just like in video one. Now we stack the three vectors, one per
row.""",

"table2": """The result is a grid of numbers with rows and columns. A rectangular grid of numbers
is called a matrix. This one has three rows and three columns, so we call it a three by three
matrix. We always say the number of rows first, and then the number of columns.""",

"table3": """Each number in the matrix has an address. We write the letter a with two small
numbers: first the row, then the column. So a two one is the number in row two, column one.
Here that is eighty, the area of the second house.""",

"table4": """Read the matrix in two directions. A row is one house: one data point. A column is
one feature across all the houses. For example, the second column lists the bedrooms: three,
two, and four. In machine learning, a batch of data is stored exactly like this, with one row
per example.""",

# ── C2: machine ──
"mach1": """Now the second face: a machine. A matrix can take a vector as input and give back a
new vector. Here is a two by two matrix A. Its first column is two, one, and its second column
is one, three. Here is an input vector x, equal to one, two. What is A times x?""",

"mach2": """Here is the key idea. A times x is a linear combination of the columns of A. The
components of x are the weights. So we take one times the first column, plus two times the
second column. One times two, one is two, one. Two times one, three is two, six.""",

"mach3": """Now add them. Two plus two is four. One plus six is seven. So A times x equals four,
seven. The machine turned the vector one, two into the vector four, seven. Every matrix times
a vector works this way, no matter how big the matrix is.""",

"mach4": """There is a second way to get the same answer: the row view. Take the first row of A,
two, one, and the vector one, two. Multiply matching entries and add. Two times one plus one
times two is four. The second row gives one times one plus three times two, which is
seven.""",

"mach5": """Same answer: four, seven. This multiply-and-add pattern between a row and a vector is
so important that it has its own name, the inner product. It gets its own video, video five.
One rule about shapes: the matrix needs as many columns as the vector has components.""",

# ── C3: grid ──
"grid1": """Let's watch the machine at work on the whole plane. Here is a grid with two special
arrows. The blue arrow, e one, is one, zero. The green arrow, e two, is zero, one. We also
follow a yellow vector, one, two. Now look at the column view again. A times e one is just
the first column. A times e two is the second column.""",

"grid2": """So the columns tell you everything: they say where e one and e two land. Take the
matrix with columns two, zero and zero, one. It sends e one to two, zero, and it leaves e two
where it is. Watch the grid. Everything becomes twice as wide, and the heights stay the
same.""",

"grid3": """The yellow vector one, two became two, two. Check it with the column rule. One times
two, zero, plus two times zero, one, is two, two. The picture and the arithmetic agree.""",

"grid4": """Now a different matrix, with columns zero, one and minus one, zero. It sends e one to
zero, one, and e two to minus one, zero. Each arrow turns a quarter turn to the left. So this
matrix rotates the whole plane by ninety degrees, counterclockwise.""",

"grid5": """The yellow vector one, two now becomes minus two, one. Notice what stays true. The
grid lines stay straight, parallel, and evenly spaced, and the origin does not move. Machines
with this property are called linear maps. They get their own video, video four.""",

# ── C4: multiplication ──
"mult1": """What happens if we use two machines, one after the other? First apply B to x. Then
apply A to the result. Is there one single matrix that does both steps at once? Yes. It is
the product, A B. Note the order: B acts first, because B stands next to x.""",

"mult2": """Computing this product is called matrix multiplication. To find the entry in row i
and column j of A B, take row i of A and column j of B. Multiply matching entries and add them
up. In symbols: A B, i j, equals the sum over k of a i k times b k j.""",

"mult3": """Let's do one completely. A has rows one, two and three, four. B has rows five, six
and seven, eight. Top left: row one of A, which is one, two, with column one of B, which is
five, seven. One times five plus two times seven is nineteen. Top right: one times six plus
two times eight is twenty-two.""",

"mult4": """Bottom left: row two of A, three, four, with column one of B, five, seven. Three
times five plus four times seven is forty-three. Bottom right: three times six plus four times
eight is fifty. So A B has rows nineteen, twenty-two and forty-three, fifty.""",

"mult5": """Now the shape rule. If A is m by n, and B is n by p, then A B is m by p. The two inner
numbers must match, and they disappear. A two by three matrix times a three by two matrix
gives a two by two matrix. But a two by three times a two by three is not allowed.""",

"mult6": """Here is a surprise: the order matters. Take the stretch S and the rotation R. Stretch
first, then rotate: e one goes to two, zero, and then to zero, two. Rotate first, then
stretch: e one goes to zero, one, and the stretch leaves it at zero, one. The results are
different.""",

"mult7": """The matrices agree with the pictures. R S has the first column zero, two. S R has the
first column zero, one. So R S is not equal to S R. With numbers, a times b always equals b
times a. With matrices, in general, A B is not equal to B A.""",

# ── C5: transpose & identity ──
"tr1": """Two more tools finish the toolbox. The first is the transpose. Take a matrix and flip
it over its diagonal, so its rows become its columns. We write a small T at the top right,
and we say A transpose.""",

"tr2": """Here is a two by three matrix, with rows one, two, three and four, five, six. Its
transpose is three by two. The first row, one, two, three, becomes the first column. The
second row becomes the second column. The shape flips: m by n becomes n by m.""",

"tr3": """Careful with a common mistake. The transpose is not the same as pouring the numbers, in
reading order, into a new shape. That gives one, two, then three, four, then five, six:
wrong. In the transpose, the entry in row i, column j comes from row j, column i.""",

"id1": """The second tool is the identity matrix. It has ones on the diagonal and zeros
everywhere else. Its columns are e one and e two, so it leaves e one and e two where they
are. So the identity matrix does nothing. I times x equals x, for every x.""",

"id2": """It is the matrix version of the number one, because one times five equals five. Multiply
any matrix by the identity, on either side, and you get it back: A times I equals A, and I
times A equals A.""",

"axiom": """Where does all of this come from? A matrix times a vector is a linear combination of
its columns. Matrix multiplication is one such machine after another. So everything in this
video is built from adding and scaling. That is axiom A, linearity.""",

# ── C6: ML ──
"ml1": """Now let's put matrices to work. Take our three houses again, as a three by three
matrix X. Suppose a toy model gives each house a score with three weights: two, ten, and minus
one. We put the weights into a vector w. Then X times w scores all three houses at once.""",

"ml2": """Row one: two times one hundred twenty, plus ten times three, minus fifteen. That is two
hundred forty plus thirty minus fifteen, which is two hundred fifty-five. The other rows give
one hundred fifty and three hundred thirty-five. One matrix product gives three predictions.
This is why data is kept in matrices.""",

"ml3": """A layer in a neural network works the same way. A layer with three inputs and two
outputs has a two by three matrix of weights, W. The layer multiplies the input vector x by
this matrix. Each output is one row of weights combined with the input.""",

"ml4": """Real models are much bigger. In the book Build a Large Language Model from Scratch, one
example layer has a weight matrix of two thousand forty-eight by two thousand forty-eight.
That is more than four million numbers in a single matrix.""",

# ── C7: practice ──
"prac1": """Now it is your turn. Pause the video and try this. Multiply the matrix with rows one,
two and zero, one by the vector three, one. Use the column view or the row view. Both give
the same answer.""",

"prac2": """Here is the answer with the row view. Row one: one times three plus two times one is
five. Row two: zero times three plus one times one is one. So the answer is five, one. With
the column view, three times one, zero plus one times two, one also gives five, one.""",

"prac3": """Second question, about shapes. A is a two by three matrix, and B is a three by four
matrix. Is A B allowed, and what is its shape? Is B A allowed? Pause and use the shape
rule.""",

"prac4": """A B is allowed, because the inner numbers are three and three. The result is two by
four. B A is not allowed. B is three by four and A is two by three, so the inner numbers are
four and two. They do not match. Even the shape can depend on the order.""",

# ── C7: code ──
"code1": """Let's check everything in Python with NumPy, where the at sign multiplies matrices. We
compute A times x, the stretch and the rotation, both orders of the product, the transpose,
and the identity check.""",

"code2": """This output is real; the script ran while the video was made. A times x is four,
seven. The stretch gives two, two, and the rotation gives minus
two, one. R S and S R are different. And X times w scores the houses: two hundred fifty-five,
one hundred fifty, and three hundred thirty-five.""",

# ── C8: summary ──
"sum1": """In summary. A matrix is a rectangular grid of numbers, with m rows and n columns. As a
table, each row is a data point. As a machine, it moves vectors: A times x is a linear
combination of the columns, and the columns show where e one and e two land.""",

"sum2": """Matrix multiplication is one machine after another. An m by n matrix times an n by p
matrix gives m by p, and the order matters. The transpose swaps rows and columns, and the
identity does nothing. Next, in video three: basis and rank, how many directions are really
there. See you there.""",
}
