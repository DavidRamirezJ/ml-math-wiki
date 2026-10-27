# 04-linear-mapping — transcript


## C0_Intro

Every layer of a neural network starts with the same step. It takes a vector in, and it gives a new vector out, in a very special way: it respects adding and scaling. Functions like this are called linear mappings. If you understand them, you know exactly what a layer can do, and also what it can never do on its own.

In this video you will learn the definition of a linear mapping, see what it does to a grid, test functions that are not linear, and find the matrix of any linear map from just two arrows. You need video one on vectors, video two on matrices, and video three on bases.


## C1_Definition

A function takes an input and gives an output. Here the input is a vector, and the output is a vector too. We write f of x. The function f is linear if it respects the two moves from video one: adding and scaling.

Here is the definition. For all vectors u and v, and all numbers a and b, f of a u plus b v equals a times f of u plus b times f of v. In words: combine first and then apply f, or apply f first and then combine. You must get the same result.

Read it symbol by symbol. The letters u and v are input vectors. The letters a and b are scalars, the weights. On the left, we apply f to the linear combination. On the right, we combine the outputs, f of u and f of v, with the same weights.

It is the same as two simpler rules. First, f of u plus v equals f of u plus f of v. Second, f of c times u equals c times f of u. Here is a tiny example with numbers. Let f of x be three x. Then f of two plus five is twenty-one, and f of two plus f of five is six plus fifteen, also twenty-one.

One consequence is very useful. A linear map always sends zero to zero. Why? Take the scalar zero. Then f of zero times u equals zero times f of u, which is the zero vector. So if a function moves the origin, it cannot be linear.


## C2_Pictures

What does a linear map look like? Watch a grid. After a linear map, the grid lines stay straight. Parallel lines stay parallel. The spacing stays even. And the origin stays where it is. That is the picture of linearity.

Here are four examples. First, a stretch, which we met in video two: everything becomes twice as wide. Second, a rotation by ninety degrees. Third, a shear. It slides each horizontal line sideways, more and more the higher it is, like pushing the top of a deck of cards.

Fourth, a projection onto a line. It drops every point straight down onto the horizontal axis. The whole plane collapses onto one line. That may look extreme, but the rules still hold: the lines stay straight and evenly spaced, the origin does not move, and adding and scaling are respected.


## C3_NonExamples

Now some functions that are not linear. First, f of x equals x plus one. It shifts every number one step to the right. Check the origin: f of zero is one, not zero. The origin moved, so this function is not linear.

Check the rule as well. f of two plus three is f of five, which is six. But f of two plus f of three is three plus four, which is seven. Six is not seven. A shift like this is called an affine map: a linear map followed by a move. It keeps lines straight, but it moves the origin.

Second, f of x equals x squared. f of one plus one is f of two, which is four. But f of one plus f of one is one plus one, which is two. Four is not two. Scaling fails too: f of two times three is thirty-six, but two times f of three is only eighteen.

Look at the graphs. A linear function of one number is a straight line through the origin, like two x. The shift, x plus one, is straight but misses the origin. The square, x squared, is curved. Only the first one is linear.


## C4_Matrix

Now the key theorem of this video. Every linear map between spaces like these can be written as a matrix. And the matrix is easy to find: its columns are where the basis vectors land. The first column is f of e one. The second column is f of e two.

Why is this true? Any vector x is x one times e one plus x two times e two. Because f is linear, f of x equals x one times f of e one plus x two times f of e two. That is a linear combination of two fixed vectors, and from video two we know: that is a matrix times x.

Let's use it. Suppose a linear map sends e one to two, one, and sends e two to minus one, one. Then its matrix A has the first column two, one, and the second column minus one, one. Two arrows were enough to know the whole map.

Where does the vector three, two go? Use the columns: three times two, one plus two times minus one, one. That is six minus two, which is four, and three plus two, which is five. So f of three, two is four, five. The row view gives the same answer.

A second example: the rotation by ninety degrees. It sends e one to zero, one, and e two to minus one, zero. So its matrix has the columns zero, one and minus one, zero. That is exactly the rotation matrix from video two. Now you know where it came from.


## C5_Compose

What if we apply two linear maps, one after the other? The result is linear too, and its matrix is the product of the two matrices. First the shear S, then the projection P. The combined map has the matrix P times S, with rows one, one and zero, zero.

Check it with the vector one, two. The shear sends it to three, two. The projection then sends that to three, zero. And P times S applied to one, two also gives three, zero. One matrix does both steps.

Two more words. The kernel of a linear map is the set of all inputs that it sends to zero. Another name is the null space. For the projection, every point on the vertical axis, like zero, five, drops onto the origin. So the kernel of the projection is the vertical axis.

The image of a map is the set of all its outputs. Another name is the range. The projection only produces points on the horizontal axis, so its image is that axis. The dimension of the image is the rank of the matrix, from video three. Here the rank is one.

There is a simple balance. The dimension of the kernel plus the dimension of the image equals the dimension of the input space. For the projection, one plus one equals two. What the map destroys, plus what it keeps, adds up to everything.


## C6_ML

Now, machine learning. A linear layer in PyTorch computes, in our notation, W times x plus b. W is the weight matrix, and b is the bias vector, a fixed vector that is added at the end. W times x is a linear map. Adding b moves the origin, so the whole layer is affine, not strictly linear.

Here is the big consequence. Two linear maps in a row are just one linear map. Take two toy layers, W one and W two. W two times W one times x equals the single matrix W two times W one, times x. A stack of ten linear layers can still only do what one matrix can do.

GPT-2, for example, has a block that expands each vector of seven hundred sixty-eight numbers to three thousand seventy-two numbers, and then brings it back. Without something in between, those two maps would collapse into one single map. That is why networks put a nonlinear activation function between the layers. It gets its own video, video sixteen.

Where does all this come from? The definition of a linear map is exactly axiom A: it respects adding and scaling. Matrices, composition, kernel and image all follow from that one rule.


## C7_Practice

Your turn. Pause the video and try this. The map f sends x, y to two x, x plus y. Is it linear? If yes, find its matrix. Hint: look at where e one and e two go.

It is linear. f of e one is f of one, zero, which is two, one. f of e two is f of zero, one, which is zero, one. So the matrix has the columns two, one and zero, one. Check with three, one: six, four from the formula, and also six, four from the matrix.

Second question. Is the function g of x equals five x minus three linear? Pause and use the fastest test you know.

The fastest test is the origin. g of zero is minus three, not zero. So g is not linear. It is affine: the linear map five x, followed by a shift of minus three. Its graph is a straight line, but the line does not pass through the origin.


## C7_Code

Let's check it in NumPy. We build the matrix from where e one and e two go, apply it, and test the linear rule with two vectors. We test the shift and the square, compose the shear and the projection, and collapse two toy layers into one matrix.

This output is real; the script ran while the video was made. A times three, two is four, five. Both sides of the rule give twenty-one, twelve. The shift gives six versus seven, and the square gives four versus two. The two orders of composition agree, and the two layers equal one matrix.


## C8_Summary

In summary. A linear mapping respects adding and scaling: f of a u plus b v equals a f of u plus b f of v. It keeps grid lines straight, parallel and evenly spaced, and it keeps the origin fixed. A shift is affine, and a square is not linear at all.

Every linear map is a matrix whose columns are the images of the basis vectors. Composition is the matrix product. The kernel goes to zero, and the image is all outputs. Next, in video five: the inner product, one number that measures similarity. See you there.


## C9_Credits

This video was made by David R. You can find more lessons at David Ramirez J dot com. Thanks for watching.


## New terms in this video

- **linear mapping** — A function f with f(a·u + b·v) = a·f(u) + b·f(v): it respects adding and scaling.
- **kernel (null space)** — All inputs that the map sends to the zero vector.
- **image (range)** — All the outputs that the map can produce.
