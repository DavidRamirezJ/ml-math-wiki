# 06-norm — transcript


## C0_Intro

How big is an error? How far apart are two words inside a language model? Is one training step too large? Machine learning asks questions like these all the time. Each answer is a single number that measures the size of a vector. That number is called the norm, and it is the topic of this video.

You will learn to compute the length of any vector, in two dimensions, in three, or in a million. You will meet three different ways to measure size, the rules that every norm must follow, and how distance comes from length. You need video one, on vectors, and video five, on the inner product.


## C1_Length

Let's start with the simplest question. How long is this blue arrow? It goes three steps to the right and four steps up. The two steps and the arrow make a right triangle. The arrow is the long side of that triangle.

Pythagoras tells us the answer. The long side squared equals three squared plus four squared. Three squared is nine. Four squared is sixteen. Nine plus sixteen is twenty five. The square root of twenty five is five. So the arrow has length five.

Now look at the sum we computed: three times three, plus four times four. That is the dot product of the vector with itself, from video five. So the length is the square root of x dot x. This length is called the norm of x. We write it with double bars around x.

Let's read the formula symbol by symbol. The double bars mean the norm of x. Inside the square root, we square each component and add them all. The small letter i counts the components, from one up to n. In two dimensions this is exactly Pythagoras. But the same recipe works in any number of dimensions.

Let's try three dimensions. Take x equal to two, three, six. Square each component: four, nine, and thirty six. Add them: forty nine. The square root of forty nine is seven. So this arrow in space has length seven. With a million components, a computer does exactly the same steps.

Where does the norm come from? It comes from the inner product, which is axiom B in this series. The inner product gives a vector space lengths and angles. The norm is the length part. This standard norm has two more names: the L two norm, and the Euclidean norm.


## C2_OtherNorms

But is the straight line the only sensible way to measure size? Imagine a taxi in a city with a grid of streets. It cannot drive through the buildings. To reach the point three, four, it drives three blocks right and four blocks up. That is seven blocks, not five.

This gives the L one norm, also called the Manhattan norm. Add the absolute values of the components. The absolute value removes the minus sign, so minus three counts as three. For the vector three, four, the L one norm is three plus four, which is seven.

A third way is to keep only the biggest component. This is the L infinity norm, also called the max norm. It takes the largest absolute value. For three, four, that is four. So the same vector has three sizes: seven, five, and four. They measure different things, and all three are real norms.

Here is a beautiful way to see the difference. Draw every vector whose norm is exactly one. For the L two norm, you get the familiar circle. For the L one norm, you get a diamond. For the L infinity norm, you get a square. Each norm has its own unit circle.

One honest note. Only the L two norm comes from an inner product, as the square root of x dot x. The L one and L infinity norms are real norms, but no inner product gives them. This is a known fact, proved with a test called the parallelogram law. So these norms give sizes, but no angles.


## C3_Rules

What makes a rule a norm? Every norm must follow three rules. Rule one: the norm is never negative. And it is zero only for the zero vector. Any arrow that really goes somewhere has a positive length.

Rule two is about scaling. Multiply a vector by a number c, and its norm is multiplied by the absolute value of c. Watch: two times the vector doubles the length. Minus two times the vector flips the arrow, but the length is still doubled. Minus two times three, four is minus six, minus eight, with norm ten.

Rule three is the triangle inequality. Walk along u, and then along v. Or walk straight to u plus v. The straight route is never longer than the detour. In symbols, the norm of u plus v is at most the norm of u plus the norm of v.

Let's check it with numbers. Take u equal to three, four, and v equal to four, minus three. Each has norm five, so the detour is ten. Their sum is seven, one. Its norm is the square root of fifty, about seven point zero seven. That is less than ten, as promised.


## C4_Unit

Sometimes we care only about direction, not about length. Then we use a unit vector: a vector whose norm is exactly one. Every unit vector in the plane has its tip on the unit circle that we drew before.

To turn any vector into a unit vector, divide it by its own norm. This is called normalizing. Take x equal to three, four. Its norm is five. Divide each component by five: zero point six and zero point eight. The direction stays the same. Only the length changes.

Let's check that the new length is one. Zero point six squared is zero point three six. Zero point eight squared is zero point six four. The sum is exactly one, and the square root of one is one. It works because of rule two: dividing by five divides the length by five.

Be careful with one case. The zero vector has norm zero, and we cannot divide by zero. So the zero vector has no direction, and it cannot be normalized. Normalizing is everywhere in machine learning. For example, cosine similarity from video five is the dot product of two normalized vectors.


## C5_Distance

Now, how far apart are two points? Take p equal to one, one, and q equal to four, five. The arrow from p to q is q minus p, which is three, four. The distance is the length of that arrow: five. So distance is simply the norm of the difference.

Here is the formula. The distance d between x and y is the norm of x minus y. In three dimensions, take x equal to one, two, three, and y equal to minus one, minus one, zero. The difference is two, three, three. The squares are four, nine, and nine. The distance is the square root of twenty two, about four point six nine.

A distance rule like this is called a metric. Every metric follows three rules. The distance is never negative, and it is zero only when the two points are the same. The distance from x to y equals the distance from y to x. And a detour through a third point is never shorter.

These rules come straight from the three norm rules. Notice one more thing. Distance and the inner product move in opposite directions. Two very similar vectors have a large inner product, but a small distance. Keep this in mind when you read about similarity search.


## C6_ML

Now let's see norms at work in machine learning. First, regularization. Very large weights often make a model fit the noise in its training data. So we add a penalty to the loss, the error score that we study in video fifteen. The penalty is lambda times the squared L two norm of the weights. Now the model prefers small weights.

If we use the L one norm instead, something special happens. The L one penalty pushes many weights to exactly zero, so the model uses only a few inputs. This is called sparsity. Look at the diamond: its sharp corners sit on the axes, where one weight is zero. Ridge regression uses the L two penalty, and lasso uses the L one penalty.

Second, gradient clipping. During training, the gradient tells the model how to change its weights. We will study it in video eight. Sometimes the gradient becomes huge, and one bad step can ruin the training. So we measure its L two norm. If the norm is larger than a maximum, we scale the gradient down.

Here is a toy example. Say the gradient is six, eight, with norm ten, and the maximum norm is five. We multiply by five divided by ten, which is one half. The new gradient is three, four, and its norm is exactly five. The direction is kept. Only the length is capped. This is a standard trick when training large language models.

Third, nearest neighbour search. Here are three toy data points and a new query point. Which one is closest? Compute the distance to each. A is about three point one six away. B is about two point two four. C is about two point eight three. So B is the nearest neighbour. Search over word vectors works the same way, with millions of points.


## C7_Practice

Now it is your turn. Pause the video and try this. Take the vector minus five, twelve. What is its L two norm? And what is its L one norm? Remember: for L two, square, add, and take the square root. For L one, add the absolute values.

Here are the answers. Minus five squared is twenty five. Twelve squared is one hundred forty four. The sum is one hundred sixty nine, and its square root is thirteen. For L one, five plus twelve is seventeen. The taxi route is never shorter than the straight line.

One more question. What is the distance between the points two, seven and five, three? Then, normalize the difference vector, so that its length is one. Pause and think.

First subtract. Two minus five is minus three, and seven minus three is four. The norm of minus three, four is five, so the distance is five. To normalize, divide by five. You get minus zero point six, and zero point eight. Its length is one.


## C7_Code

Let's check everything in Python with NumPy. The function norm, from the linear algebra part of NumPy, computes norms. The argument ord chooses which one: one, two, or infinity. We also normalize a vector, compute a distance, clip a gradient, and find a nearest neighbour.

This output is real; the script ran while the video was made. Three, four has norms seven, five, and four. Two, three, six has norm seven. The unit vector is zero point six, zero point eight. The distance is four point six nine. The clipped gradient is three, four, and the nearest neighbour is B.


## C8_Summary

In summary. The norm measures the size of a vector as one number. The standard L two norm is the square root of x dot x: Pythagoras in any dimension. The L one norm adds absolute values, and the L infinity norm takes the largest one. Every norm is positive, scales with the absolute value of c, and obeys the triangle inequality.

Divide by the norm to get a unit vector. The norm of a difference is a distance. In machine learning, norms keep weights small, cap gradients, and find nearest neighbours. Next, in video seven, we meet the derivative: the slope of a curve right here, at one point. See you there.


## C9_Credits

This video was made by David R. You can find more lessons at David Ramirez J dot com. Thanks for watching.


## New terms in this video

- **norm** — A rule that gives the size (length) of a vector as one number that is never negative.
- **unit vector** — A vector whose norm is exactly 1. It keeps only a direction.
- **distance** — How far apart two vectors are: the norm of their difference, x − y.
