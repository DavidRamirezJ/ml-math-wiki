# 01-vector — transcript


## C0_Intro

Every photo, every word, every song, and every customer that a machine learning model sees becomes a list of numbers first. A photo becomes its pixel brightness values. A word becomes a few hundred learned numbers. A customer becomes age, income, and number of purchases. That list of numbers has a name. It is called a vector.

In this video you will learn the two moves that vectors allow: adding and scaling. They look simple, and they are. But every other idea in this series, from matrices to gradients to attention, is built from these two moves. You only need arithmetic to follow. Let's discover it step by step.


## C1_ThreeFaces

A vector has three faces, and it helps to know all three. The first face is an arrow. Here is a grid. Start at the center, the origin. Walk three steps to the right and two steps up. The arrow from the origin to that point is a vector.

The second face is a list. We can write the same arrow as two numbers: three and two. Each number is called a component. The first component says how far to go right. The second component says how far to go up. The arrow and the list carry exactly the same information.

Lists are how data looks in practice. Take a house. It has one hundred twenty square meters, three bedrooms, and it is fifteen years old. That is a vector with three components. We cannot draw three-dimensional houses as flat arrows easily, and with a hundred features we cannot draw them at all. But the list still works perfectly.

The third face is the deepest one. Mathematicians say: a vector is anything you can add to another thing of its kind, and stretch by a number, and still get the same kind of thing back. Arrows do that. Lists do that. Later we will see that even sound waves do that. Keep these two moves in mind: add and scale.


## C2_Add

Let's start with adding. Here is a blue vector u, equal to three, one. Here is a green vector v, equal to one, two. What should u plus v mean? Think of it as two walks. First walk along u. Then, from where you stopped, walk along v.

Put the tail of v on the tip of u. The yellow arrow from the origin to the final point is u plus v. Where did we end up? Three plus one is four steps right. One plus two is three steps up. So u plus v equals four, three.

Notice the rule. To add two vectors, you add them component by component. First with first, second with second. Nothing more. The arrow picture and the list rule always agree. That is why we can trust the list rule even in a hundred dimensions, where no picture exists.

Does the order matter? Walk v first, then u. You trace the other side of this shape, a parallelogram, and you arrive at the same point. So u plus v equals v plus u. This is our first rule of vectors: the order of adding does not matter.


## C3_Scale

Now the second move: scaling. We multiply a vector by an ordinary number. In linear algebra, an ordinary single number is called a scalar, because it scales things. Take u equal to two, one. Two times u is four, two. The arrow keeps its direction and gets twice as long.

Watch the scalar c change. When c is bigger than one, the arrow stretches. When c is between zero and one, it shrinks. At zero, it collapses to the zero vector, a point. And when c is negative, the arrow flips around and points the opposite way.

The rule is again component by component. Multiply every component by the scalar. Minus one times two, one is minus two, minus one. One half times two, one is one, one half. Every multiple of u lies on the same straight line through the origin.


## C4_Combine

Now combine the two moves. Scale some vectors, then add them. The result is called a linear combination. With two vectors u and v, a linear combination looks like a times u plus b times v, where a and b are scalars. This single pattern is the heart of linear algebra.

Here is a puzzle. Let u be two, one, and v be one, two. Can we reach the yellow point five, four? Try a equals two and b equals one. Two times u is four, two. Add v, which is one, two. Four plus one is five. Two plus two is four. Yes, we reached five, four.

In fact, with these two vectors we can reach every point on the whole plane. Just choose the right a and b. All the points you can reach with linear combinations are called the span. Here, the span of u and v is the entire plane.

But be careful. If the two vectors point along the same line, like two, one and four, two, the span is only that line. The second vector adds nothing new. This idea of redundant vectors is so important that it gets its own video: basis and rank, video three.


## C5_Rules

So what is a vector, really? Here is the precise answer. A vector space is a collection of objects where two things are true. First, adding two members gives another member. Second, scaling a member gives another member. You never fall out of the collection.

The adding and scaling must also follow a few common-sense rules. Order does not matter: u plus v equals v plus u. There is a zero vector that changes nothing. Every vector has an opposite that cancels it. And scaling spreads over a sum: a times u plus v equals a u plus a v.

Here is the surprise. Functions follow these rules too. Take two sound waves. Add them, point by point, and you get a new sound wave. Make one louder by multiplying it by two, and it is still a sound wave. So sound waves are vectors, even though they are not arrows.

That is why the definition is so powerful. Anything we prove using only add and scale works for arrows, for lists of data, for sound, and for images. In this series, this is our first axiom, axiom A: linearity. Everything else is built on top of it.


## C6_ML

Now let's see vectors at work in machine learning. Here are two tiny toy images, just two by two pixels. Each pixel is a brightness between zero and one. Write the four pixels in a row, and each image becomes a vector with four components.

What is one half of the first image plus one half of the second? It is a linear combination. Each pixel becomes the average of the two pixels. The result is a blend of the two pictures. Mixing images, mixing sounds, averaging data: all of them are linear combinations.

And here is the most important example. A neuron in a neural network takes an input vector, multiplies each component by a weight, and adds everything up. That weighted sum is a linear combination too. Billions of these sums, stacked together, are what a large language model computes.


## C7_Practice

Now it is your turn. Pause the video and try this. What is three times the vector one, minus two, plus the vector two, five? Use the two rules: scale component by component, then add component by component. Take a few seconds.

Here is the answer. Three times one, minus two, is three, minus six. Now add two, five. Three plus two is five. Minus six plus five is minus one. So the answer is five, minus one. If you got it, you already know how to compute any linear combination.

One more question, about span. Use only one vector, two, one. Can you reach the point six, three? And can you reach the point one, one? Remember, with one vector, a linear combination is just c times that vector. Pause and think.

Six, three is three times two, one, so yes, it is in the span. But one, one is not a multiple of two, one. To get the first component one, c must be one half. Then the second component is one half, not one. So one, one is outside the span: one vector only reaches a line.


## C7_Code

Let's check everything in Python with NumPy, a library for working with arrays of numbers. We make u and v, then add them, scale them, and build the linear combination two u plus v. Then we ask NumPy to find a and b for the point five, four. Finally, we blend two toy images.

This output is real; the script ran while the video was made. u plus v is four, three. Two times u is four, two. Two u plus v is five, four, and NumPy found a equals two and b equals one, just like our puzzle. The blended image is the pixel-by-pixel average.


## C8_Summary

In summary. A vector is anything you can add and scale. In practice, it is a list of numbers, and in two dimensions you can draw it as an arrow. You add and scale component by component. Scale and then add, and you get a linear combination. All the reachable points form the span.

In machine learning, every data point is a vector, and every neuron computes a linear combination. Next, in video two, we stack many vectors side by side and get a matrix: a grid of numbers that is also a machine that moves vectors. See you there.


## C9_Credits

This video was made by David R. You can find more lessons at David Ramirez J dot com. Thanks for watching.


## New terms in this video

- **component** — One entry of a vector: the number in one position of the list.
- **vector** — Anything you can add to another of its kind and scale by a number, and still get the same kind of thing.
- **scalar** — An ordinary single number, used to stretch, shrink or flip a vector.
- **linear combination** — Scale some vectors, then add them: a·u + b·v.
- **span** — All the vectors you can reach with linear combinations of a set of vectors.
- **vector space** — A collection where adding and scaling never take you outside it, and the usual rules hold.
