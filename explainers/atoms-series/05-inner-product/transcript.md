# 05-inner-product — transcript


## C0_Intro

How does a language model know that the word cat is closer to dog than to car? Every word is a vector, as we saw in video one. To compare two vectors, the model computes a single number that says how much they point the same way. That number is the inner product, and it is the engine inside attention.

In this video you will compute the dot product by hand, see what it means as a picture, use its sign to detect right angles, and measure similarity with the cosine. You need video one on vectors and video two on matrices, especially the transpose.


## C1_Dot

Here is the recipe. Take two vectors with the same number of components. Multiply the matching components, first with first, second with second, and so on. Then add up all the products. The result is one number, not a vector. This is called the dot product.

In symbols: x dot y equals the sum over i of x i times y i. The letter i runs over the components, from one to n. In matrix language we also write x transpose y: a row times a column, which gives a one by one matrix, just a number.

Example one, with three components. Let x be one, two, three, and y be four, minus five, six. First components: one times four is four. Second: two times minus five is minus ten. Third: three times six is eighteen. Add them: four minus ten plus eighteen is twelve. So x dot y is twelve.

Example two, in the plane. Take one, one and one, two. One times one plus one times two is three. Notice something familiar. In video two, the row view of a matrix times a vector was exactly this: multiply matching entries and add. Every entry of a matrix product is a dot product.


## C2_Geometry

What does this number mean? Here is the picture. Let y lie along the horizontal axis, y equals three, zero. Let x be two, one. Shine a light from above, and x casts a shadow on the line of y. The shadow has length two.

Now compute x dot y: two times three plus one times zero is six. And the length of y times the length of the shadow is three times two, also six. That is the meaning: the dot product is the length of one vector times the shadow of the other on it. This shadow is called a projection.

This gives a famous formula. x dot y equals the length of x, times the length of y, times the cosine of the angle theta between them. The cosine is one when the angle is zero, and zero for a right angle. Here, length means the usual length of the arrow. Video six makes that idea exact.

Let's use the formula with one, one and one, two. Their dot product is three. By Pythagoras, the length of one, one is the square root of two, and the length of one, two is the square root of five. So the cosine of theta is three divided by the square root of ten, about zero point nine five.

That means the angle is about eighteen degrees. The two arrows point almost the same way, and the dot product told us that without any protractor. From the dot product we get angles, and in video six we will get lengths from it too.


## C3_Sign

The sign of the dot product tells you a lot. Watch the green vector turn while the blue vector stays fixed. When they point roughly the same way, the dot product is positive. As the angle grows, the value shrinks.

At a right angle, the dot product is exactly zero: the shadow has no length. Turn further, and the value becomes negative: the vectors point in opposite directions. When they point exactly opposite, the dot product is as negative as it can be.

Vectors whose dot product is zero are called orthogonal. It is the general word for perpendicular, at a right angle. Check one, one and minus one, one: one times minus one plus one times one is zero. So they are orthogonal. And the zero vector is orthogonal to every vector.

Orthogonal directions share nothing. If you walk purely north, you make no progress east. In machine learning, orthogonal features carry separate information, with no overlap. Keep this in mind for later videos about data and directions.


## C4_Cosine

Often we care only about direction, not about size. A long arrow and a short arrow that point the same way should count as equally similar. So we divide the dot product by the two lengths. The result is called cosine similarity.

Cosine similarity is always between minus one and one. One means the same direction. Zero means orthogonal. Minus one means opposite. It is exactly the cosine of the angle between the two vectors.

Here is a toy example. These are not real word vectors, just small numbers chosen to show the idea. Cat is zero point nine, zero point eight, zero point one. Dog is zero point eight, zero point nine, zero point two. Car is zero point one, zero point two, zero point nine.

The cosine similarity of cat and dog is about zero point nine nine: almost the same direction. The cosine similarity of cat and car is only about zero point three. So, in this toy space, cat is much closer to dog than to car. Real embeddings work the same way, with hundreds of components.


## C5_AxiomB

So far we used the dot product. Mathematicians write a more general inner product with angle brackets: x, y inside brackets. Any rule that pairs two vectors into one number can be an inner product, if it follows three rules.

Rule one: it is symmetric. The order does not matter: x with y equals y with x. Rule two: it is linear in each input. Scale or add inside one slot, and the result scales or adds the same way. Rule three: it is positive. A vector paired with itself gives a positive number, unless the vector is zero.

Check rule three with the dot product. One, two, three dotted with itself is one plus four plus nine, which is fourteen. Squares are never negative, so the sum is positive for any non-zero vector. The dot product follows all three rules.

Why does this matter? A vector space only knows how to add and scale. It has no lengths and no angles. The inner product adds them. This is our second axiom, axiom B: the inner product, the source of all geometry in this series. There are other inner products too, and they give different angles.


## C6_ML

Now the inner product at work. In attention, each word asks every other word: how related are we? The answer is a dot product, called an attention score. In the book Build a Large Language Model from Scratch, the word Your has the vector zero point four three, zero point one five, zero point eight nine.

The word journey has the vector zero point five five, zero point eight seven, zero point six six. Their dot product is about zero point nine five four four. A bigger score means the two words are more related, so attention mixes in more of that word.

Similarity search works the same way. To find documents or products like a query, a system compares the query vector with many stored vectors and keeps the ones with the largest cosine similarity. And a single neuron computes w dot x, the weights dotted with the input: a weighted sum is a dot product.


## C7_Practice

Your turn. Pause the video and try this. Compute the dot product of two, minus one, three and one, four, two.

Two times one is two. Minus one times four is minus four. Three times two is six. Add them: two minus four plus six is four. So the dot product is four. It is positive, so the vectors point roughly the same way.

Second question. Are the vectors three, six and minus two, one orthogonal? Pause and check.

Three times minus two is minus six. Six times one is six. Minus six plus six is zero. The dot product is zero, so yes, they are orthogonal. They meet at a right angle.


## C7_Code

Let's check it in NumPy. The at sign gives the dot product of two vectors. We compute our examples, the angle, the orthogonal pair, the rule checks, the toy cosine similarities, and the attention score from the book.

This output is real; the script ran while the video was made. x dot y is twelve. The angle is about eighteen point four degrees. The orthogonal pair gives zero. The toy cosine similarities are zero point nine nine and zero point three, and the book's attention score is zero point nine five four four.


## C8_Summary

In summary. The dot product multiplies matching components and adds them, giving one number. It equals the length of x times the length of y times the cosine of the angle. Its sign says same direction, right angle, or opposite. Zero means orthogonal.

Cosine similarity divides by the lengths and keeps only direction. An inner product is symmetric, linear in each input, and positive: axiom B, the source of geometry. Next, in video six: the norm, which measures length and distance. See you there.


## C9_Credits

This video was made by David R. You can find more lessons at David Ramirez J dot com. Thanks for watching.


## New terms in this video

- **dot product** — Multiply matching components and add: the result is one number.
- **orthogonal** — Two vectors are orthogonal when their inner product is zero: they meet at a right angle.
- **cosine similarity** — The dot product divided by the two lengths: the cosine of the angle, from −1 to 1.
- **inner product** — A rule ⟨x, y⟩ that turns two vectors into one number. It is symmetric, linear in each input, and positive.
