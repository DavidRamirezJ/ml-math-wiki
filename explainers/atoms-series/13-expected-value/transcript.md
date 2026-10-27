# 13-expected-value — transcript


## C0_Intro

A casino does not know who will win tonight. But it knows, almost exactly, how much money it will make over a million games. A machine learning model does not know its error on the next example. But we can talk about its average error over all examples. That long-run average has a name: the expected value.

In this video you will learn to compute an expected value, see why a fair die expects three point five, check a lottery, and discover a shortcut called linearity. Then we watch the average of real simulated rolls settle down. You need video twelve, the probability distribution. This atom is built from axiom A, linearity, and axiom D, the probability rules.


## C1_Weighted

Start with something familiar: a grade. Your homework score is eighty, and your exam score is ninety. But the exam counts more. Homework has a weight of thirty percent, the exam seventy percent. A plain average would give eighty five. That is not fair to the weights.

Instead, multiply each score by its weight and add. Zero point three times eighty is twenty four. Zero point seven times ninety is sixty three. Twenty four plus sixty three is eighty seven. This is a weighted average. The weights add up to one, and big weights pull the result toward their score.

Now the key idea. A probability distribution is a list of weights that add up to one. So we can use the probabilities as weights. Each value of the random variable is weighted by how likely it is. That weighted average is the expected value.


## C2_Definition

Here is the formula. The expected value of X, written E of X, equals the sum over all values x of p of x times x. Read it piece by piece. The values x are what can happen. p of x is the weight of each value. Multiply, then add everything up.

First example: one fair die. The values are one to six, each with probability one sixth. So E of X is one sixth times one, plus one sixth times two, and so on up to six. The sum of one to six is twenty one. Twenty one over six is three point five.

Notice something strange. You can never roll three point five. The expected value does not have to be a possible value. It is a balance point. If the bars of the distribution were weights on a ruler, the ruler would balance exactly at three point five.

Second example: a toy lottery. A ticket costs two euros. With probability zero point one you win ten euros, so your net win is eight. With probability zero point nine you win nothing, so your net win is minus two. E of the win is zero point one times eight, plus zero point nine times minus two.

That is zero point eight minus one point eight, which is minus one euro. On average, every ticket loses one euro. One single ticket can still win. But over many tickets, you lose about one euro per ticket. The expected value tells you the long-run truth, not the result of one game.


## C3_Linearity

Now a powerful property. The expected value is linear. E of a X plus b Y equals a times E of X, plus b times E of Y. Here a and b are ordinary numbers. This is axiom A again: the expected value respects adding and scaling, just like a linear mapping.

Why is it true? E of X is a sum of probabilities times values. Scaling every value by a scales the whole sum by a. And the expected value of a sum splits into two sums. The sum rule from video twelve does the bookkeeping. Linearity holds even when X and Y are not independent.

Here is the shortcut. Roll two dice. What is the expected sum? With the table, you would add thirty six products. With linearity, E of X plus Y is E of X plus E of Y. That is three point five plus three point five, which is seven. No table needed.

One more example with numbers. Take a die X, and make a new score: two times X plus one. Its expected value is two times three point five, plus one, which is eight. Be careful: linearity works for adding and scaling. For a curved function, like X squared, the expected value of X squared is not E of X, squared.


## C4_LongRun

What does long-run average mean in practice? Let's roll a die many times in a computer, with a fixed seed, and keep a running average. The first ten rolls are five, one, two, two, two, five, six, four, one, one. Their average is only two point nine.

Keep rolling. Watch the running average move. After one hundred rolls it is three point four six. After one thousand, three point four five three. After ten thousand, three point four eight eight three. The wild jumps at the start calm down, and the line stays close to three point five.

This is the law of large numbers. The average of many independent samples gets close to the expected value, as the number of samples grows. It never promises an exact match. It promises that big errors become rare. This is why a casino can plan, and why averages over data are reliable.


## C5_DataMean

Now look at a data set: two, four, four, five, ten. Its mean is the sum divided by the count. Twenty five over five is five. The mean of a data set is the everyday average.

Here is the connection. Pick one row of the data at random, each row with probability one fifth. That defines a random variable. Its expected value is one fifth times each value, added up, which is again five. So the mean of a data set is an expected value, under the pick a row at random distribution.

The value four appears twice, so it gets the weight two fifths. Big values pull the mean up: the ten moves it more than any other row. In machine learning, we often write E of X for the true long-run average, and x bar for the mean of our samples, our best estimate of it.


## C6_ML

Now the most important use in machine learning. For every training example, the model makes an error, measured by a number called the loss. We will study losses in video fifteen. Training does not try to fix one example. It tries to make the average loss over all examples small.

Take four toy examples with losses zero point two, zero point five, zero point one, and one point two. The average loss is two over four, which is zero point five. This average is an expected value under the pick a row at random distribution. It is our estimate of the expected loss on new data.

And because expectation is linear, we can split the average loss into pieces, estimate it from small random batches, and take derivatives inside the sum. Most of modern training is: estimate an expected value from samples, then improve the model to make it smaller.


## C7_Practice

Your turn. A random variable takes the value zero with probability zero point five, the value two with probability zero point three, and the value ten with probability zero point two. What is its expected value? Pause and try.

Multiply and add. Zero point five times zero is zero. Zero point three times two is zero point six. Zero point two times ten is two. The sum is two point six. The rare value ten pulls the expected value up a lot.

Second question. Roll three dice. What is the expected sum? And what is the expected value of three times one die, minus two? Use linearity. Pause and try.

For three dice, add three expected values: three point five, three times, which is ten point five. For three times X minus two, multiply and shift: three times three point five is ten point five, minus two is eight point five.


## C7_Code

Let's compute it all in NumPy. We multiply each face by its probability and sum. We do the same for the lottery. For two dice, we average all thirty six sums. Then we roll one die ten thousand times with a fixed seed, and compute the running average with a cumulative sum.

This output is real; the script ran while the video was made. The die gives three point five, the lottery minus one, and two dice seven. The first ten rolls average two point nine. Then the averages go to three point four six, three point four five three, and three point four eight eight three. The data mean is five.


## C8_Summary

In summary. The expected value is a weighted average of the values, with the probabilities as weights: E of X equals the sum of p of x times x. A fair die expects three point five, a value you can never roll. It is linear: E of a X plus b Y equals a E of X plus b E of Y.

The average of many samples approaches the expected value: the law of large numbers. The mean of a data set is an expected value, and training minimizes an average loss. Next, in video fourteen: variance and covariance, how far values spread around this average.


## C9_Credits

This video was made by David R. You can find more lessons at David Ramirez J dot com. Thanks for watching.


## New terms in this video

- **expected value** — The probability-weighted average of the values: E[X] = Σ p(x)·x. The long-run average.
- **mean** — The average of values; for data: sum ÷ count. The expected value is the mean of a distribution.
