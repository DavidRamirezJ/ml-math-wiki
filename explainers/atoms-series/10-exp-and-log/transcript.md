# 10-exp-and-log — transcript


## C0_Intro

A language model reads a text with one thousand words. For each word, it gives the correct word a probability, say one in a hundred. To score the whole text, we multiply all one thousand probabilities. Ask a computer to do that, and it answers zero. Not a small number. Exactly zero. The model has learned something, but the computer cannot see it.

Two functions fix this problem: the exponential and the logarithm. They are a pair. One undoes the other. In this video you will learn what they mean, their key rules, and why machine learning uses logs everywhere. You need school arithmetic with powers. For the slopes, you need the derivative from video seven.


## C1_Exp

Start with repeated growth. A bacterium splits in two every hour. After one hour you have two. After two hours, four. After three hours, eight. That is two times two times two, written two to the power three. The power counts how many times we multiply.

Now a special base. Suppose money grows by one hundred percent per year. Paid once, one euro becomes two. Paid twelve times, in small monthly pieces, it becomes about two point six one. Paid every day, about two point seven one. Paid all the time, it approaches a fixed number: e, about two point seven one eight.

The function e to the power x is called the exponential. Look at its graph. At x equals zero, it is one. At x equals one, it is e. At x equals two, about seven point three nine. To the left, it gets smaller and smaller, but it never reaches zero. The exponential is always positive.

The exponential has one key rule. e to the a plus b equals e to the a times e to the b. Why? e squared is two copies of e multiplied. e cubed is three copies. Together, that is five copies. Check it: seven point three eight nine times twenty point zero eight six is one hundred forty eight point four one, which is e to the five.


## C2_Log

Now turn the question around. e to what power gives seven point three eight nine? The answer is two. This question, what power gives this number, is called the logarithm. We write the natural logarithm as l n. So l n of seven point three eight nine is two.

The logarithm undoes the exponential. Put x into the exponential machine, then put the result into the logarithm machine. You get x back. So l n of e to the x equals x. On the graph, the log curve is the mirror image of the exponential curve, reflected across the line y equals x.

Read some values. l n of one is zero, because e to the zero is one. l n of e is one. For numbers between zero and one, the logarithm is negative. And the logarithm of zero or of a negative number does not exist. No power of e can give zero, because e to the x is always positive.


## C3_Products

Here is the most useful rule of logs. The log of a times b equals the log of a plus the log of b. Products become sums. Why is it true? Write a as e to the p, and b as e to the q. Then a times b is e to the p plus q. Its logarithm is p plus q.

Let's check it with numbers. l n of two is about zero point six nine three. l n of three is about one point zero nine nine. Add them: one point seven nine two. And l n of six, computed directly, is also one point seven nine two. The rule works.

Two more rules follow from it. The log of a power: the log of a to the n equals n times the log of a, because a power is repeated multiplication. And the log of a division is the log of the top minus the log of the bottom. Multiply becomes add, divide becomes subtract.

Before computers, people used printed tables of logarithms for exactly this reason. To multiply two long numbers, they looked up the two logs, added them, and looked up the answer. Adding is much easier than multiplying. Machine learning uses the same trick today, for a different reason.


## C4_Tiny

A probability is a number between zero and one. Its logarithm is zero or negative. Probability one gives zero. One half gives about minus zero point six nine. One in a hundred gives minus four point six one. And zero point zero zero zero one gives minus nine point two one. Tiny numbers become friendly negative numbers.

Notice the pattern. Every time the probability gets ten times smaller, the log goes down by the same step, about two point three. The log measures how many factors of ten we lost. It is like a ruler that counts zeros, so very small numbers stay easy to read and compare.

Now the computer problem. A standard computer number, called float sixty four, cannot store positive numbers smaller than about ten to the minus three hundred twenty four. Anything smaller is rounded to zero. This is called underflow. Multiply probabilities of one in a hundred, and the product hits zero after only one hundred sixty two factors.

With logs, there is no problem. The log of the product is the sum of the logs. One thousand times minus four point six one is about minus four thousand six hundred five. That is a normal number. Also, the log keeps the order: a bigger probability always has a bigger log. So the best answer stays the best answer.


## C5_Slopes

Now the slopes. Remember the derivative from video seven: the slope of a curve at a point. The exponential has a beautiful property. Its derivative is itself. The slope of e to the x equals e to the x. Watch the tangent line. At zero, height one and slope one. At one, height and slope are both e.

The derivative of the logarithm is one over x. At x equals one, the slope is one. At x equals four, the slope is one quarter. Near zero, the slope is huge. At zero point zero one, it is one hundred. In learning, this means a tiny probability on the right answer gives a big slope, so the model is pushed hard.

Where do exp and log come from in our series? They are not one of the five axioms. They are school math, the floor that the axioms stand on. But together with the chain rule from video nine, these two slopes are enough to differentiate most loss functions in machine learning.


## C6_ML

First use: making numbers positive. A classifier produces scores, and scores can be negative. But probabilities must be positive. So we apply the exponential to every score. Scores two, one, and minus one become seven point three nine, two point seven two, and zero point three seven. All positive.

Then divide by the total, about ten point four eight. We get zero point seven zero five, zero point two five nine, and zero point zero three five. They add up to one. This recipe is called softmax. It is a molecule built from this atom, and we will meet it again later in the series.

Second use: the log-likelihood. To score a whole data set, we multiply the probabilities of all examples. Take three examples with probabilities zero point nine, zero point eight, and zero point five. The product is zero point three six. The sum of the logs is minus one point zero two two, which is exactly the log of zero point three six.

So training can maximize a sum of logs instead of a fragile product. Deep learning code prefers to minimize, so it flips the sign and takes the average. That is the cross-entropy loss, which we meet in video fifteen. And libraries go one step further for numerical stability: they compute the log directly from the scores, without ever forming the tiny probability.


## C7_Practice

Your turn. Pause the video and try this. You know that l n of two is about zero point six nine three and l n of five is about one point six zero nine. Without a calculator, what is l n of ten?

Ten is two times five. A log of a product is a sum of logs. So l n of ten is zero point six nine three plus one point six zero nine, which is about two point three zero. That is also the step we saw for every factor of ten.

Second question. A model gives four examples the probability one half each. What is the product of the four probabilities? And what is the sum of their logs? Use l n of one half, about minus zero point six nine three. Pause and try.

The product is one half to the power four, which is one sixteenth, or zero point zero six two five. The sum of logs is four times minus zero point six nine three, about minus two point seven seven. And l n of zero point zero six two five is indeed about minus two point seven seven.


## C7_Code

Let's check it in Python with NumPy. We test the exponential rule and the log rule. Then we make one thousand probabilities of zero point zero one. We multiply them with n p dot prod, and we add their logs with n p dot sum. We also find when the product first becomes zero.

This output is real; the script ran while the video was made. e squared times e cubed equals e to the five. l n two plus l n three equals l n six. The product of one thousand probabilities is zero point zero, exactly zero. It became zero after one hundred sixty two factors. But the sum of the logs is minus four thousand six hundred five point one seven. The information is safe.


## C8_Summary

In summary. The exponential, e to the x, is repeated growth with base e, about two point seven one eight. It is always positive, and it turns sums into products. The logarithm answers the question: what power gives this number? It undoes the exponential, and it turns products into sums.

Logs turn tiny probabilities into friendly negative numbers and prevent underflow. The slope of e to the x is e to the x, and the slope of l n x is one over x. Next, in video eleven, we start probability: a random variable, a way to turn chance into numbers.


## C9_Credits

This video was made by David R. You can find more lessons at David Ramirez J dot com. Thanks for watching.


## New terms in this video

- **exponential** — The function e to the power x, with e ≈ 2.718: repeated growth that is always positive.
- **logarithm** — The power you must put on e to get a number: ln y = x means e^x = y.
