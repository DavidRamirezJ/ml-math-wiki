"""Narration for: Loss function — one number for how wrong you are (ML Math Atoms · Video 15 of 16).
One block <= ~75 words. Source: MML book §8.2 (Eq. 8.6, Examples 8.1–8.2); wiki loss-function,
empirical-risk-minimization."""

N = {
"intro1": """How does a machine learning model know that it is getting better? It needs a score.
Not a feeling, not a picture, but one single number that says how wrong it is right now. That
number is called the loss. Training a model means one thing: change the model so that this
number gets smaller.""",

"intro2": """Even a large language model works this way. In the book Build a Large Language Model,
an untrained GPT model starts with a loss near eleven on its training text. After ten passes
over the text, the training loss is below zero point four. In this video you will build a loss
by hand. You need the mean from video thirteen.""",

# ── C1: one prediction ──
"one1": """Start with one guess. You guess that your friend is one hundred seventy centimeters
tall. In fact, your friend is one hundred eighty centimeters tall. The true value is called the
label, y. The guess is called the prediction, y hat. How bad was the guess? We want a rule that
answers with one number.""",

"one2": """Such a rule is called a loss function. It takes the label and the prediction, and it
returns a number that is never negative. Zero means a perfect guess. Bigger means worse. We write
it as l of y and y hat. Different rules give different loss functions.""",

"one3": """The most common rule is the squared error: y minus y hat, all squared. Here, one
hundred eighty minus one hundred seventy is ten. Ten squared is one hundred. So the squared error
is one hundred. A second rule is the absolute error: the size of the gap, without its sign. Here
it is just ten.""",

"one4": """The squared error grows fast. A gap of one costs one. A gap of two costs four. A gap of
ten costs one hundred. The absolute error grows evenly: one, two, ten. So the squared error
punishes big mistakes much more. We will see why that matters soon.""",

"one5": """Some tasks have no distance at all. Is this photo a cat or a dog? Here we can use the
zero one loss. If the prediction is right, the loss is zero. If it is wrong, the loss is one. It
simply counts mistakes. It is honest, but we will see that it is hard to learn from.""",

# ── C2: average over data ──
"risk1": """One guess is not enough. A model makes many predictions, one for each example in a
data set. Here is a toy data set with three examples. The labels are two, four and six. The model
predicts one, four and eight. Let's compute the loss of each prediction.""",

"risk2": """With squared error: two minus one is one, and one squared is one. Four minus four is
zero. Six minus eight is minus two, and minus two squared is four. So the three losses are one,
zero and four. Notice that the minus sign disappears when we square.""",

"risk3": """Now take the mean of the losses: one plus zero plus four is five, divided by three.
That is about one point six seven. This average loss over the data is called the empirical risk.
Empirical means measured on the data we actually have.""",

"risk4": """Here is the formula. R equals one over N, times the sum over n from one to N, of the
loss of example n. N is the number of examples. The sum adds up all the losses. One over N turns
the sum into a mean. With absolute error, the losses are one, zero and two, and the mean is
one.""",

# ── C3: fitting a line ──
"fit1": """Now we can use the loss to choose a model. Here are four toy points. We want a straight
line through the origin that fits them. The line is y hat equals w times x. The number w is the
slope. Different values of w give different lines.""",

"fit2": """The number w is a parameter of the model. A parameter is a knob that we can turn to
change what the model predicts. A straight line has one knob. A large language model has
millions or billions of knobs. Learning means choosing the knob settings.""",

"fit3": """Let's try some knob settings. With w equal to one, the line is too flat. The mean
squared error is eight point seven five. With w equal to three, the line is too steep, and the
loss is six point seven five. With w equal to two, the line is close to all points, and the loss
is only zero point two five.""",

"fit4": """Now let the computer try many values. On the right we draw the loss for every slope,
from zero to four in small steps. This curve is L of w. Watch the line turn on the left, while a
dot moves along the curve on the right. Each slope gives one point on the loss curve.""",

"fit5": """The curve has the shape of a bowl, and the bottom is the best slope. Our demo tried four
hundred and one slopes. The lowest loss is at w equal to two point zero seven, with a loss of
about zero point two two. That slope is the fitted model.""",

# ── C4: smooth vs 0-1 ──
"smooth1": """Why is the squared error so popular? Because its loss curve is smooth. At every point,
the curve has a slope, a derivative. The slope tells us which way is downhill. If the slope is
positive, make w smaller. If it is negative, make w bigger.""",

"smooth2": """Now look at the zero one loss. Here is a toy task with six points on a line. Each point
has a class, zero or one. The rule is: predict one when x is bigger than a threshold b. For each
b, we count the mistakes.""",

"smooth3": """The loss curve is a staircase. Our demo tried seven hundred and one values of b. It
found only three different loss values. Between the steps the curve is completely flat. A flat
curve has slope zero, so the slope gives no direction. And at the steps there is no slope at all.""",

"smooth4": """That is why we usually train with a smooth loss, and we only report the count of
mistakes at the end. The smooth loss is a stand-in that we can actually minimize. For
classification, this stand-in is often the cross-entropy loss, a molecule we will meet later.""",

# ── C5: outliers ──
"out1": """The choice of loss also changes the answer. Here are five toy points. The first four
lie exactly on the line y equals two x. The fifth point should be ten, but it was recorded as
thirty. A point far away from the pattern like this is called an outlier.""",

"out2": """At w equal to two, the outlier has an error of twenty. The absolute error charges
twenty. The squared error charges four hundred. So with squared error, this one point is louder
than all the others together. The fit will bend toward it.""",

"out3": """Our demo found the best slope for each loss. With squared error, the best slope is three
point eight two. The line is pulled far up by one bad point. With absolute error, the best slope
is exactly two. The absolute error ignores the outlier much better.""",

"out4": """So neither loss is always better. Squared error is smooth and easy to minimize, but
sensitive to outliers. Absolute error is robust to outliers, but it has a sharp corner at zero.
Choosing the loss is choosing what kind of mistakes you care about.""",

# ── C6: optimization ──
"opt1": """Now we can say what learning is. Learning means choosing the parameters that make the
loss small on the data. Finding the smallest value of a function is called optimization. In this
series, this is our last axiom, axiom E: learning equals optimization.""",

"opt2": """Axiom E is a choice, not a theorem. No law of mathematics forces it. Machine learning
simply adopts it, and it works very well. Every model you will meet, from a straight line to GPT,
is trained by minimizing some loss.""",

"opt3": """But how do we find the minimum? Trying four hundred slopes worked for one knob. With a
billion knobs, a grid would need far too many tries. Instead, we follow the slope downhill, step
by step. That method is called gradient descent. It is a molecule: it combines the gradient with
the loss.""",

"opt4": """And for language models, the loss is the cross-entropy. It uses the logarithm of the
probability of the correct next token. Both gradient descent and cross-entropy are molecules
built from this atom. You will find them in the molecules map.""",

# ── practice ──
"prac1": """Now it is your turn. Pause the video and try this. The labels are three and five. The
predictions are one and six. What is the mean squared error? And what is the mean absolute
error? Take a moment.""",

"prac2": """Here is the answer. Three minus one is two, and two squared is four. Five minus six is
minus one, and minus one squared is one. The mean squared error is four plus one, divided by two,
which is two point five. The absolute errors are two and one, so the mean absolute error is one
point five.""",

"prac3": """One more question. Two points: one, three, and two, six. The model is y hat equals w
times x. Which slope is better, w equal to two or w equal to three? Use the mean squared error.
Pause and try.""",

"prac4": """With w equal to two, the predictions are two and four. The errors are one and two, so
the squared errors are one and four, and the mean is two point five. With w equal to three, the
predictions are three and six. Both are exact, so the loss is zero. The slope three wins.""",

# ── code ──
"code1": """Let's check everything in Python with NumPy. First we compute squared and absolute
errors for the three predictions. Then we try four hundred and one slopes and keep the one with
the smallest loss. Then we count mistakes with the zero one loss. Finally, we fit the data with
the outlier, using both losses.""",

"code2": """This output is real; the script ran while the video was made. The mean squared error
is one point six seven, and the mean absolute error is one. The best slope is two point zero
seven. The zero one loss takes only three different values. And with the outlier, squared error
gives three point eight two, while absolute error gives two.""",

# ── summary ──
"sum1": """In summary. A loss function turns how wrong one prediction is into one number that is
never negative. Squared error, absolute error and the zero one loss are three common choices. The
mean loss over a data set is the empirical risk. Parameters are the knobs of the model.""",

"sum2": """Learning is optimization: choose the parameters that make the loss small. Smooth losses
are easier to minimize, and squared error punishes outliers hard. Next, in video sixteen, the last
atom: the activation function, the bend that makes networks deep. See you there.""",
}
