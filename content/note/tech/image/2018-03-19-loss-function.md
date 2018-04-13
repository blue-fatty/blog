---
title: "Loss Function"
date: 2018-03-19T19:44:24+08:00
slug: loss
group: deep
weight: 4
---

This article will introdcue:

- expected risk
- some common loss function

## Loss function in classification

The goal of classification problem, or many machine learning porblem, is given training sets \\(\\{(x^{(i)}, y^{(i)}); i=1,\cdots,m\\}\\),
to find a good predictor \\(f\\) so that
\\(f(x^{(i)})\\) is a good estimate of \\(y^{(i)}\\).

**Why we need a loss function?**

We need a loss function to measure how "close" of estimate value \\(\hat y^{(i)}\\) and the target value \\(y^{(i)}\\)
and we usually optimize our model by minimizing the loss.

<!--more-->

### Notations

The notations will be used in this article:

- \\(\mathbf{X}\\): The space of input values, \\(\mathbf{X} = \\mathbb{R}^n, n \\) is the number of input features.
- \\(\mathbf{Y}\\): The space of output values or **target** values, \\(\mathbf{Y} = \\mathbb{R} \\). Currently, target value must be a scalar.
- \\(x^{(i)}\\): \\(i_{th}\\) "input" variable or features.
- \\(y^{(i)}\\): \\(i_{th}\\) target variable that we are trying to predict.
In this article, \\(y^{(i)} \in \\{1, -1\\}\\) unless otherwise indicated.
- \\(f\\): Predictor. Our goal is to learn a \\(f: \mathbf{X} \mapsto \mathbf{Y}\\), so that \\(f(x^{(i)})\\) is a good estimate of \\(y^{(i)}\\).
- \\(\hat y^{(i)}\\): Estimate of \\(y^{(i)}\\), sometimes we use this to represent \\(f(x^{(i)})\\)
- \\(L\\): Lost function or cost function or error function.
We use \\(L(\hat y^{(i)}, y^{(i)})\\) to measure how close are \\(\hat y^{(i)}\\) and \\(y^{(i)}\\).

### Expected risk

Before we talk about various of loss functions, we need explain what is expected risk.

The goal of the learning problem is to minimize **expected risk** \\(J\\),
defined as[^wikipedia]:

$$
{\displaystyle J_{real}=\displaystyle \int _{\mathbf X\times \mathbf Y}L(f({{x}}),y)p({{x}},y)\,d{{x}}\,dy}
$$

\\(p(x,y)\\) is the probability density function of the process that generated the data.

$$
(p(x, y) = p(y|x)p(x)
$$

But, in practice, we couldn't know the probability distribution \\(p(x, y)\\).

With the assumption that all samples are **independently and identically distributed**,
we can get the estimation of expected risk:

$$
\displaystyle J_{approx}=\displaystyle \frac{1}n \sum\_{i=1}^n L(f(x^{(i)}), y^{(i)})
$$

### Common loss functions

Now we will introduce various of Loss function \\(L\\), including:

1. Logistic loss
1. Mean squared error (MSE)
1. Root mean squared error (RMSE)
1. Mean absolute error (MAE)
1. Mean absolute percentage error (MAPE)
1. Mean squared logarithmic error (MSLE)
1. Hinge loss
1. Squared hinge loss
1. Cross entropy
1. Kullback leibler divergence
1. logcosh

#### Logistic loss

$$
J = \frac{1}n \sum\_{i=1}^n \log(1+ e^{-y^{(i)}\hat y^{(i)}})
$$

#### Mean squared error (MSE)

$$
J = \frac{1}n \sum\_{i=1}^n(y^{(i)}-\hat y^{(i)})^2
$$

#### Root mean squared error (RMSE)

$$
J = \sqrt{\frac{\displaystyle\sum\_{i=1}^n(y^{(i)}-\hat y^{(i)})^2}n }
$$

#### Mean absolute error (MAE)

$$
J = \frac{1}n \sum\_{i=1}^n |y^{(i)}-\hat y^{(i)}|
$$

#### Mean absolute percentage error (MAPE)

$$
J = \frac{100\%}n \sum\_{i=1}^n |\frac{y^{(i)} - \hat y^{(i)}}{y^{(i)}}|
$$

#### Mean squared logarithmic error (MSLE)

$$
\begin{align}
& J = \frac{1}n \sum\_{i=1}^n(\log(y^{(i)}+1)-\log(\hat y^{(i)}+1))^2
& (y^{(i)} \in \\{0, 1\\})
\end{align}
$$

This will penalize under estimates more than over estimates.

#### Hinge loss

$$
J = \frac{1}n \sum\_{i=1}^n \max(0, 1-y^{(i)}\hat y^{(i)})
$$

#### Squared hinge loss

$$
J = \frac{1}n \sum\_{i=1}^n ({\max(0, 1-y^{(i)}\hat y^{(i)})})^2
$$

#### Cross entropy

Also called as Logarithmic loss (Log loss).

$$
\begin{align}
& J = -\frac{1}n \sum\_{i=1}^n [y^{(i)}\log \hat y^{(i)} + (1-y^{(i)})\log(1- \hat y^{(i)})]
& (y^{(i)} \in \\{0, 1\\})
\end{align}
$$

#### Kullback leibler divergence

Also called as Relative entropy.

$$
J = \sum\_{i=1}^n y^{(i)} \log \frac{y^{(i)}}{\hat y^{i}}
$$

#### logcosh

Logarithm of the hyperbolic cosine of the prediction error.

According to the code in Keras[^keras], there is a `logcosh` function:

$$
f(x) = \ln \frac{e^{x}+e^{-x}}2
$$

(It seems writing \\(\ln\\) here is more comfortable.)

And its graph looks like this:

{{< img "https://i.imgur.com/TzQkQYF.png" "log(cosh(x))" >}}

>`log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
to `abs(x) - log(2)` for large `x`. This means that `logcosh` works mostly
like the mean squared error, but will not be so strongly affected by the
occasional wildly incorrect prediction.[^keras]

So the formula of this error is:

$$
J = \frac{1}n \sum\_{i=1}^n \log(\cosh(y^{(i)}-\hat y^{(i)}))
$$

#### Cosine proximity

Also called as cosine similarity.

$$
J = \cos \theta = \frac {\displaystyle\sum\_{i=1}^n y^{(i)}\hat y^{(i)}} {\sqrt{\displaystyle\sum\_{i=1}^n {y^{(i)}}^2}\sqrt{\displaystyle\sum\_{i=1}^n {\hat y^{(i)}}^2}}
$$

[^wikipedia]: [Loss functions for classification](https://en.wikipedia.org/wiki/Loss_functions_for_classification), Wikipedia.
[^cs229]: Andrew Ng, CS229 Lecture Notes 1
[^list]: [Metrics list](https://github.com/fxia22/ebola-1/blob/master/SharedData/Kaggle-Setup.md)
[^keras]: [losses.py](https://github.com/keras-team/keras/blob/master/keras/losses.py), Keras source code.
