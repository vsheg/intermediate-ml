#import "../template.typ": *
#show: template

= Quantile $QQ_q$

== Quantile of a random variable

#margin[
  For a continuous random variable $Y$, the probability of $Y$ being less than or equal to $y^*$
  is given by the cumulative distribution function (CDF):

  $ cdf_Y (y^*) := Pr[Y <= y^*] = integral_(y=-oo)^(y^*) pdf_Y (y) dd(y), $

  where $pdf_Y (y)$ is the probability density function (PDF).
]

#margin[
  For a discrete random variable $Y$, the CDF is defined as:

  $ cdf_Y (y^*) := Pr[Y <= y^*] = sum_(y <= y^*) pmf_Y (y) $

  where $pmf_Y (y)$ is the probability mass function (PMF).
]

By definition, the quantile $QQ_q$ of a random variable $Y$ is the inverse of its CDF:

$ QQ_q [Y] := cdf_Y^(-1)(q) = inf { y | cdf_Y (y) >= q}, $ <eq-quantile-def>

where $inf$ denotes the infimum, meaning that among all $y$ such that $cdf_Y (y) >= q$, we
select the smallest $y$ if it exists in the set; otherwise, we take the limit of $y$ as it
approaches the set.

#margin[
  $QQ_(1\/4) [Y], QQ_(1\/2) [Y], QQ_(3\/4) [Y]$ are 1st, 2nd (median), and 3rd quantiles

  The expected value and $1\/2$-quantile (median) provide different measures of central
  tendency:
  $Ex[Y] = macron(y)$ vs. $QQ_(1\/2)[Y] = "med" Y$
]

// TODO: add example of quantile to illustrate infimum

== Quantile $QQ_q$ and probability $Pr$
CDF maps real numbers $y in RR$ to probabilities $p in [0..1]$:

$
  cdf_Y (y) := Pr[Y <= y] colon y -> p
$

Quantile $QQ_q$, being the inverse of CDF, maps probabilities $p in [0..1]$ to real
numbers $y in RR$:

$
  QQ_p [Y] = cdf_Y^(-1) (p) colon p -> y,
$

We specifically denote probability $p$ as $q$ to emphasize its connection to quantiles.

#margin[
  Technically, $QQ_q [Y]$ is a function of $q$, and it is usually denoted as $Q_Y (p)$,
  similar to PDF $pdf_Y (y)$ and CDF $cdf_Y (y)$.

  However, the notation $QQ_q [Y]$ is used to emphasize the connection between quantiles $QQ_q [Y]$ and
  expectation $Ex[Y]$.
]

The meaning of $QQ_q [Y]$ is that it is the value of $Y$ such that the probability of $Y$ being
less than or equal to $QQ_q [Y]$ is $q$:

$ Pr[Y <= QQ_q [Y]] = q. $ <eq-quantile-probability>

== Conditional quantile of a random variable
The generalization of the quantile $QQ_q [Y]$ to the conditional case is straightforward;
it's defined as the inverse of the conditional CDF:

$ QQ_q [Y|X] := cdf_(Y|X)^(-1)(q) = inf { y | cdf_(Y|X)(y) >= q }, $

where $cdf_(Y|X)(y) := Pr[Y <= y|X]$ is the conditional CDF defined via the conditional
PDF $pdf_(Y|X)(y) equiv pdf_Y (y|X)$ (continuous case) or PMF $pmf_(Y|X)(y) equiv pmf_Y(y|X)$ (discrete
case).

The meaning of the conditional quantile $QQ_q [Y|X]$ is that it is the value of $Y$ such
that the probability of $Y$ being less than or equal to $QQ_q [Y|X]$ given $X$ is $q$:

$ Pr[ thick Y <= QQ_q [Y|X] thick | thick X thick] = q. $ <eq-quantile-probability-conditional>

== Expectation $Ex[Y|X]$ and quantile $QQ_q [Y|X]$
In ordinary least squares (LS) regression, we predict the expected value of a random
variable:
$ hat(y)(bold(x)^*) equiv Ex[Y|X = bold(x)^*] equiv a(bold(x)^*). $

In quantile regression, we predict the $q$-quantile of a random variable:
$ hat(y)_q (bold(x)^*) equiv QQ_q [Y|X=bold(x)^*] equiv a(bold(x)^*), $

where $q$ is a hyperparameter.

== Prediction $hat(y)_q$ and error term $hat(epsilon)_q$
In LS regression, the prediction $hat(y)(bold(x)^*)$ is singular; there are no two
expectations
$Ex[Y|X=bold(x)^*]$ for the same random variable and parameter. The corresponding error
term (residual) $epsilon(bold(x)^*) = y(bold(x)^*) - hat(y)(bold(x)^*)$ is also singular.

Alternatively, in quantile regression, the prediction $hat(y)_q (bold(x)^*)$ is
parameterized by $q$; there are many (potentially infinite) predictions $QQ_q [Y|X=bold(x)^*]$ for
the same random variable and parameter.

// TODO: add plot of quantiles of a random variable

The corresponding error term (residual) is:

$ hat(epsilon)_q (bold(x)^*) := QQ_q [Y|X=bold(x)^*] - y(bold(x)^*) = hat(y)_q (bold(x)^*) - y(bold(x)^*). $

= Quantile loss $cal(L)_q$

== Check-loss
Consider an asymmetric loss function parameterized by $q in (0, 1)$:
#margin[
  This loss function is also called the _pinball loss_ and _quantile loss_ (more on this
  below)
]
$
  cal(L)_q (epsilon) := cases(q dot epsilon & quad epsilon >= 0, -(1-q) dot epsilon & quad epsilon < 0
  ) quad ,
$ <eq-check-loss>

#margin[
  Strictly speaking, this is an estimation of the error: $hat(epsilon) := y - hat(y)$; for
  different estimations of $hat(y)$, there are different $hat(epsilon)$
]

where $epsilon := y - hat(y)$ is the error term (residual) and $hat(y)$ is the prediction
of a regression model.

// TODO: add check loss plot

== Constant model
Let's first consider the simplest case, where we look for $a^*$ in the family of all
constant models $a^* in {a | a = const}$.

#margin[
  For a pair $(bold(x), y)$ taken from the joint distribution $pdf(bold(x), y)$, a function
  $hat(y)(bold(x)) = a^*(bold(x))$ that minimizes $cal(R)(a = a^*)$ can be found by
  minimizing $cal(R)(a)$:

  $ cal(R)(a) &:= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (a(bold(x)), y)] \
            &= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (y - a(bold(x)))] \
            &= integral cal(L)_q (y - a(bold(x))) dot pdf(bold(x), y) dot dd(bold(x)) dd(y) \
            &= integral cal(L)_q (y - a(bold(x))) dot dd(cdf(bold(x), y)) -> min_a $
]

The empirical risk (expected check-loss) can be expressed as:

$
  cal(R)(a) &= integral cal(L)_q (focus(epsilon = y - a)) dd(cdf(bold(x), y)) comment(a(bold(x)) -> a = const) \
            &= integral cal(L)_q (epsilon = y - a) focus(dd(cdf(y))) comment("as" a "is not a function of" bold(x)) \

            &= limits(integral)_focus(y - a >= 0) cal(L)_q (y - a) dd(cdf(y)) + limits(integral)_focus(y - a < 0) cal(L)_q (y - a) dd(cdf(y)) #comment[nonoverlapping regions: $epsilon >= 0$ and $epsilon < 0$] \

            &= limits(integral)_(y >= a) focus((y - a) dot q) dd(cdf(y)) - limits(integral)_(y < a) focus((y - a) dot (1-q)) dd(cdf(y)) -> min_a #comment[expand $cal(L)_q (epsilon)$ according to @eq-check-loss] \
$

== Risk minimization
The integral is split at $a = a^*$ into two independent regions: $(-infinity..a^*)$ and $[a^*..+infinity)$.
By differentiating both integrals with respect to $a$, we can find $a^*$:

$
  pdv(, a) cal(R)(a) &= focus(q) dot integral_(y>=a) pdv(, a) (y - a) dd(cdf(y)) - focus((1-q)) dot integral_(y < a) pdv(, a) (y - a) dd(cdf(y)) #comment[constants] \
                     &= -q dot focus(integral_(y = a^*)^(+oo) dd(cdf(y))) + (1-q) dot focus(integral_(y=-oo)^(a^*) dd(cdf(y))) comment(dd(cdf(y)) = pdf(y) dd(y)) \
                     &= -q dot (1 - cdf_Y (a)) + (1-q) dot cdf_Y (a) = -q + cdf_Y (a) comment(cdf_Y (a) equiv cdf(Y = a)) \
$

At the extreme point $a = a^*$, the derivative of the risk is zero:

$ -q + cdf_Y (a^*) = 0. $

Thus, the optimal constant model $a^*$ is the $q$-quantile of the random variable $Y$:

$ a^* = cdf_Y^(-1) (q) = QQ_q [Y]. $

== Implications
We assumed that $a$ is a constant function of $bold(x)$ and derived the optimal constant
model $hat(y)(bold(x)) = a^*$ that minimizes the empirical risk (expected check-loss) $cal(R)(a)$.
Notably, if we differentiate $cal(R)(a)$ with respect to any general function $a(bold(x))$,
the result remains the same.

Minimizing the check loss $cal(L)_q (epsilon)$ for a regression model $hat(y)(bold(x))$ is
equivalent to finding the $q$-quantile of the random variable $Y$. Therefore, the
algorithm $a^*$ derived from solving the minimization problem $cal(R) = Ex[cal(L)_q] -> min$
effectively predicts the $q$-quantile of $Y$.

== Quantile parameter $q$
By using the check loss $cal(L)_q (epsilon)$, we can train a regression model $hat(y)_q (bold(x))$
that predicts the $q$-quantile of the random variable $Y$ given the input $bold(x)$:

$ hat(y)_q (bold(x)) = QQ_q [Y|X=bold(x)], $

where $hat(y)_q$ *depends both on hyperparameter $q$* and on the input $bold(x)$. This
means that *predictions $hat(y)_q (bold(x))$ are different for different values of $q$*.

Likewise, the error term (residual) depends on $q$:

$ epsilon_q (bold(x)) = QQ_q [Y|X=bold(x)] - y(bold(x)) = hat(y)_q - y, $

and the check loss in @eq-check-loss is actually $cal(L)_q (epsilon) equiv cal(L)_q (epsilon_q)$.

= Quantile regression