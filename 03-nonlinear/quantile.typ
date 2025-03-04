#import "../template.typ": *
#show: template

= Quantile $QQ_q$ of a random variable

== Quantile of a random variable
By definition, the quantile $QQ_q$ of a random variable $Y$ is the inverse of its CDF:

$ QQ_q [Y] := cdf_Y^(-1)(q) = inf { y | cdf_Y (y) >= q}, $ <eq-quantile-def>

where $inf$ denotes the infimum, meaning that among all $y$ such that $cdf_Y (y) >= q$, we
select the smallest $y$ if it exists in the set; otherwise, we take the limit of $y$ as it
approaches the set.

#let example(title: [], content) = note(content, title: "Ex")

#compare(
  example[
    For a uniform distribution on interval $[a,b]$, the CDF is:

    $ cdf_Y (y^*) = cases((y^* - a)/(b - a)\, & y^* in [a..b], 0\, & y^* < a, 1\, & y^* > b
    ) $

    The corresponding quantile function is:

    $ QQ_q [Y] = cdf_Y^(-1)(q) = a + q dot (b - a). $
  ],
  example[
    For a normal distribution, the CDF is derived from its PDF:

    $
      cdf(y^*) = integral_(-oo)^(y^*) 1/sqrt(2 pi) e^(-y^2\/2) dd(y) = 1/2 (1 + erf(y^* / sqrt(2))),
    $

    where $erf$ is the error function. The quantile function is therefore:

    $ QQ_q [Y] = cdf_Y^(-1)(q) = sqrt(2) dot erf^(-1)(2q - 1). $

  ],
)

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

#margin[
  Some important quantiles:
  - $QQ_0 [Y] = min Y$ is the minimum value
  - $QQ_(1\/4) [Y]$ is the 1st quartile ($Q_1$)
  - $QQ_(1\/2) [Y]$ is the median or 2nd quartile ($Q_2$)
  - $QQ_(3\/4) [Y]$ is the 3rd quartile ($Q_3$)
  - $QQ_1 [Y] = max Y$ is the maximum value

  Percentiles are also quantiles, e.g. $QQ_(0.95) [Y]$ is the 95th percentile.
]

// TODO: add example of quantile to illustrate infimum

== Quantile $QQ_q$ and probability $Pr$
CDF maps real numbers $y in RR$ to probabilities $p in [0..1]$:

$
  cdf_Y (y) := Pr[Y <= y] colon y -> p.
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

  However, the notation $QQ_q [Y]$ is used here to emphasize the analogy between quantiles $QQ_q [Y]$ and
  expectation $Ex[Y]$.
]

The meaning of $QQ_q [Y]$ is that it is the value of $Y$ such that the probability of $Y$ being
less than or equal to $QQ_q [Y]$ is $q$:

$ Pr[Y <= QQ_q [Y]] = q. $ <eq-quantile-probability>

== Conditional quantile $QQ_q [Y|X]$
The generalization of the quantile $QQ_q [Y]$ to the conditional case is straightforward;
it's defined as the inverse of the conditional CDF:

$ QQ_q [Y|X] := cdf_(Y|X)^(-1)(q) = inf { y | cdf_(Y|X)(y) >= q }, $

where $cdf_(Y|X)(y) := Pr[Y <= y|X]$ is the conditional CDF defined via the conditional
PDF $pdf_(Y|X)(y) equiv pdf_Y (y|X)$ (continuous case) or PMF $pmf_(Y|X)(y) equiv pmf_Y(y|X)$ (discrete
case).

The meaning of the conditional quantile $QQ_q [Y|X]$ is that it is the value of $Y$ such
that the probability of $Y$ being less than or equal to $QQ_q [Y|X]$ given $X$ is $q$:

$ Pr[ thick Y <= QQ_q [Y|X] thick | thick X thick] = q. $ <eq-quantile-probability-conditional>

= Quantile $QQ_q$ vs. expectation $Ex$

== Robustness
Due to the linearity of expectation, it is sensitive to outliers. The quantile, on the
other hand, is robust to outliers.

#example[
  #compare[
    Consider a sample $Y = {1, 2, 3, 4, 5}$ with an outlier introduced by a typo: $Y' = {1, 2, 3, 4, 50}$.

    The sample mean is significantly affected:
    $ Ex[Y] = 1/5 dot (1 + 2 + 3 + 4 + 50) = 12. $

    While the median remains completely unaffected by the outlier:
    $ QQ_(1\/2) [Y] = 3. $

    Most of other quantiles are also unaffected.
  ][

    #table(
      columns: 4,
      [],
      [$Y$],
      [$Y'$],
      [comment],
      [$Q_0 equiv min$],
      [1],
      [1],
      [not affected],
      [$Q_(1\/4) equiv$ 1st quantile],
      [2],
      [2],
      [not affected],
      [$Q_(1\/2) equiv "med"$ (2nd quantile)],
      [3],
      [3],
      [not affected],
      [$Q_(3\/4) equiv$ 3rd quantile],
      [4],
      [4],
      [not affected],
      [$Q_0.99 equiv$ 99th percentile],
      [4],
      [4],
      [not affected],
      [$Q_1 equiv max$],
      [5],
      [50],
      [affected],
    )]
]

== Median $QQ_(1\/2) [Y]$
For $q = 1\/2$, the quantile $QQ_(1\/2) [Y]$ corresponds to the value $y^*$ such that
$Pr[Y <= y^*] = 1\/2$; i.e., the value $y^*$ cuts the distribution of $Y$ in half. This is
what the median ($y^* = "med" Y$) of a random variable $Y$ is.

Expectation $Ex[Y]$ and median $QQ_(1\/2) [Y]$ are two different measures of central
tendency of a random variable $Y$:

#compare(
  [
    - In ordinary least squares (LS), we predict the expected value of a random variable:
    $
      hat(y)(bold(x)) = Ex[Y|X = bold(x)].
    $
  ],
  [
    - We can also build a regression model that predicts the conditional median:
    $ hat(y)_"med" (bold(x)) = QQ_(1\/2) [Y|X=bold(x)]. $
  ],
)

==

However, median regression is not limited to $q = 1\/2$; we can construct a regression
model for any conditional quantile $QQ_q [Y|X]$ where
$q$ is a hyperparameter.

== Prediction $hat(y)_q$ and error term $hat(epsilon)_q$
In LS regression, the prediction
$hat(y)(bold(x))$ is singular; there are no two expectations $Ex[Y|X=bold(x)]$ for the
same random variable and parameter. The corresponding error term (residual) $epsilon(bold(x)) = y(bold(x)) - hat(y)(bold(x))$ is
also singular.

Alternatively, in quantile regression, the prediction $hat(y)_q (bold(x))$ is
parameterized by $q$; there are many (potentially infinite) predictions $QQ_q
[Y|X=bold(x)]$ for the same random variable and parameter.

// TODO: add plot of quantiles of a random variable

The corresponding error term (residual) is:

$
  hat(epsilon)_q (bold(x)^*) := QQ_q [Y|X=bold(x)^*] - y(bold(x)^*) = hat(y)_q (bold(x)^*) - y(bold(x)^*).
$

= Quantile loss $cal(L)_q$

== Check-loss
Consider an asymmetric loss function parameterized by $q in (0, 1)$:
#margin[
  This loss function is also called the _pinball loss_ and _quantile loss_ (more on this
  below)
]

$
  cal(L)_q (epsilon)
  :&= cases(q dot epsilon & quad epsilon >= 0, -(1-q) dot epsilon & quad epsilon < 0
  ) \
   &= epsilon dot q dot Ind(epsilon >=0) - epsilon dot (1-q) dot Ind(epsilon<=0)
  ,
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
#margin[
  Some implications of minimizing @eq-check-loss:

  #show math.equation: math.display

  - $Ex[Y] = arg min_bold(theta) sum_(bold(x) in X^ell) {y(bold(x)) - hat(y)(bold(x)|bold(theta))}^2$

  - $"med" Y = arg min_bold(theta) sum_(bold(x) in X^ell) abs(y(bold(x)) - hat(y)(bold(x)|bold(theta)))$

  - $QQ_q [Y] = arg min_bold(theta) sum_(bold(x) in X^ell) cal(L)_q (y(bold(x)) - hat(y)(bold(x)|bold(theta)))$
]

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

== Probabilistic model
Suppose the distribution of the data $(bold(x), y)$ is modeled as a joint distribution $pdf(bold(x), y)$.
Our goal is to predict the quantile $QQ_q [Y] = fun(bold(x))$ for a given $bold(x)$, i.e.,
to predict the conditional quantile $QQ_q [Y|X=bold(x)]$.

== Optimization problem
The empirical risk is defined as the average quantile loss @eq-check-loss over the
distribution $pdf(bold(x), y)$. By minimizing the empirical risk, we can find the optimal
model $a^*(bold(x))$ that predicts the quantile $QQ_q [Y|X=bold(x)]$:

$
  a^*(bold(x)) = arg min_a { ub(Ex_((bold(x), y) ~ pdf(bold(x), y)) [ cal(L)_q (y - a(bold(x))) ], cal(R)(a)) }.
$

== Practical reformulation
From the theoretical expression of the empirical risk, we can derive a practical
reformulation of the quantile regression problem.

For a specific pair $(bold(x)^*, y^*)$ drawn from the joint distribution $pdf(bold(x), y)$ represented
by a training set $(X, Y)^ell$, the empirical risk can be expressed via the check loss
@eq-check-loss:

$ cal(R)(a) = 1 / ell dot sum_((bold(x)^*, y^*) in (X, Y)^ell) cal(L)_q (y^* - a(bold(x)^*)) -> min_a. $

The model $a(bold(x)) equiv a(bold(x)|bold(theta); q)$ can be any regression model.

== Linear quantile regression
The conditional quantile $QQ_q [Y|X]$ can be modeled as a linear function of predictors $bold(x)$:

$ hat(y)_q (bold(x)) = QQ_q [Y|X = bold(x)] = bra bold(x), bold(beta)(q) ket, quad beta_j (q) = beta_(j|q), $

where $bold(beta)(q)$ is a vector of regression coefficients, and $beta_j (q) = beta_(j|q) in RR$ are
regression coefficients for the feature $bold(x)^j$ and a _predefined hyperparameter_ $q$.

Coefficients $beta_j (q)$ are estimated by minimizing the empirical risk:

$ cal(R)(bold(beta)) = 1 / ell dot sum_(bold(x) in X^ell) cal(L)_q (y(bold(x)) - hat(y)_q (bold(x)|bold(beta))) -> min_(bold(beta)). $

== Gradient boosting quantile regression
Gradient boosting can also be used for quantile regression:

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(loss="quantile", alpha=0.5)
```

== Neural quantile regression
An ordinary neural network can model conditional quantiles $QQ_q [Y|X]$ as well. The
quantile loss @eq-check-loss determines that the model $a(bold(x))$ will predict
conditional quantiles $QQ_q [Y|X]$; it can be implemented as follows:

```python
import lightning as L

class QuantileLoss(L.LightningModule):
  def __init__(self, q):
    super().__init__()
    self.q = q

  def forward(self, y_pred, y_true):
    errors = y_true - y_pred
    return torch.max(self.q * errors, (self.q - 1) * errors).mean()
```

This loss function can be used to train a general regression model.

= Interpretation

== Targets

- Median (quantiles) of $y$ is sometimes more interpretable and a better measure of
  centrality than the mean, particularly in right-skewed or bimodal data. For example,
  median salary or house price is more interpretable than mean values. In some cases, a "mean"
  prediction does not exist, such as with binary groups (deceased = 0 and deceased = 1),
  where the prediction for an "average" $bold(x)$ is meaningless.

- Quantile regression predictions represent the *conditional quantiles of the response
  variable* $y$ given the predictors $bold(x)$. These predictions act *as boundaries*,
  splitting the distribution into $q$-lower and $(1-q)$-upper portions by value of $y$, much
  like the median divides the data into two equal groups. These predictions do not
  correspond to any specific object or individual in the training data but rather model the
  distribution of $y$ itself.

== Parameters

- Coefficients $beta$ in linear quantile regression are noisier than in OLS and depend on
  quantile $q$, making them harder to interpret. The Gauss-Markov theorem ensuring
  convergence and variance in OLS does not apply to quantile regression.

- Exact values of $beta$ in OLS are interpretable, but in quantile regression, they are
  generally not. In simple cases, they can be close to OLS coefficients and interpretable.
  However, when quantile regression is applied to transformed data (e.g., $log(y)$),
  coefficients remain invariant, but their contribution to $y' = log(y)$ becomes less
  obvious. For skewed data where OLS fails, quantile regression coefficients differ
  significantly from OLS but may still be interpretable.
