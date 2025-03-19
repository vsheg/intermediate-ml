#import "../template.typ": *
#show: template

= Quantile $QQ_q$ of a random variable

== Quantile of a sample
Given an unordered sample $y_1, y_2, ..., y_ell$, we can construct a sorted sample $y^((1)) <= y^((2)) <= ... <= y^((ell))$,
where $y^((i))$ is the $i$th smallest value of the sample ($i = 1..ell$), also known as the $i$th _order statistic_.

#margin[
  Some order statistics:
  - $y^((1)) = min Y$ is the 1st order statistic
  - $y^((2))$ is the 2nd order statistic (2nd smallest value)
  - $y^((ell\/2))$ is the median, which divides the sample in half
  - $y^((ell)) = max Y$ is the last ($ell$th) order statistic
]

#margin[NB][
  In $y^((i))$, values $i = 1..ell$ are integers, while in $y^((q))$, values $q in [0..1]$ are
  fractional, e.g., $y^((10))$ refers to the 10th order statistic, while $y^((0.1))$ refers to the
  0.1-quantile.
]

Informally, the $q$-quantile $y^((q))$ is the value that divides the ordered sample into two parts
with proportions $q : (1 - q)$. However, this definition is ambiguous. One practical approach is to
use different formulas for $q dot ell in.not NN$ and $q dot ell in NN$:

#margin[
  In practice, other definitions of quantiles $y^((q))$ are also used, e.g., $y^((q)) := y^((ceil(q dot ell)))$ for
  any $ell$
]

$
  y^((q)) := cases(
    y^((ceil(q dot ell))) & quad q dot ell "is not integer",
    1/2 (y^((q dot ell)) + y^((q dot ell + 1))) & quad q dot ell "is integer"
  ,

  )
$

== Quantile of a random variable
For a random variable $Y$, the quantile function, denoted either as $QQ_q [Y]$ or $y^((q))$, is
defined as the inverse of its CDF:

$ QQ_q [Y] := cdf_Y^(-1)(q) = inf { y | cdf_Y (y) >= q }, $ <eq-quantile-def>

where $inf$ denotes the infimum, which is the greatest lower bound, i.e., $QQ_q [Y]$ is the smallest
value $y$ for which the probability $Pr[Y <= y]$ is at least $q$.

#grid(
  columns: 2,
  example[
    For a uniform distribution on interval $[a,b]$, the CDF is:

    $
      cdf_Y (y^*) = cases((y^* - a)/(b - a)\, & y^* in [a..b], 0\, & y^* < a, 1\, & y^* > b
      )
    $

    The corresponding quantile function is:

    $ QQ_q [Y] = cdf_Y^(-1)(q) = a + q dot (b - a). $
  ],
  example[
    For a normal distribution, the CDF is derived from its PDF:

    $
      cdf(y^*) = integral_(-oo)^(y^*) 1 / sqrt(2 pi) e^(-y^2\/2) dd(y) = 1 / 2 (1 + erf(y^* / sqrt(2))),
    $

    where $erf$ is the error function. The quantile function is therefore:

    $ QQ_q [Y] = cdf_Y^(-1)(q) = sqrt(2) dot erf^(-1)(2q - 1). $

  ],
)

For a sample ${y_1, ..., y_ell}$, the empirical CDF

$ cdf_Y (y) := 1 / ell dot sum_(i=1)^ell Ind(y_i <= y) $

can be used in @eq-quantile-def to define quantiles $QQ_q$.

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

Quantile $QQ_q$, being the inverse of CDF, maps probabilities $p in [0..1]$ to real numbers $y in RR$:

$
  QQ_p [Y] = cdf_Y^(-1) (p) colon p -> y,
$

We specifically denote probability $p$ as $q$ to emphasize its connection to quantiles.

#margin[
  Technically, $QQ_q [Y]$ is a function of $q$, and it is usually denoted as $Q_Y (p)$, similar to PDF $pdf_Y (y)$ and
  CDF $cdf_Y (y)$.

  However, the notation $QQ_q [Y]$ is used here to emphasize the analogy between quantiles $QQ_q [Y]$ and
  expectation $Ex[Y]$.
]

The meaning of $QQ_q [Y]$ is that it is the value of $Y$ such that the probability of $Y$ being less
than or equal to $QQ_q [Y]$ is $q$:

$ Pr[Y <= QQ_q [Y]] = q. $ <eq-quantile-probability>

== Conditional quantile $QQ_q [Y|X]$
The generalization of the quantile $QQ_q [Y]$ to the conditional case is straightforward; it's
defined as the inverse of the conditional CDF:

$ QQ_q [Y|X] := cdf_(Y|X)^(-1)(q) = inf { y | cdf_(Y|X)(y) >= q }, $

where $cdf_(Y|X)(y) := Pr[Y <= y|X]$ is the conditional CDF defined via the conditional PDF $pdf_(Y|X)(y) equiv pdf_Y (y|X)$ (continuous
case) or PMF $pmf_(Y|X)(y) equiv pmf_Y(y|X)$ (discrete case).

The meaning of the conditional quantile $QQ_q [Y|X]$ is that it is the value of $Y$ such that the
probability of $Y$ being less than or equal to $QQ_q [Y|X]$ given $X$ is $q$:

$
  Pr[ thick Y <= QQ_q [Y|X] thick | thick X thick] = q.
$ <eq-quantile-probability-conditional>

= Quantile $QQ_q$ vs. expectation $Ex$

== Robustness
Due to the linearity of expectation, it is sensitive to outliers. The quantile, on the other hand,
is robust to outliers.

#example(cols: 2)[
  Consider a sample $Y = {1, 2, 3, 4, 5}$ with an outlier introduced by a typo: $Y' = {1, 2, 3, 4, 50}$.

  The sample mean is significantly affected:
  $ Ex[Y] = 1 / 5 dot (1 + 2 + 3 + 4 + 50) = 12. $

  While the median remains completely unaffected by the outlier:
  $ QQ_(1\/2) [Y] = 3. $

  Most of other quantiles are also unaffected.

  #colbreak()

  #table(
    columns: 4,
    [], [$Y$], [$Y'$], [comment],
    [$Q_0 equiv min$], [1], [1], [not affected],
    [$Q_(1\/4) equiv$ 1st quantile], [2], [2], [not affected],
    [$Q_(1\/2) equiv "med"$ (2nd quantile)], [3], [3], [not affected],
    [$Q_(3\/4) equiv$ 3rd quantile], [4], [4], [not affected],
    [$Q_0.99 equiv$ 99th percentile], [4], [4], [not affected],
    [$Q_1 equiv max$], [5], [50], [affected],
  )
]

== Median $QQ_(1\/2) [Y]$
For $q = 1\/2$, the quantile $QQ_(1\/2) [Y]$ corresponds to the value $y^*$ such that
$Pr[Y <= y^*] = 1\/2$; i.e., the value $y^*$ cuts the distribution of $Y$ in half. This is what the
median ($y^* = "med" Y$) of a random variable $Y$ is.

Expectation $Ex[Y]$ and median $QQ_(1\/2) [Y]$ are two different measures of central tendency of a
random variable $Y$:

#grid(
  columns: 2,
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

However, median regression is not limited to $q = 1\/2$; we can construct a regression model for any
conditional quantile $QQ_q [Y|X]$ where
$q$ is a hyperparameter.

== Prediction $hat(y)_q$ and error term $hat(epsilon)_q$
In LS regression, the prediction
$hat(y)(bold(x))$ is singular; there are no two expectations $Ex[Y|X=bold(x)]$ for the same random
variable and parameter. The corresponding error term (residual) $epsilon(bold(x)) = y(bold(x)) - hat(y)(bold(x))$ is
also singular.

Alternatively, in quantile regression, the prediction $hat(y)_q (bold(x))$ is parameterized by $q$;
there are many (potentially infinite) predictions $QQ_q
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
  This loss function is also called the _pinball loss_ and _quantile loss_ (more on this below)
]

$
  cal(L)_q (epsilon)
  :&= cases(q dot epsilon & quad epsilon >= 0, -(1-q) dot epsilon & quad epsilon < 0
  ) \
  &= epsilon dot q dot Ind(epsilon >=0) - epsilon dot (1-q) dot Ind(epsilon<=0)
  ,
$ <eq-check-loss>

#margin[
  Strictly speaking, this is an estimation of the error: $hat(epsilon) := y - hat(y)$; for different
  estimations of $hat(y)$, there are different $hat(epsilon)$
]

where $epsilon := y - hat(y)$ is the error term (residual) and $hat(y)$ is the prediction of a
regression model.

#margin({
  // NOTE: when plotted on the same graph, it looks like a six-legged spider: not illustrative at all
  let x = lq.linspace(-2, 2)

  let sub-figure(q) = figure(
    caption: [Check loss $cal(L)_#q (epsilon)$],
    lq.diagram(
      width: 3cm,
      height: 3cm / 2,
      xlabel: $epsilon$,
      lq.plot(
        mark: none,
        x,
        x.map(epsilon => if (epsilon >= 0) { q * epsilon } else { -(1 - q) * epsilon }),
      ),
    ),
  )

  multi-figure(sub-figure(0.25), sub-figure(0.5), sub-figure(0.75))
})

== Constant model
Let's first consider the simplest case, where we look for $a^*$ in the family of all constant models $a^* in {a | a = const}$.

#margin[
  For a pair $(bold(x), y)$ taken from the joint distribution $pdf(bold(x), y)$, a function
  $hat(y)(bold(x)) = a^*(bold(x))$ that minimizes $cal(R)(a = a^*)$ can be found by minimizing $cal(R)(a)$:

  $
    cal(R)(a) &:= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (a(bold(x)), y)] \
    &= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (y - a(bold(x)))] \
    &= integral cal(L)_q (y - a(bold(x))) dot pdf(bold(x), y) dot dd(bold(x)) dd(y) \
    &= integral cal(L)_q (y - a(bold(x))) dot dd(cdf(bold(x), y)) -> min_a
  $
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
We assumed that $a$ is a constant function of $bold(x)$ and derived the optimal constant model $hat(y)(bold(x)) = a^*$ that
minimizes the empirical risk (expected check-loss) $cal(R)(a)$. Notably, if we differentiate $cal(R)(a)$ with
respect to any general function $a(bold(x))$, the result remains the same.

Minimizing the check loss $cal(L)_q (epsilon)$ for a regression model $hat(y)(bold(x))$ is
equivalent to finding the $q$-quantile of the random variable $Y$. Therefore, the algorithm $a^*$ derived
from solving the minimization problem $cal(R) = Ex[cal(L)_q] -> min$
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

where $hat(y)_q$ *depends both on hyperparameter $q$* and on the input $bold(x)$. This means that
*predictions $hat(y)_q
(bold(x))$ are different for different values of $q$*.

Likewise, the error term (residual) depends on $q$:

$ epsilon_q (bold(x)) = QQ_q [Y|X=bold(x)] - y(bold(x)) = hat(y)_q - y, $

and the check loss in @eq-check-loss is actually $cal(L)_q (epsilon) equiv cal(L)_q (epsilon_q)$.

= Quantile regression

#margin[
  Quantile regression was introduced by Roger Koenker and Gilbert Bassett in @koenker1978regression.

  For a short overview and examples see @koenker2001quantile and @Koenker2005quantile for details.
]

== Probabilistic model
Suppose the distribution of the data $(bold(x), y)$ is modeled as a joint distribution $pdf(bold(x), y)$.
Our goal is to predict the quantile $QQ_q [Y] = fun(bold(x))$ for a given $bold(x)$, i.e., to
predict the conditional quantile $QQ_q [Y|X=bold(x)]$.

== Optimization problem
The empirical risk is defined as the average quantile loss @eq-check-loss over the distribution $pdf(bold(x), y)$.
By minimizing the empirical risk, we can find the optimal model $a^*(bold(x))$ that predicts the
quantile $QQ_q [Y|X=bold(x)]$:

$
  a^*(bold(x)) = arg min_a ub(Ex_((bold(x), y) ~ pdf(bold(x), y)) [ cal(L)_q (y - a(bold(x))) ], cal(R)(a)).
$

== Practical reformulation
From the theoretical expression of the empirical risk, we can derive a practical reformulation of
the quantile regression problem.

For a specific pair $(bold(x)^*, y^*)$ drawn from the joint distribution $pdf(bold(x), y)$ represented
by a training set $(X, Y)^ell$, the empirical risk can be expressed via the check loss
@eq-check-loss:

$
  cal(R)(a) = 1 / ell dot sum_((bold(x)^*, y^*) in (X, Y)^ell) cal(L)_q (y^* - a(bold(x)^*)) -> min_a.
$

The model $a(bold(x)) equiv a(bold(x)|bold(theta); q)$ can be any general regression model
supporting custom loss functions or the quantile loss $cal(L)_q$ specifically.

#let quantile-model-plot(file) = {
  let data = lq.load-txt(read(file), header: true)
  let x = data.remove("x")
  let y = data.remove("y")

  let label-fn(col) = if (col == "mean") { $Ex$ } else { $QQ_#col$ }

  lq.diagram(
    width: 4cm,
    height: 3cm,
    legend: (position: right + bottom),
    lq.scatter(x, y, size: 3pt, stroke: none, color: ghost-color),
    ..data.keys().map(col => lq.plot(mark: none, x, data.at(col), label: label-fn(col))),
  )
  // TODO: add link to code
  // FIX: legend overlaps with the plot
}

#grid(
  columns: (1fr, 1.3fr),
  [== Linear quantile regression
    The conditional quantile $QQ_q [Y|X]$ can be modeled as a linear function of predictors $bold(x)$:

    #margin(
      figure(
        caption: [Linear quantile regression for non-normaly distributed noise],
        quantile-model-plot("linear/out.csv"),
      ),
    )

    $
      QQ_q [Y|X = bold(x)] = bra bold(x), bold(beta) ket, quad beta_j equiv beta_j (q),
    $ <eq-linear-quantile-regression>

    where $bold(beta)(q)$ is a vector of regression coefficients, and $beta_j (q) = beta_(j|q) in RR$ are
    regression coefficients for the feature $bold(x)^j$ and a _predefined hyperparameter_ $q$.
    Coefficients $beta_j (q)$ are estimated by minimizing the empirical risk:

    $
      cal(R)(bold(beta)) &= 1 / ell dot sum_(bold(x) in X^ell) cal(L)_q (y(bold(x)) - bra bold(x), bold(beta) ket) \
      &-> min_(bold(beta)).
    $
  ],
  [== Neural quantile regression
    Neural networks inherently support custom loss functions and can model conditional quantiles $QQ_q [Y|X]$ as
    well (@fig-neural-quantile-regression). A model predicting conditional quantiles $QQ_q [Y|X]$ must
    be trained with a quantile loss, which can be easily implemented:

    #margin[#figure(
        caption: [Quantile regression performed by a neural network],
        quantile-model-plot("nn/out.csv"),
      ) <fig-neural-quantile-regression>
    ]

    ```python
    class QuantileLoss(L.LightningModule):
        def __init__(self, q: float):
            super().__init__()
            self.q = q

        def forward(self, y_pred, y_true):
            return T.where(
                (epsilon := y_true - y_pred) >= 0,
                self.q * epsilon,
                (self.q - 1) * epsilon,
            ).mean()```
  ],

  [== Gradient boosting quantile regression
    Quantile loss @eq-check-loss is differentiable if $epsilon != 0$:

    #margin[#figure(
        caption: [Quantile regression performed by a gradient boosting model],
        quantile-model-plot("boosting/out.csv"),
      ) <fig-boosting-quantile-regression>
    ]

    $
      pdv(, epsilon) cal(L)_q (epsilon) = cases(q & quad epsilon > 0, -(1-q) & quad epsilon < 0
      ) med ,
    $

    thus, gradient boosting can approximate the quantile function $QQ_q [Y|X]$ to handle non-linear
    dependencies between features and quantiles (@fig-boosting-quantile-regression).],
  [== Random forest quantile regression
    Random forests use ensemble averaging to make final predictions:

    $
      A(bold(x)) = 1 / T dot sum_(t=1)^T a_t (bold(x)).
    $

    If each base algorithm $a_t (bold(x))$ is trained to predict quantiles $QQ_q [Y|X]$, the ensemble $A(bold(x))$ will
    estimate the expectation of the quantile $Ex [QQ_q [Y|X]]$.],
)

= Convergence and reliability of quantile regression parameters

== Linear quantile regression
For linear quantile regression @eq-linear-quantile-regression, the conditional quantile $QQ_q [Y|X]$ is
modeled as a linear function of predictors $bold(x)$. The theoretical properties of linear quantile
parameters $hat(bold(beta))(q)$ such as convergence and variance can be derived, though the analysis
is more complex than for traditional Gaussian regression.

== Parameter expectation
All regression coefficients $beta_j = beta_j (q)$ are functions of $q$. Under appropriate conditions
(independent observations with finite second moments), the asymptotic distribution of the quantile
regression estimator $hat(bold(beta))(q)$ is unbiased:

$ hat(bold(beta))(q) -> bold(beta)(q), $

i.e., theoretically the estimator $hat(bold(beta))(q)$ converges to the expected value of the
parameter $bold(beta)(q)$ as the sample size $ell$ approaches infinity:

$ hat(bold(beta))(q) -> Ex[bold(beta)(q)]). $ <eq-quantile-linear-parameter-expectation>

== Parameter variance
The estimator $hat(bold(beta))(q)$ is asymptotically normally distributed with variance#margin[and mean $bold(beta)(q) = Ex[bold(beta)(q)]$ according to @eq-quantile-linear-parameter-expectation]

$
  Var[bold(beta)(q)] -> ub(1/ell, "I") dot ub(q dot (1-q), "II") dot ub(D^(-1) Omega D^(-1), "III").
$ <eq-quantile-linear-parameter-variance>

The variance in @eq-quantile-linear-parameter-variance depends on three terms:

+ The 1st multiplier determines the convergence rate of the estimator $hat(bold(beta))(q)$ as a
  function of the sample size $ell$; the larger the sample size, the smaller the variance.

+ The 2nd multiplier depends on the quantile $q$. As $q$ approaches the tails (0 or 1), this term
  decreases, which would seemingly lower the variance. It reduces variance if isolated, however, this
  is not the primary contributor to overall variance.

  #margin(
    figure(
      caption: [
        $q dot (1-q)$ term in @eq-quantile-linear-parameter-variance reaches its maximum at $q = 0.5$
      ],
      {
        let x = lq.linspace(0, 1)
        lq.diagram(
          width: 4cm,
          height: 2cm,
          xlabel: $q$,
          ylabel: $q dot (1-q)$,
          lq.plot(mark: none, x, x.map(q => q * (1 - q))),
        )
      },
    ),
  )

  + The 3rd multiplier is the sandwich variance estimator, which depends on both the estimated
    parameters $hat(bold(beta))(q)$ and the robust variance matrix $Omega$. Typical formulations
    include:

    $
      D(hat(y)_q) = 1 / ell dot sum_(bold(x) in X^ell) hat(f)_(Y|X) (hat(y)_q (bold(x))) dot bold(x) bold(x)^Tr
    $

    $
      hat(Omega) = 1 / ell dot sum_(bold(x) in X^ell) (q - Ind(y(bold(x)) <= hat(y)_q (bold(x)) )) dot bold(x) bold(x)^Tr
    $

Consequently, the *variance of estimated parameters $hat(bold(beta))(q)$ increases as $q$
approaches 0 or 1*. In practice, predictions near the median are typically more precise, while
predictions for extreme quantiles (e.g., 0.01 or 0.99) are less reliable.

#note[While $q dot (1-q)$ decreases near the tails, the sandwich term $D^(-1) Omega D^(-1)$ becomes poorly
  estimated and tends to dominate.
]

== Bad statistical guarantee
While ordinary least squares (OLS) estimates benefit from the Gauss-Markov theorem, which
establishes OLS as the best linear unbiased estimator (BLUE) under classical assumptions, quantile
regression follows different asymptotic properties.

Quantile regression estimators remain unbiased and consistent, but their variance behavior is more
complex. As shown in equation @eq-quantile-linear-parameter-variance, the variance depends on both
the quantile level $q$ and the underlying data distribution through the sandwich estimator term $D^(-1) Omega D^(-1)$.

In practice, quantile regression estimates exhibit higher statistical variability than OLS
estimates, particularly for extreme quantiles (e.g., $q < 0.1$ or $q > 0.9$). This occurs because:

1. The sparsity of data in the tails leads to less reliable sandwich term estimation
2. The conditional density at extreme quantiles becomes more difficult to estimate accurately
3. The effective sample size for determining extreme quantiles is effectively reduced

This statistical efficiency trade-off is a necessary cost of gaining robustness to outliers and
insights into the complete conditional distribution.

#note[
  The variance of the quantile regression estimator is larger than that of OLS, especially for extreme
  quantiles.
]

= Interpretation

== Targets
Median (and other quantiles) is sometimes more interpretable and a better measure of centrality than
the mean, particularly for skewed or multimodal data:

- Median salary or house price is more interpretable than mean values as both distributions are
  usually skewed.

  - In some cases, a "mean" prediction does not exist, such as with binary groups (deceased = 0 and
    deceased = 1), where the prediction for an "average" $bold(x)$ is meaningless.

- Quantile regression predictions represent the *conditional quantiles of the response variable* $y$ given
  the predictors $bold(x)$. These predictions act *as boundaries*, splitting the distribution into $q$-lower
  and $(1-q)$-upper portions by value of $y$, much like the median divides the data into two equal
  groups. These predictions do not correspond to any specific object or individual in the training
  data but rather model the distribution of $y$ itself.

== Parameters

- Coefficients $beta$ in linear quantile regression are noisier than in OLS and depend on quantile $q$,
  making them harder to interpret. The Gauss-Markov theorem ensuring convergence and variance in OLS
  does not apply to quantile regression.

- Exact values of $beta$ in OLS are interpretable, but in quantile regression, they are generally not.
  In simple cases, they can be close to OLS coefficients and interpretable. However, when quantile
  regression is applied to transformed data (e.g., $log(y)$), coefficients remain invariant, but their
  contribution to $y' = log(y)$ becomes less obvious. For skewed data where OLS fails, quantile
  regression coefficients differ significantly from OLS but may still be interpretable.

= Robustness of quantile regression

Quantile regression offers several advantages over ordinary least squares (OLS) regression, but also
comes with certain limitations:

== Advantages

=== Non-normality (skew, heavy tails, multimodality)
Quantile regression models *conditional quantiles*, capturing skewed or heavy-tailed distributions
*without relying on normality assumptions*. OLS assumes normality and may produce misleading results
when this assumption is violated.

=== Robustness to outliers and noise
By focusing on quantiles rather than the mean, quantile regression reduces sensitivity to random
noise and outliers, emphasizing specific distributional trends. Quantile regression also *does not
assume any specific noise distribution*. In OLS, a few outliers can have a pronounced effect on
parameter estimates.

=== Complete picture of conditional distributions
Quantile regression allows modeling multiple quantiles, providing a comprehensive view of how
predictors affect the entire conditional distribution of the response, not just its center. This
reveals heterogeneous effects that OLS cannot capture.

=== Handling censored data
Some quantiles can be robustly calculated even for censored data. Censoring can also be accounted
for explicitly by modifying the loss function:

$
  cal(L)_q^"cens" (epsilon) := cases(cal(L)_q (epsilon)\, & delta = 1, -q dot min{epsilon, 0} \, & delta = 0 )
$

where $delta$ is the event indicator (0=censored, 1=real), and $min{epsilon, 0}$ represents
underpredictions.

=== Invariance to monotonic transformations
Quantile regression is *invariant to monotonic transformations* of $y$ like logarithm or square
root. In OLS this is not the case, although transformations are sometimes used to normalize data.

=== Accommodating heteroscedasticity
Quantile regression does not assume homoscedasticity (constant variance). Instead, it models
different parts of the conditional distribution independently, allowing for varying spread (e.g.,
wider or narrower intervals) across predictors. OLS assumes homoscedasticity (or equal weight of all
observations).

== Limitations

=== Computational complexity
Quantile regression requires solving a linear programming problem, which is computationally more
intensive than OLS's closed-form solution, especially for large datasets or when modeling multiple
quantiles.

=== Statistical efficiency
When OLS assumptions are met (normality, homoscedasticity, etc.), OLS estimates are more efficient
(lower variance) than quantile regression estimates.

=== Interpretation challenges
Interpreting multiple quantile regression models simultaneously can be complex, especially when
communicating results to non-technical audiences.

=== Crossing quantiles
In practice, estimated conditional quantile curves may cross, violating the monotonicity property of
quantile functions. This problem becomes more pronounced when modeling many quantiles
simultaneously.

=== Sparse data in tails
Estimates for extreme quantiles (e.g., $q = 0.01$ or $q = 0.99$) are often less reliable due to
sparse data in distribution tails, resulting in higher variance as shown in the parameter
convergence section.

= Interpretation of linear coefficients

== Impact on target variable
Quantile regression coefficients $beta_j (q)$, similar to OLS coefficients $beta_j$, represent the
impact of a unit change in predictor $bold(x)^j$ on the response variable $y$. However, since they
are functions of $q$, these coefficients capture more nuanced relationships between features and the
target variable across different parts of its distribution.

By analyzing how coefficients $beta_j(q)$ vary across different quantile levels, we gain insight
into how predictors differentially affect the entire conditional distribution of $y$, not just its
central tendency.

For instance, if $y$ represents patient survival time, quantile regression illustrates how a feature
influences various segments of the target distribution: high-risk patients ($q << 0.5$, shorter
survival), typical patients ($q approx 0.5$, median survival), or low-risk patients ($q >> 0.5$,
longer survival).

== Data
The ACTG 320 trial, initiated in 1997 by Merck, was designed to evaluate the effectiveness of the
antiretroviral drug indinavir when used in a triple-drug regimen compared to a standard two-drug
treatment for HIV patients.

#margin[#figure(
    caption: [
      ACTG 320 dataset features (simplified)
    ],
    table(
      columns: (auto, 1fr),
      [Variable], [Description],
      [`time` \ (target)],
      [Follow-up time to AIDS progression or death (in days). Represents the time from enrollment to the
        event (end of study or death).],

      [`age`], [Age of the patient at the time of enrollment (in years).],
      [`cd4_cell_count`],
      [Baseline CD4 T-cell count (cells/mL), a key indicator of immune function.],

      [`race_*`], [Indicator variables representing the patient's race.],
      [`group_*`], [Indicator variables representing the treatment group.],
    ),
  ) <tab-aids-320-features>]

The associated dataset contains approximately 1,150 records of HIV-infected patients who were
randomized to receive either the novel triple-drug regimen or the conventional two-drug therapy.

// TODO: Add reference to https://search.r-project.org/CRAN/refmans/GLDreg/html/actg.html

== Quantile regression
The target variable is `time`, representing the follow-up duration. Quantile regression was used to
analyze the impact of various linear predictors $bold(x)^j$ from @tab-aids-320-features on the time $y$ to
AIDS progression or death.

Quantile regression coefficients $beta_j (q)$ as functions of quantile $q$ are plotted in
@fig-aids-quantile-plot.

#figure(
  caption: [
    Quantile regression coefficients for ACTG 320 dataset
  ],
  grid(columns: (1fr, 1fr, 1fr), row-gutter: 1em, ..{
      let data = lq.load-txt(read("aids/out.csv"), header: true)
      let quantile = data.remove("quantile")

      data
        .keys()
        .map(col => (
          {
            let coeffs = data.at(col)
            let lim = calc.max(..coeffs.map(calc.abs))
            lq.diagram(
              ylim: if (col == "intercept") { auto } else { (-lim, lim) },
              xlim: (0, 1),
              ylabel: $beta_#raw(col)$,
              xlabel: $q$,
              width: 2.5cm,
              height: 3cm,
              margin: 15%,
              lq.plot(quantile, coeffs, mark-size: 2pt),
            )
          }
        ))
    }),
) <fig-aids-quantile-plot>

- Low $q$ values represent individuals who progressed to AIDS or died quickly, while high $q$ values
  correspond to individuals with longer survival times.

- Positive $beta_j (q)$ values indicate that the predictor $bold(x)^j$ makes a positive contribution
  to survival time $y$, meaning $Delta y$ increases proportionally to $beta_j (q) dot Delta bold(x)^j$ for
  individuals around the $q$-quantile of the distribution.

- Negative $beta_j (q)$ values suggest that the predictor $bold(x)^j$ makes a negative contribution to
  survival time $y$, meaning $Delta y$ decreases proportionally to $beta_j (q) dot Delta bold(x)^j$ for
  individuals around the $q$-quantile of the distribution.
