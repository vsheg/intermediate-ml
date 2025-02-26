#import "../template.typ": *
#show: template

= Check loss

== Check-loss
Consider an asymmetric loss function parameterized by $q in (0, 1)$:

$
  cal(L)_q (epsilon) := cases(q dot epsilon & quad epsilon >= 0, -(1-q) dot epsilon & quad epsilon < 0
  ) quad ,
$ <eq-check-loss>

#margin[Strictly speaking, this is an estimation of the error: $hat(epsilon) := y - hat(y)$; for
  different estimations of $hat(y)$, there are different $hat(epsilon)$]

where $epsilon := y - hat(y)$ is the error term (residual) and $hat(y)$ is the prediction
of a regression model.

#margin[
  This loss function is also called the _pinball loss_ and _quantile loss_ (more on this
  below)
]

// TODO: add check loss plot

== Empirical risk
The empirical risk $cal(R)$ is the expected value of the loss $cal(L)_q (epsilon)$.

For a pair $(bold(x), y)$ taken from the joint distribution $pdf(bold(x), y)$, a function
(algorithm) $hat(y) = a^*(bold(x))$ that minimizes $cal(R)(a = a^*)$ can be found by
minimizing $cal(R)(a)$:

$ cal(R)(a) &:= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (a(bold(x)), y)] \
          &= Ex_((bold(x), y) ~ pdf(bold(x), y)) [cal(L)_q (y - a(bold(x)))] \
          &= integral cal(L)_q (y - a(bold(x))) dot pdf(bold(x), y) dot dd(bold(x)) dd(y) \
          &= integral cal(L)_q (y - a(bold(x))) dot dd(F(bold(x), y)) -> min_a $

== Constant model
Let's first consider the simplest case, we will look for $a^*$ in a family of all constant
models $a^* in {a | a = const}$:

#let comment(content) = text(fill: ghost-color, size: 0.8em, $   &$ + content)
#let focus(content) = {
  text(fill: accent-color, content)
}

$
  cal(R)(a) &= integral cal(L)_q (focus(epsilon = y - a)) dd(F(bold(x), y)) comment(a(bold(x)) -> a = const) \
            &= integral cal(L)_q (epsilon = y - a) focus(dd(F(y))) comment("as" a "is not a function of" bold(x)) \

            &= limits(integral)_focus(y - a >= 0) cal(L)_q (y - a) dd(F(y)) + limits(integral)_focus(y - a < 0) cal(L)_q (y - a) dd(F(y)) #comment[nonoverlapping regions: $epsilon >= 0$ and $epsilon < 0$] \

            &= limits(integral)_(y >= a ) focus((y - a) dot q) dd(F(y)) - limits(integral)_(y < a) focus((y - a) dot (1-q)) dd(F(y)) -> min_a #comment[expand $cal(L)_q (epsilon)$ according to @eq-check-loss] \
$

== Risk minimization
The integral is split at $a = a^*$ into two independent regions: $(-infinity..a^*)$ and $(a^*..+infinity)$.
By differentiating both integrals over $a$, we can find $a^*$:

$
  pdv(, a) cal(R)(a) &= focus(q) dot integral_(y>=a) pdv(, a) (y - a) dd(F(y)) - focus((1-q)) dot integral_(y < a) pdv(, a) (y - a) dd(F(y)) #comment[constants] \
                     &= -q dot focus(integral_(y = a^*)^(+oo) dd(F(y))) + (1-q) dot focus(integral_(y=-oo)^(a^*) dd(F(y))) comment(dd(F(y)) = pdf(y) dd(y)) \
                     &= -q dot (1 - F_Y (a)) + (1-q) dot F_Y (a) = -q + F_Y (a) comment(F_Y (a) equiv F(Y = a)) \
$

At the extreme point $a = a^*$, the derivative of the risk is zero:

$ -q + F_Y (a^*) = 0. $

Thus, the optimal constant model $a^*$ is the $q$-quantile of the random variable $Y$:

$ a^* = F_Y^(-1) (q) = QQ_q [Y]. $

== Implications
We assumed that $a$ is a constant function of $bold(x)$ and derived the optimal constant
model $hat(y)(bold(x)) = a^*$ that minimizes the empirical risk (expected check-loss) $cal(R)(a)$.
Notably, if we differentiate $cal(R)(a)$ with respect to any general function $a(bold(x))$,
the result remains the same.

Minimizing the check loss $cal(L)_q (epsilon)$ for a regression model $hat(y)(bold(x))$ is
equivalent to finding the $q$-quantile of the random variable $Y$. Therefore, the
algorithm $a^*$ derived from solving the minimization problem $cal(R) = Ex[cal(L)_q] -> min$,
effectively predicts the $q$-quantile of $Y$.

When the algorithm $a$ is a function of $bold(x)$, the prediction is conditional on $X$:

$ hat(y)_q (bold(x)) = QQ_q [Y|X=bold(x)], $

where prediction $hat(y)_q$ depends both on hyperparameter $q$ and on the input $bold(x)$.


In ordinary least squares (LS) regression, we use conditional expectation as prediction:
$ hat(y)(bold(x)^*) equiv Ex[Y|X = bold(x)^*] equiv a(bold(x)^*) $

In quantile regression, we predict $q$-quantile of a random variable:
$ hat(y)_q (bold(x)^*) = QQ_q [Y|bold(x)^*] = a(bold(x)^*), $

where $q$ is a hyperparameter that defines the quantile.

The corresponding error term (residual) in $q$-quantile regression is defined as:
$ hat(epsilon)_q (bold(x)^*) = QQ_q [Y|bold(x)^*] - y(bold(x)^*) = hat(y)_q - y $

== Check-loss
Consider an asymmetric loss function:

$ cal(L)_q (epsilon) := cases(q dot epsilon\, & epsilon >= 0, -(1-q) dot epsilon\, & epsilon < 0
) $

For regression, the empirical risk is the expectation of a loss. A function $a(bold(x)) = a^*(bold(x))$ that
corresponds to a minimum of $cal(R)(a = a^*)$ can be found by minimizing the empirical
risk:

$ cal(R)(a) &:= Ex_((bold(x), y) ~ f(bold(x), y)) [cal(L)(a(bold(x)), y)] \
          &= Ex_((bold(x), y) ~ f(bold(x), y)) [cal(L)(a(bold(x)) - y)] \
          &= integral cal(L)(a(bold(x)) - y) dot f(bold(x), y) dot dd(bold(x)) dd(y) \
          &= integral cal(L)(a(bold(x)) - y) dot dd(F(bold(x), y)) -> min_a $

Let's first consider a simpler case where $a = "const"$. We can show that $a = a^*$ minimizes
this loss if $a^*$ is a $q$-quantile:

$ cal(R)(a) &= integral cal(L)(epsilon = a - y) dd(F(bold(x), y)) \
          &= integral cal(L)(epsilon = a - y) dd(F(y)) \
          &= integral_(a>=y) (a - y) dot q dot dd(F(y)) - integral_(a < y) (a - y) dot (1-q) dot dd(F(y)) -> min_a $

At $a = a^*$, the loss has a discrepancy at the point dividing integrals over $(-infinity..a^*)$ and $(a^*..+infinity)$.
By differentiating both integrals, we can identify $cal(R)(a = a^*) = 0$:

$ pdv(, a) cal(R)(a) &= q dot integral_(a>=y) pdv(, a) (a - y) dd(F(y)) - (1-q) dot integral_(a < y) pdv(, a) (a - y) dd(F(y)) \
                   &= q dot integral_(y = -oo)^(a^*) dd(F(y)) - (1-q) dot integral_(y=a^*)^oo dd(F(y)) \
                   &= q dot F_Y (a^*) - (1-q) (1-F_Y (a^*)) = F_Y (a^*) - q = 0 $

Therefore:
$ a^* = F_Y^(-1) (q) = QQ_q [Y] $

#note[
  The loss can be used to find a quantile of a single random variable by assuming that $a$ is
  a constant prediction.
]

For regression, we use the conditional quantile:
$ a^*(bold(x)) = F_(Y|X=bold(x))^(-1) (q) = QQ_q [Y|X=bold(x)] $

This gives us a prediction $hat(y)_q$ that corresponds to the $q$-quantile of $Y$ conditional
on $X$.