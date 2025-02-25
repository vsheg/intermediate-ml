#import "../template.typ": *
#show: template

= Quantile loss

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