#import "../template.typ": *
#show: template

= Exponential Family 1: Canonical form (1D)

== Canonical form 1
The exponential family represents a parametric class of probability distributions defined
by their probability density function (pdf) or probability mass function (pmf):

$ f(xi|theta) := 1/Z(theta) dot h(xi) dot e^(theta dot xi), $<eq-exp-family-1d>

where $xi in RR$ represents a value of random variable $Xi$, $theta in RR$ is a parameter, $Z(theta) in RR$ represents
a normalization constant, and $h(xi) in RR$ controls the distribution shape. In short
notation, $Y tilde Exp(theta).$
#margin[
  This family encompasses many common probability distributions. Any distribution whose pdf
  can be expressed in the form of @eq-exp-family-1d belongs to the exponential family.
]

The equation @eq-exp-family-1d is the _canonical form_ of the exponential family. The
canonical form provides a standardized way to express all exponential and pre-exponential
terms.

== Partition function
To hold normalization, the term called the _partition function_ is introduced:

$ Z(theta) := integral h(xi) dot e^(theta dot T(xi)) dd(xi). $

Corresponding logarithm $A(theta) := log Z(theta)$ is called the _log partition function_ or _cumulant function_.

== Sufficient statistics
If the random variable $Xi$ does not have a linear relationship with the parameter $theta$,
a function called _sufficient statistics_ $T(xi)$ is introduced to make the relationship
linear:
#margin[Technically, $T(xi)$ is a new random variable $xi'$ for which @eq-exp-family-1d holds.]

$ f(xi|theta) := 1/Z(theta) dot h(xi) dot e^(theta dot T(xi)), $<eq-sufficient-statistics-1d>

== Canonical form 2
Equivalently to @eq-sufficient-statistics-1d, the exponential family can be rewritten as a
single exponential function when all pre-exponential terms are gathered:

$
  f(xi|theta) := e^( theta dot xi - A(theta) + C(xi))
$

where $A(theta) := log Z(theta)$ is the log-partition (cumulant) function, and $C(xi) := log h(xi)$
controls the shape of the distribution. Both forms are canonical as they are equivalent.

== Fitting parameter $theta$
For a data points $x^* ~ Exp(theta)$, we can estimate $theta$ by standard approaches, e.g.
by maximizing the likelihood function:

$
  theta^* = arg max_theta ell(theta) &= arg max_theta { log product_(x^* in X^ell) Pr[x = x^*|theta] } \
                                     &= arg max_theta sum_(x^* in X^ell) log { 1/Z(theta) dot h(x^*) dot e^(theta dot T(x^*)) }\
                                     &= arg max_theta sum_(x^* in X^ell) {-log Z(theta) + log h(x^*) + theta dot T(x^*)} -> max_theta.
$
the terms $log h(x^*)$ are constant and can be ignored.

== Modeling
While 1D exponential family can be used to model 1D densities, relationships between two
variables $x$ and $y$ still can be modeled. If we assume that $y$ has an exponential
family distribution $y ~ Exp(theta)$, and joint distribution is in the form of $f_(X,Y)(x, y|theta)$:
#margin(
  title: [Assumptions],
)[
  Distributions between $X$ and $Theta$ are independent (inputs do not depend on the
  parameter), the distribution of $X$ is assumed uniform (data density is constant); as a
  result, the distribution of answers $Y$ is conditioned both by the input $X$ and the
  parameter $theta$ of the exponential family.
]

$
  f_(X,Y)(x, y|theta) &= (f_(X,Y,Theta)(x, y, theta)) / (f_Theta (theta)) = (f_Y (y|x, theta) dot f_(X, Theta)(x, theta)) / (f_Theta (theta)) \
                      &= (f_Y (y|x, theta) dot f_X (x) dot cancel(f_Theta (theta))) / cancel(f_Theta (theta)) = f_Y (y|x, theta).
$<eq-exp-family-modeling-1d>

If $f_Y (y|x, theta)$ can be expressed as $f_Y (y|theta(x))$, then $y ~ Exp(theta(x))$.

= Exponential family 2: Canonical form ($n$D)

== Vector parameter $bold(theta) in RR^m$
The scalar parameter $theta in RR$ combines with sufficient statistics $T(xi) in RR$ to
produce a scalar value $theta dot T(xi) in RR$ within the exponential function $e^(theta dot T(xi))$.

Generalizing $theta$ to a vector $bold(theta) in RR^n$ requires only that the inner
product
$bra bold(theta), T (xi) ket$ exists, where $T (xi) in RR^m$ maps the random variable $Xi$
into the same space $RR^m$ where $bold(theta)$ resides:

$ e^(theta dot T(xi)) :> e^(bra bold(theta), T(xi) ket). $

== Random vector $bold(xi) in RR^k$
The generalization from scalar random variable $Xi$ to random vector $bold(xi) in RR^k$ follows
naturally through sufficient statistics $T: RR^k -> RR^m$ that ensures the inner product
$bra bold(theta), T(bold(xi)) ket$ exists.
#margin[
  While dimensions of $bold(xi)$ and $bold(theta)$ need not match, the sufficient statistics $T$
  must create a valid inner product $bra bold(theta), T(bold(xi)) ket$.
]

== Canonical form
A random vector $bold(xi) in RR^k$ follows the exponential family distribution with
parameter
$bold(theta) in RR^m$ when its pdf takes the form:
#margin[
  These equivalent canonical forms relate through $A(bold(theta)) = log Z(bold(theta))$ and
  $C(bold(xi)) = log h(bold(xi))$.
]

$
  f(bold(xi)|bold(theta))
    &:= 1/Z(bold(theta)) dot h(bold(xi)) dot e^(bra bold(theta), T(bold(xi)) ket) \
    &:= exp lr({ bra bold(theta), T(bold(xi)) ket - A(bold(theta)) + C(bold(xi)) }, size: #200%).
$<eq-exp-family>

The remaining terms generalize naturally:

- Partition function:
  $ Z(bold(theta)) := integral_supp(bold(x)) h(bold(xi)) dot e^(bra bold(theta), T(bold(xi)) ket) dd(bold(xi)), $
  where $dd(bold(xi)) = dd(xi_1) ... dd(xi_k)$ represents the differential volume element.

- Log partition function:
  $ A(bold(theta)) := log Z(bold(theta)). $

- Shape function and its logarithm must be defined for vector argument:
$ h(xi) :> h(bold(xi)), wide C(xi) :> C(bold(xi)). $

== Modeling scalar response $y in RR$
Given a joint distribution of $k$-dimensional inputs $bold(x)$ and scalar responses $y$,
we can model their relationship analogous to @eq-exp-family-modeling-1d:
#margin[
  Note that $y$ is a scalar random variable, with only the parameters being vectors:
  $y ~ Exp(bold(theta))$.
]

$ f_(X, Y) (bold(x), y|bold(theta)) = f_Y (y|bold(x), bold(theta)) $

For training data $(bold(x)^*, y^*) in (X, Y)^ell$, we estimate $bold(theta)$ by
maximizing:

$
  ell(bold(theta)) = sum_((bold(x)^*, y^*) in (X, Y)^ell) log f_Y (y = y^*|bold(x) = bold(x)^*, bold(theta)) -> max_bold(theta).
$

For prediction on new data $bold(x')$, we calculate the conditional expectation:

$ hat(y)(bold(x')) = Ex[y|bold(x) = bold(x'), bold(theta) = bold(theta)^*]. $

== Modeling all responses $bold(y) in RR^ell$
A straightforward approach collects all responses $y(bold(x))$ for $bold(x) in X^ell$ into
a column vector $bold(y) = row(y(bold(x)_1), ..., y(bold(x_ell)))^Tr in RR^ell$. The
notation
$bold(y) ~ Exp(bold(theta))$ indicates each training example $y in bold(y)$ shares a
common parameter $bold(theta)$.

== Modeling a single vector response $bold(y) in RR^m$
For vector-valued responses, each $bold(y)$ represents multiple outputs for a single input
$bold(x) in RR^k$. The joint distribution follows:

$ f_(X, Y) (bold(x), bold(y)|bold(theta)) = f_Y (bold(y)|bold(x), bold(theta)) $

Different responses $bold(y)$ can be modeled with a shared vector $bold(theta)$ or
individual parameters:

$ bold(y)_1, ..., bold(y)_ell ~ Exp(bold(theta)_1), ..., Exp(bold(theta)_ell). $

= Exponential family 3:
= GLM: Cross-entropy and log-loss

== Model
Logistic regression represents a special case of GLM where the binary response variable
$Y$ follows a Bernoulli distribution: $
  y_i tilde cal(B)(p), quad p := Pr[y_i = 1]
$
Here,
$p$ represents the success probability in a single trial. The canonical form of the
Bernoulli distribution is: $
  y_i tilde f(y|theta) = sigma(-theta) dot e^(theta dot y), quad sigma(theta) = 1 / (1 +
  e^(-theta))
$
Starting from the general GLM form:
$
  Y tilde f(y|bold(theta)) = exp[bold(theta) dot T(y) - A(bold(theta)) + C(y)]
$
We can derive both cross-entropy and log-loss directly, assuming only the Bernoulli
distribution of
$Y$.

== Link Function
The link function $psi$ connects the response variable's mean $mu = Ex[Y]$ to the
distribution's canonical parameters $bold(theta)$: $ bold(mu) = psi(bold(theta)) $
In GLM, we assume the canonical parameters are linear:
$ theta_i = bold(x)_i^Tr bold(beta), quad bold(theta) = X bold(beta) $
where
$bold(beta)$ represents the linear coefficients corresponding to features in $bold(x)$.

For the Bernoulli distribution, the link function takes the form: $
  psi(mu) = log mu/(1-mu) = "logit" mu
$
#columns(
  2,
)[
  == Cross-entropy Loss
  We begin with the log-likelihood function $l(theta)$ for the Bernoulli-distributed
  response variable $Y$, assuming $theta = bold(x)^Tr bold(beta)$:

  $
    l(theta) &= log product_i f(y_i|theta) \
             &= log product_i sigma(-theta) dot e^(theta dot y_i) \
             &= sum_i {theta dot y_i + log sigma(-theta)} \
             &= sum_i {theta dot y_i + log 1 / (1+e^(-(-theta)))} \
             &= sum_i {y_i log mu / (1 - mu) + log 1 / (1 + mu / (1 - mu))} \
             &= sum_i {y_i log mu / (1 - mu) + log (1 - mu) / (1 - mu + mu)} \
             &= sum_i {y_i log mu - y_i log (1 - mu) + log (1 - mu)} \
             &= sum_i {y_i log mu + (1 - y_i) log (1 - mu)} \
             &= sum_i {y_i log p + (1 - y_i) log (1 - p)} \
             &= l(p(bold(beta))) -> max_(bold(beta))
  $

  #colbreak()

  == Log-loss
  The log-loss function $ell(M)$ can be derived by taking the negative log-likelihood:

  $
    -l(theta) &= -sum_i {theta dot y_i + log e^(-theta) / (1+e^(-theta))} \
              &= sum_i cases(
      -log e^(theta) + log e^(-theta) / (1+e^(-theta)) \, &"if" y = 1,
      -log e^(-theta) / (1+e^(-theta)) \, "if" y = 0,

    ) \
              &= sum_i cases(log(1+e^(-theta)) \, "if" y = 1, log(1+e^(theta)) \, "if" y
    = 0) \
              &= sum_i log(1 + e^(theta dot sgn y_i)) \
              &= sum_i log(1 + e^(bra bold(x)_i, bold(beta) ket dot sgn y_i)) \
              &= sum_i log(1 + e^(-M_i)) \
              &= ell(M(bold(beta))) -> min_(bold(beta))
  $
]

== Making Predictions
To make a prediction:
$
  hat(p)(bold(x)) = mu(bold(x)) = psi(theta = bold(x) dot bold(beta)) = 1 / (1 + e^(bold(x)
  dot bold(beta)))
$

