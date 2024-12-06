#import "../template.typ": *
#show: template
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

