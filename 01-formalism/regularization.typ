#import "../template.typ": *
#show: template

= Regularization

== Overview

Regularization is a technique that imposes constraints on model parameters to control the
solution space. By limiting where parameters can be found, it effectively shrinks the
space of possible solutions. This reduction in parameter flexibility not only enhances
model generalizability but also helps prevent overfitting by focusing on simpler
solutions.

The term derives from Latin "regula" (rule) and "regularis" (in accordance with rules),
reflecting its role in establishing systematic constraints on model behavior.

Through these controlled parameter constraints and reduced solution space, regularization
helps create simpler, more robust models by reducing their sensitivity to noise in the
training data.

== Probabilistic interpretation of regularization

=== Probabilistic framework
Consider a joint distribution of data $bold(x) in RR^k, y in RR$ and model's parameters $bold(theta) in RR$:

$ bold(x), y, bold(theta) tilde.op X, Y, Theta. $

+ The prior distribution of $bold(x)$ is independent of parameters $bold(theta)$ and can
  assumed to be uniform and ignored in the model:
  $ X ~ f_X (bold(x)|bold(theta)) :> f_X (bold(x)) ~ cal(U). $

+ The posterior distribution of responses $y$ depends on parameters $bold(theta)$ and
  specific data point $bold(x)'$, following a semi-probabilistic model formalism. The model
  is specified by defining the conditional distribution of responses $Pr[y|bold(x) = bold(x)^*, bold(theta)]$ given
  a specific $bold(x) = bold(x)^*$ and model parameters $bold(theta)$. When the parameters
  are fitted, we make predictions for new data points $bold(x)'$ by maximizing the
  probability of a response $y$ given $bold(x)'$:

  $ a_(bold(theta)) (bold(x)') = arg max_(y in supp Y) ub(f_Y (y|bold(x) = bold(x)', bold(theta) = hat(bold(theta))), "model"). $

+ The prior distribution of parameters $bold(theta)$ is assumed to be known and defined by
  the hyperparameter vector $bold(gamma)$:
  $ f_Theta (bold(theta)) :> f_Theta (bold(theta)|bold(gamma)) ~ Theta(bold(gamma)) $

#note[
  We ommit random variables $X, Y, Theta$ in the pdf's underscripts for brevity. Just look
  at the arguments before the bar to understand to which random variable the pdf refers:
  e.g. $f(x, y|theta)$ means $f_(X,Y) (bold(x),y|theta)$.
]

#note[
  $
    Pr[y|x, theta] &= Pr[{Y = y}|{X = x}, {Theta = theta}] \
                   &= Pr[{Y = y}|{X = x}{Theta = theta}] \
                   &= Pr[{Y = y}{X = x}{Theta = theta}] / Pr[{X = x}{Theta = theta}] \
                   &= Pr[x, y, theta] / Pr[x, theta]
  $
]

=== Applying MAP
The joint distribution of data and parameters can be rewritten as a product of conditional
pdfs:

$
  f(bold(x), y, bold(theta)) &= f(y|bold(x),bold(theta)) dot f(bold(x), bold(theta)) \
                             &= f(y|bold(x), bold(theta)) dot cancel(f(bold(x)|bold(theta))) dot f(bold(theta)|bold(gamma)) \
                             &= f(y|bold(x), bold(theta)) dot f(bold(theta)|bold(gamma))
$

As it was mentioned, the canceled prior distribution of data $f (bold(x)|bold(theta))$ is
independent of the model parameters $bold(theta)$. We ignore it (or assume uniform).

Still, we didn't ignore the prior distribution of parameters $bold(theta)$, which is $f(bold(theta)|bold(gamma))$.
Because of that, it's MAP (Maximum a Posteriori) estimation, not MLE (Maximum Likelihood
Estimation).

#note[
  A pdf $f(y|x, theta)$ becomes a likelihood function when we consider it as a function of
  arguments behind the bar, e.g.
  - $h(y) := f(y|x = x^*, theta = theta^*)$ is still a pdf of $y$ given $x = x^*$ and $theta = theta^*$.
  - $g(theta) := f(y = y^*|x = x^*, theta)$ is already a likelihood function of $theta$ given $x = x^*$ and $y = y^*$.
]

=== Finding parameters
For specific training samples $y^*, bold(x)^*$ and predefined hyperparameters $bold(gamma)^*$,
we write the joint likelihood of data and model parameters and maximize it:

$
  ell(bold(theta))
    &=
  log product_((y^*, bold(x)^*) in (X, Y)^ell) ub(
    f(y = y^*|bold(x)=bold(x)^*, bold(theta)) dot f(bold(theta)|bold(gamma)=bold(gamma)^*),
    "MAP",

  ) \

    &=
  sum_((y^*, bold(x)^*) in (X, Y)^ell) ub(log f(y = y^*|bold(x)=bold(x)^*, bold(theta)), "log-likelihood") + ub(log f(bold(theta)|bold(gamma)=bold(gamma)^*), "prior reguralizer")
  -> max_(bold(theta))
$

The last term is the regularizer, its strength determined by the model hyperparameters.
