#import "../template.typ": *
#show: template

= Normal distribution

== Univariate
A random variable $xi$ is said to have a normal distribution with mean $mu$ and variance
$sigma^2$ if its probability density function (pdf) is given by

$
  f_xi (x) = 1 / (sigma sqrt(2 pi)) exp{ -1/2 ((x - mu) / sigma)^2 }
$

where $mu$ is the mean and $sigma^2$ is the variance of the distribution. More compactly,
it can be written as

$
  xi tilde cal(N)(mu, sigma^2)
$

== Uncorrelated multivariate
A random vector $bold(xi) = vec(xi_1, dots.v, xi_k)$ is said to have an uncorrelated
multivariate normal distribution with mean $bold(mu) = vec(mu_1, dots.v, mu_k)$ and
variances $sigma_1^2, dots, sigma_k^2$ if the pdf of every random component of $bold(xi)$
is given by

$
  f_(xi_j) (x) = 1 / (sigma_j sqrt(2 pi)) exp{ -1/2 ((x - mu_j) / sigma_j)^2 }
$

where $mu_j$ is the mean and $sigma_j^2$ is the variance of the $j$-th component of the.

All components of $bold(xi)$ are assumed to be independent, so the joint pdf of $bold(xi)$ is
the product of the pdfs of its components:

$
  f_bold(xi) (x_1, ..., x_k) &= product_(i=1)^k f_(xi_i) (x_i) \
                             &= product_(i=1)^k 1 / (sigma_i sqrt(2 pi)) exp{ -1/2 ((x_i - mu_i) / sigma_i)^2 }
$

== Covariance matrix
All variance parameters $sigma_1^2, dots, sigma_k^2$ can be combined into a covariance
matrix $Sigma$. The covariance matrix is a symmetric positive definite matrix that
describes the covariance between the components of $bold(xi)$.

$
  Sigma = dmat(sigma_1^2, dots.down, sigma_k^2)
$

Here, the covariance matrix is diagonal (all off-diagonal elements are zero), because we
assumed that the components of $bold(xi)$
are uncorrelated, i.e., $Cov[xi_i, xi_j] = 0$ for all $i != j$.

The pdf of the multivariate normal distribution can be written in terms of the covariance

$
  f_bold(xi) (x_1, ..., x_k) = exp{ -1/2 (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu)) } / sqrt((2 pi)^k det Sigma)
$<multivariate-normal-distribution>

The covariance matrix $Sigma$ above is a diagonal matrix, but in general, it's a symmetric
positive definite matrix that describes the covariance between the components of $bold(xi)$:

$ Sigma := mat(
  Cov[xi_1, xi_1], dots, Cov[xi_1, xi_k];dots.v, dots.down, dots.v;Cov[xi_k, xi_1], dots, Cov[xi_k, xi_k]
). $

If we substitute the non-diagonal covariance matrix $Sigma$ into the pdf, we get the
general form of the multivariate normal distribution.

Technically, each component of $Sigma$ is the covariance between the corresponding
components
#margin[
  For a sample $X = {x_1, ..., x_ell} subset RR$, the variance is the average of the squared
  differences from the mean:
  $ Var[X] := 1 / ell sum_(i=1)^ell (x_i - macron(x))^2. $

  Given another sample $Y = {y_1, ..., y_ell} subset RR$, the _co_-variance between two
  samples is characterized by how much they vary together:
  $
    Cov[X, Y] := 1 / ell sum_(i=1)^ell (x_i - macron(x)) dot (y_i - macron(y)).
  $

  Both per sample variance and two samples covariance can be combined into a covariance
  matrix.
  $
    Sigma = mat(Cov[x, x], Cov[x, y];Cov[y, x], Cov[y, y]) = mat(Var[x], Cov[x, y];Cov[y, x], Var[y]).
  $
  It will be shown below that this is equivalent to the covariance matrix for a sample of 2D
  vectors $bold(v)_i = vec(x_i, y_i) in RR^2$.
]
#margin[
  To characterize _co_-variance of multiple samples
  $
    X_1 & = {x_(1,1), ..., x_(1,ell)}, quad
    ..., quad
    X_k & = {x_(k,1), ..., x_(k,ell)}
  $
  all together, we combine them into one sample of $k$-dimentional data:
  $ V = {bold(v)_1, ..., bold(v)_ell}, quad bold(v)_i = vec(x_(1,i), dots.v, x_(k,i)). $

  The covariance between any two samples $X_t$ and $X_q$ is
  $ Cov[X_t, X_q] := 1 / ell sum_(i=1)^ell (x_(t,i) - macron(x)_t) dot (x_(q,i) - macron(x)_q). $

  Generally, for a sample of vectors $bold(v)_1, ..., bold(v)_ell in RR^k$:
  $
    Sigma :&= 1/ell sum_(i=1)^ell (bold(v)_i - macron(bold(v))) (bold(v)_i - macron(bold(v)))^Tr \
           &= 1/ell sum_(i=1)^ell Cov[bold(v)_i, bold(v)_i] \
           &= Ex[(bold(v) - macron(bold(v))) (bold(v) - macron(bold(v)))^Tr].
  $
  which resambles the variance but in multiple dimensions.
]
$ Sigma_(i,j) := Cov[xi_i, xi_j] = Ex[(xi_i - mu_i) (xi_j - mu_j)]. $

The term $det Sigma$ is the generalized variance.

== Mahalanobis distance
The distance between a point $bold(x)$ and the distribution $cal(N)(bold(mu), Sigma)$ can
be measured using the Mahalanobis distance.
#margin[
  Quadratic form $Q(bold(x))$ is a scalar function of a vector $bold(x)$ that can be
  expressed as as weighted sum of the squares of the components of $bold(x)$:

  $ Q(bold(x)) = sum_(i,j) w_(i,j) x_i x_j. $

  These weights can be gathered into a matrix $W$, and the quadratic form can be written as
  a matrix product:

  $ Q(bold(x)) = bold(x)^Tr W bold(x). $
]

The premise is that the covariance matrix $Sigma$ captures the correlations between the
components of $bold(xi)$. The Mahalanobis distance is a measure of how many standard
deviations away a point $bold(x)$ is from the mean $bold(mu)$, taking into account the
correlations between the components of $bold(xi)$.

We can define a quadratic form
$
  Q(bold(x)) :&= (bold(x) - bold(mu))^Tr Sigma^(-1) (bold(x) - bold(mu)) \
              &= sum_(i,j) (x_i - mu_i) (Cov[xi_i, xi_j])^(-1) (x_j - mu_j).
$
it can be interpreted