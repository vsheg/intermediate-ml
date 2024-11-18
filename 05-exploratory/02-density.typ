#import "@preview/physica:0.9.3": *
#import "../defs.typ": *
#import "../template.typ": *

#show: template

= Density estimation

Density estimation is a fundamental problem in statistics and machine learning. It is used
to estimate the probability density function (pdf) of a random variable from a sample of
data. The goal is to find a function that best fits the data and can be used to generate
new samples.

== Parametric density estimation

Parametric density estimation assumes that the data comes from a known distribution with a
fixed number of parameters. The most common parametric models are the normal distribution.

=== Univariate normal distribution
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

=== Uncorrelated multivariate normal distribution
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

=== Covariance matrix
#note[Diagonal matrices $Lambda$ are useful because operations with them are simple.
  $ Lambda = dmat(lambda_1, dots.down, lambda_k) $

  + Multiplying a vector $bold(v)$ by a diagonal matrix scales each component of the vector by
    the corresponding diagonal element.
    $ Lambda bold(v) = vec(lambda_1 v_1, dots.v, lambda_k v_k) $
  + The inverse of a diagonal matrix is simply the reciprocal of each diagonal element,
    provided none of the diagonal elements are zero.
    $ Lambda^(-1) = diag(1/lambda_1, dots, 1/lambda_k) $
  + Diagonal matrices commute with each other under multiplication:
    $ Lambda_1 Lambda_2 = Lambda_2 Lambda_1 $
  + The eigenvalues of a diagonal matrix are the diagonal elements themselves, and the
    eigenvectors are the standard basis vectors.]
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
#note[
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