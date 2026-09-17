#import "../_template.typ": *
#show: template.with(home: "index.html", assets: "assets")

= Intermediate ML

This is a work-in-progress draft of intermediate-level machine learning materials. Thanks to LLMs for the high quality; any errors are mine.

Topics:

#let item(path, title) = link(path, title)

- #item("1a-information/index.html")[Information theory]
- #item("1b-mle-map/index.html")[Maximum likelihood and maximum _a posteriori_ estimation]
- #item("1c-regularization/index.html")[Regularization]
- #item("2a-gaussian-linear/index.html")[Gaussian linear regression]
- #item("2b-gaussian-nonlinear/index.html")[Gaussian nonlinear regression]
- #item("2c-quantile/index.html")[Quantile regression]
// - #item("2d-poisson/index.html", "Poisson regression")
- #item("3-density/index.html", "Density estimation")
- #item("4-exp-family/index.html", "Exponential family")
- #item("5-glm/index.html", "Generalized linear models")
- #item("6-gam/index.html", "Generalized additive models")
- #item("7-bayesian/index.html", "Bayesian models")
- #item("8-pca/index.html", "Principal component analysis")

For reference:

- #item("9a-linear-algebra/index.html", "Linear algebra")
- #item("9b-distributions/index.html", "Probability distributions")
