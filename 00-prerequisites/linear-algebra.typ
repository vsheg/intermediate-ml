#import "../template.typ": *
#show: template

= Linear algebra

== Linear span

The linear span of a set of vectors $bold(v)_1, dots, bold(v)_k$ is the set of all
possible linear combinations of these vectors. It is denoted by $L(bold(v)_1, dots, bold(v)_k)$.

== Diagonal matrices

Diagonal matrices are special matrices where all non-diagonal elements are zero. They have
important applications in linear algebra and are particularly useful for representing
linear transformations.

Matrix $Lambda$ is diagonal if all its off-diagonal elements are zero:

#let Lambda-full = $mat(lambda_1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, lambda_k)$
#let Lambda-short = $dmat(lambda_1, dots.down, lambda_k)$

$ Lambda = diag(lambda_1, ..., lambda_k) = #Lambda-full equiv #Lambda-short $

where $lambda_1, dots, lambda_k$ are the diagonal elements of $Lambda$. For simplicity,
off-diagonal elements are often omitted.

/ Zero matrix $O$: A diagonal matrix where all diagonal elements are zero:
  $ O = diag(0, dots, 0) = mat(0, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, 0) $

/ Identity matrix $I$: A diagonal matrix where all diagonal elements are one:
  $ I = diag(1, dots, 1) = mat(1, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, 1) $

/ Scalar matrix: A diagonal matrix with constant diagonal elements:
  $ lambda I = diag(lambda, dots, lambda) = mat(lambda, dots.c, 0;dots.v, dots.down, dots.v;0, dots.c, lambda) $

Diagonal matrices have several important properties:

=== Linear independence
A fundamental property of diagonal matrices is that their columns are linearly independent
if and only if all diagonal elements are non-zero. This follows from the fact that any
linear combination of the columns that equals zero must have all coefficients equal to
zero.
#example[
  For any two distinct columns $i != j$ of a diagonal matrix $Lambda$, their dot product is
  zero since they have non-overlapping non-zero elements:

  $ vec(dots.v, hg1(lambda_i), 0, dots.v)^Tr dot vec(dots.v, 0, hg2(lambda_j), dots.v) = 0 $
]

=== Basis set
The columns of a diagonal matrix with non-zero diagonal elements form an orthogonal basis.
Each column $bold(b)_i$ has exactly one non-zero element $lambda_i$ in position $i$:

$ cal(B): wide bold(b)_1 = vec(lambda_1, 0, dots.v, 0), quad bold(b)_2 = vec(0, lambda_2, dots.v, 0), quad ... , quad bold(b)_k = vec(0, 0, dots.v, lambda_k) $

This orthogonal basis has a special geometric meaning - each basis vector points along a
coordinate axis and has length $|lambda_i|$. When $|lambda_i| = 1$ for all $i$, the basis
becomes orthonormal.

Diagonal matrices are well suited for storing basises. If all diagonal elements are equal
to one, the matrix is an identity matrix $I$ and it stores orthonormal basis vectors of
the standard basis.

#example[
  In 3D space, the identity matrix $I$ stores the standard basis vectors:
  $ bold(i) = vec(1, 0, 0), quad bold(j) = vec(0, 1, 0), quad bold(k) = vec(0, 0, 1) $
]

=== Scaling
A diagonal matrix $Lambda$ acts as a scaling transformation, where each coordinate is
scaled independently by its corresponding diagonal element:

$ Lambda bold(v) = vec(lambda_1 v_1, dots.v, lambda_k v_k) $

This represents stretching or compressing the space along each coordinate axis by factors $lambda_1,dots,lambda_k$.

#example[
  $
    Lambda bold(v) = #Lambda-full dot vec(v_1, dots.v, v_k) = vec(lambda_1 v_1 + 0, dots.v, 0 + lambda_k v_k)
  $
]

=== Inverse matrix
The inverse of a diagonal matrix is simply the reciprocal of each diagonal element,
provided none of the diagonal elements are zero.
$ Lambda^(-1) = diag(1/lambda_1, dots, 1/lambda_k) $

=== Commutativity
Diagonal matrices commute with each other under multiplication:
$ Lambda_1 Lambda_2 = Lambda_2 Lambda_1 $

=== Eigenvalues and eigenvectors
The eigenvalues of a diagonal matrix are the diagonal elements themselves, and the
eigenvectors are the standard basis vectors.