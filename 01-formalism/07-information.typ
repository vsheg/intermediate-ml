#import "@preview/physica:0.9.3": *
#import "../defs.typ": *
#import "../template.typ": *

#show: template

= Information Theory

== Hartley function

=== Bits.
When there are multiple possible outcomes, we can distinguish between them if we have the
necessary information. The minimal amount of information is 1 bit. *By definition*, each
bit of information distinguishes between 2 possibilities. For example, 1 bit of
information is required to unambiguously identify the sex of a child. The event:

$ B = {"Masha gave birth to a boy"} $

corresponds to exactly 1 bit of information, and the inverse event similarly corresponds
to 1 bit of information:

$ macron(B) = {"Masha gave birth to a girl"} $

=== Additivity.
For two *independent and equally probable* events:

$ B = {"Masha gave birth to a boy"}, quad G = {"Lena gave birth to a girl"} $

we expect the total amount of information received when both events have occurred to be
additive:

$ I(A B) = I(A) + I(B) $

Since the logarithm satisfies this property, the possible choice is:

$ log f(A B) = log f(A) + log f(B) $

=== Probability.
The probability $p$ characterizes the frequency of an event. An event with a high
probability provides a small amount of information. For instance, the probability of the
sunrise is nearly one, so the information that there will be a sunrise tomorrow carries
little value. In contrast, the information that there will be no sunrise tomorrow conveys
a significant amount of information. Thus, *the lower the probability, the more
information is conveyed*:

$ I(A) = I(Pr[A]) wide "and" wide p arrow.t <=> I arrow.b. $

The appropriate formula that satisfies these conditions is:

$
  I(f(A)) = log 1/Pr[A], quad f(A) = Pr(A).
$

$ I(A B) = log 1 / Pr[A] + log 1 / Pr[B] $

=== Inverse probability.
The inverse probability $1/p$ represents the *expected number of trials* needed to achieve
*one occurrence* of an event with probability $p$. For example, if $p = 0.01$, the event
occurs, on average, once every 100 trials.

=== Information content.
#note[
  In $k$-valued logic, each $k$-valued digit (0, 1, ..., $k-1$) represents information:
  - In binary logic, each digit is a bit (0 or 1).
  - In ternary logic, each digit is a trit (0, 1, or 2).
]
Information is the *capacity to distinguish* between possibilities. Each bit of
information distinguishes 2 possibilities, and it can assume 2 different values, 0 and 1. $n$ bits
of information distinguish $2^n$ possibilities. Hence, the amount of information required
to distinguish between $2^n$ possibilities is $n$ bits.

If there are $N$ outcomes, each time you assign a bit value to an outcome, you divide all
outcomes into 2 sets corresponding to the bit values:

$ Omega = ub({ omega in Omega | omega"'s bit value" = 0 }, Omega_0)
union ub({ omega in Omega | omega"'s bit value" = 1 }, Omega_1) $

Now, for any $omega$, knowing the corresponding bit value allows you to determine whether $omega in Omega_0$ or $omega in Omega_1$,
thereby halving the uncertainty.

Repeating this $I$ times, you partition $Omega$ into $2^I$ disjoint sets, or more
precisely, $min(2^I, N)$, as $Omega$ contains only $N$ elements:

$ Omega = { omega_1 } union { omega_2 } dots union { omega_N } $

Once the sets contain only one element, further bits do not provide additional meaningful
information. Therefore, the amount of information is proportional to the size of $Omega$.
Each bit splits $Omega$ into 2 parts, and each subsequent bit continues dividing the sets
into 2 parts. However, it is only meaningful to repeat these binary divisions up to $log_2 N$ times.

Thus, the exact number of bits needed to distinguish all outcomes is:

$
  I = log_2 N = log_2 1 / p.
$

== Shannon self-information

Self-information, introduced by Claude Shannon, quantifies the amount of information or "surprise"
associated with the occurrence of an event. The key properties of Shannon's
self-information are:

- An event with a probability of 100% is unsurprising and thus carries no information.
- Events that are less probable yield more information when they occur.
- For two independent events, the total information is the sum of their individual
  self-informations.

The self-information for an event $A$ is defined as:

$ I(A) := lg_2 1 / Pr[A]. $

For a random variable $X$ taking a specific value $x$ with probability $Pr[X = x]$, the
self-information is:

$ I_X (x) := lg_2 1 / Pr[X = x]. $

== Odds ratio

The *odds* of an event $A$ is defined as the difference in self-information (also known as
surprisal) between the event $A$ and its complement $macron(A)$:

$
  "Odds" A :&= I(A) - I(macron(A)) \
           &= log Pr[A] / (1 - Pr[A]).
$

== Shannon entropy
#note(
  title: [Boltzmann distribution],
)[
  maximizes thermodynamic probability $W$ and provides the *probability for each state*, so
  the Boltzmann formula defines a *probability distribution*:

  $ n_i / N = e^(-beta E_i) / (sum_i e^(-beta E_i)). $

  Shannon's information entropy can be applied to this probability distribution:

  $ p_i := e^(-beta E_i) / Z, quad Z := sum_i e^(-beta E_i), $

  $ H &= sum_i p_i dot ln p_i \
    &= sum_i e^(-beta E_i) / Z dot (-beta E_i - ln Z) \
    &= - beta sum_i ub(e^(-beta E_i) / Z, p_i) dot E_i - ln Z dot ub(sum_i e^(-beta E_i)/Z, =1) \
    &= - beta bra E ket - ln Z $

  As a result, we get statistical entropy expressed via the partition function:

  $ S_"stat" = k_B dot H $

  The amount of information needed to encode the probability distribution by energy levels
  can be calculated via Shannon's formula. This amount of information directly corresponds
  to the statistical thermodynamic entropy with $k_B$ units, which translates bits into
  energy per temperature units.
]

=== Information associated with a probability distribution.
Information corresponds to the amount of uncertainty: the more uncertain (less probable)
an outcome is, the more information it carries.

As for a single outcome $omega in Omega$, we can define the information associated with
the probability distribution ${ Pr[omega] | omega in Omega }$. Let it be the expected
value of the self-information:

$
  H[Pr, Omega] :&= sum_(omega in Omega) Pr[omega] dot I(omega) \
                &= sum_(omega in Omega) Pr[omega] dot log_2 1 / Pr[omega].
$

This quantity is known as the Shannon entropy, it can be interpreted as the average amount
of information produced by the probability distribution.

As a random variable $X$ induces its own probability distribution ${ Pr[X = x] | x in supp(X) }$,
the entropy can be defined specifically for the random variable:

$
  H(X) :&= sum_(x in supp(X)) Pr[X = x] dot I(X = x) \
        &= sum_(x in supp(X)) Pr[X = x] dot log_2 1 / Pr[X = x],
$

which is equivalent to the distribution of the events ${ [X = x] | x in supp(X)}$.

=== Practical interpretation.
#note(
  title: [Fair dice],
)[
  induces the uniform distribution over the set of possible outcomes $p_1 = ... = p_6 = 1/6$.

  We need exactly $I = log_2 6 approx 2.58$ bits to encode each outcome. All outcomes are
  equally probable, so we need the same $I$ bits of information to encode any outcome on
  average.

  On the other hand, when we roll a fair dice, we receive $log_2 6$ bits of information from
  any outcome, and on average we receive the _entropy_ $H$ amount of information:
  $ H = sum_(i = 1)^6 1/6 dot log_2 1/6 = log_2 1/6 approx 2.58 "bits". $
]
