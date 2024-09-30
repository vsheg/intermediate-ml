#let const = $"const"$

#let hr = {
  line()
}

#let xb = math.bold("x")
#let tb = math.bold(math.theta)
#let ub = math.underbrace
#let Tr = math.sans("T ")
#let accent-color = eastern

#let hg(content) = text(fill: accent-color, $#content$)