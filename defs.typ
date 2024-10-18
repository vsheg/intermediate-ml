#let const = $"const"$

#let hr = {
  line()
}

#let xb = math.bold("x")
#let tb = math.bold(math.theta)
#let ub = math.underbrace
#let ob = math.overbrace
#let Tr = math.sans("T ")
#let accent-color = eastern

#let hg(content) = text(fill: accent-color, $#content$)

#let RSS = "RSS"
#let Pr = math.bb("P")
#let Ex = math.bb("E")
#let Var = "Var"
#let Cov = "Cov"
#let Cor = "Cor"
#let supp = "supp"

#let bra = sym.angle.l
#let ket = sym.angle.r

#let dmat(a, b, c) = $mat(#a, thin, thin;thin, #b, thin;thin, thin, #c)$