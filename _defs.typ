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
#let ghost-color = rgb(50%, 50%, 50%, 50%)

#let All = sym.forall
#let Exi = sym.exists

#let row(..args) = $(#args.pos().join("  "))$

#import "@preview/quick-maths:0.2.1": shorthands

#let RSS = "RSS"
#let ESS = "ESS"
#let TSS = "TSS"
#let Pr = math.bb("P")
#let Ex = math.bb("E")
#let Var = math.bb("D")
#let Cov = math.op("Cov")
#let Cor = math.op("Cor")
#let supp = math.op("supp")
#let pdf = $cal(f)$
#let pmf = $cal(p)$
#let cdf = $cal(F)$
#let Exp = math.op("Exp")
#let logit = math.op("logit")
#let odd = math.op("odd")
#let fun = math.op($(dot)$)

#let scr(it) = text(features: ("ss01",), box($cal(it)$))

#let fn(args, operations) = {
  args.join(",")
  $|->$
  operations.join(",")
}

#let Ind(..sink) = {
  let args = sink.pos()
  if args.len() == 0 {
    return $[||]$
  } else {
    return $[|#args.at(0)|]$
  }
}

#let bra = sym.chevron.l
#let ket = sym.chevron.r

#let dmat(a, b, c) = $mat(#a, thin, thin; thin, #b, thin; thin, thin, #c)$
#let frame(body) = rect(stroke: 0.3pt, inset: 7pt, body)

// SHORTHANDS
#let replacements = (($+-$, $plus.minus$), ($:>$, $#h(0.5em) ⧴ #h(0.5em)$))

// Discrete probability plots share the same renderer as the other lessons.
#import "@preview/lilaq:0.6.0" as lq
#let discrete-plot(ys: (), x-label: $x$, y-label: $y$, width: 2cm, x-ticks: (), y-ticks: ()) = {
  let x = range(1, ys.len() + 1)
  lq.diagram(
    width: width,
    height: width,
    xlabel: x-label,
    ylabel: y-label,
    ylim: (0, 1.1),
    xlim: (0, ys.len() + 1),
    lq.stem(x, ys, color: accent-color),
  )
}
