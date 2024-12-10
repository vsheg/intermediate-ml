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

#import "@preview/quick-maths:0.2.0": shorthands

#let hg(content) = text(fill: accent-color, $#content$)

#let RSS = "RSS"
#let Pr = math.bb("P")
#let Ex = math.bb("E")
#let Var = math.op("Var")
#let Cov = math.op("Cov")
#let Cor = math.op("Cor")
#let supp = math.op("supp")
#let pdf = math.op("pdf")
#let cdf = math.op("cdf")
#let Exp = math.op("Exp")
#let logit = math.op("logit")

#let Ind(..sink) = {
  let args = sink.pos()
  if args.len() == 0 {
    return $[||]$
  } else {
    return $[|#args.at(0)|]$
  }
}

#let bra = sym.angle.l
#let ket = sym.angle.r

#let dmat(a, b, c) = $mat(#a, thin, thin;thin, #b, thin;thin, thin, #c)$
#let frame(body) = rect(stroke: 0.3pt, inset: 7pt, body)

// SHORTHANDS
#let replacements = (($+-$, $plus.minus$), ($:>$, $#h(0.5em) ⧴ #h(0.5em) $),)

// PLOTS
#import "@preview/cetz:0.3.1"
#import "@preview/cetz-plot:0.1.0"
#import "@preview/subpar:0.1.1"

#let discrete-plot(ys: (), x-label: $x$, y-label: $y$, width: 2cm, x-ticks: (), y-ticks: ()) = {
  let n = ys.len()

  cetz.canvas(length: width, {
    import cetz.draw: *
    import cetz-plot: *

    let x = 0
    plot.plot(
      size: (1, 1),
      y-max: 1,
      x-max: 1 + 1 / n,
      x-tick-step: none,
      y-tick-step: 1,
      x-label: x-label,
      y-label: y-label,
      {
        for i in range(n) {
          x += 1 / n
          plot.annotate(line((x, 0), (x, ys.at(i)), stroke: accent-color))
          plot.annotate(content(
            (x + 0.2, ys.at(i) + 0.1),
            angle: 60deg,
            anchor: "mid-east",
            x-ticks.at(i, default: []),
          ))
          plot.add(((x, ys.at(i)),), mark: "o", mark-size: 0.1, hypograph: true)
        }
      },
    )
  })
}
