#import "@preview/cetz:0.3.2": canvas, draw
#import "@preview/cetz-plot:0.1.1": plot

#let plot-canvas(..args) = canvas({
  import draw: *

  set-style(axes: (x: (padding: 1), y: (padding: 1)))

  plot.plot(..args)
})