#import "@preview/tufted:0.1.1"
#import "@preview/physica:0.9.8": *
#import "@preview/quick-maths:0.2.1": shorthands
#import "@preview/lilaq:0.6.0" as lq
#import "_defs.typ": *

// Paths are relative to the generated page, including on project-hosted sites.
#let template(title: "Intermediate ML", home: "../index.html", assets: "../assets", is-draft: true, doc) = {
  show: tufted.tufted-web.with(
    title: title,
    header-links: ((home): "Intermediate ML"),
    css: (
      "https://cdnjs.cloudflare.com/ajax/libs/tufte-css/1.8.0/tufte.min.css",
      assets + "/tufted.css",
      assets + "/custom.css",
    ),
  )
  show: shorthands.with(..replacements)
  // Render diagram/layout primitives as SVG while keeping prose as HTML.
  show grid: it => if target() == "html" {
    html.div(class: "course-grid", it.children.map(child => html.div(child.body)).join())
  } else { it }
  show colbreak: it => html.div(class: "column-break")
  show columns: it => html.div(class: "course-columns", it.body)
  show rect: html.frame
  show math.cancel: set math.cancel(stroke: black.transparentize(50%))
  set heading(numbering: "1")
  doc
  bibliography(title: [References], style: "apa", "assets/citations.bib")
}

#let note-style(content) = {
  set math.equation(numbering: none)
  content
}

#let margin(title: none, ..parts) = {
  let items = parts.pos()
  let body = if items.len() == 1 { items.first() } else {
    title = items.first()
    items.slice(1).join(linebreak())
  }
  tufted.margin-note(note-style({
    if title != none { strong(title) + [: ] }
    body
  }))
}

#let uplift(content) = html.div(class: "note", content)
#let note(cols: 1, title: none, content) = uplift(note-style({
  if title != none { html.strong(title + [: ]) }
  content
}))
#let example(cols: 1, content) = note(cols: cols, title: [Example], content)
#let divider = html.hr()
#let focus(content) = text(fill: accent-color, content)
#let comment(content) = text(fill: ghost-color, size: 0.8em, $&$ + content)

// Default color palette
#import "@preview/typpuccino:0.1.0": latte
#let palette = latte
#let colors = (
  palette.teal,
  palette.pink,
  palette.flamingo,
  palette.mauve,
  palette.green,
)

// Plot internals use paged layout; export only the plot itself as SVG.
#let diagram(..args) = html.frame(lq.diagram(..args))
#let multi-figure(caption: none, label: none, ..args) = {
  let body = html.div(class: "plot-group", args.pos().join())
  let result = figure(body, caption: caption)
  if label != none { [#result #label] } else { result }
}
