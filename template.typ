#import "@preview/drafting:0.2.0": margin-note, set-page-properties
#import "defs.typ": accent-color

#let note(title: [], content) = {
  set text(size: 0.9em, fill: luma(20%))
  title = strong(title)
  margin-note(side: right, stroke: none, title + content)
}

#let example(content) = {
  set text(fill: luma(20%), size: 0.9em)
  align(center, rect(width: 90%, stroke: (thickness: 0.1pt, dash: "dashed"))[
    #set align(left)
    *Example*. #content
  ])
}

#let draft-pattern = {
  let element = text(size: 2em, fill: gray.opacify(-90%))[*DRAFT*]
  let pat = pattern(size: (90pt, 40pt), element)
  rotate(25deg, rect(width: 150%, height: 150%, fill: pat))
}

#let template(is-draft: true, doc) = {
  // page setup
  let back = none

  if is-draft {
    back = draft-pattern
  }

  let full-width = 600pt
  let margin = 10pt

  set page(
    width: full-width + 2 * margin,
    height: 24cm,
    margin: (y: margin, left: margin, right: 2 / 5 * full-width),
    background: back,
  )

  // important to margin notes from drafting package
  set-page-properties()

  // font
  set text(size: 9pt, hyphenate: true, font: "New Computer Modern")

  // math equations
  // set math.equation(numbering: "(1)")

  // headings
  show heading.where(level: 1): it => box(height: 2em, it)
  show heading.where(level: 2): it => box(height: 1em, smallcaps(it))
  show heading.where(level: 3): it => text(weight: "bold", style: "italic", it.body)

  // Styling
  // show strong: it => {
  //   set text(weight: "semibold", fill: luma(30%))
  //   highlight(fill: accent-color.lighten(90%), it)
  // }

  // header
  {
    set text(style: "italic")
    grid(
      columns: 2,
      column-gutter: 1fr,
    )[Vsevolod Shegolev][Updated: #datetime.today().display()]
  }

  doc
}

