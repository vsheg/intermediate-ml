#import "@preview/physica:0.9.3": *
#import "@preview/drafting:0.2.0": margin-note, set-page-properties
#import "defs.typ": *

#let note(title: [], content) = {
  set text(size: 0.9em, fill: luma(20%))
  title = strong(title)
  margin-note(side: right, stroke: none, title + content)
}

#let example(title: [], content) = {
  set text(fill: luma(20%), size: 0.9em)
  title = if title != [] { strong(title) }
  align(center, rect(width: 90%, stroke: (thickness: 0.1pt, dash: "dashed"))[
    #set align(left)
    *Example*: #title #content
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

  let full-width = 195mm
  let full-heigh = 265mm
  let margin = 1cm

  set page(
    width: full-width,
    height: full-heigh,
    margin: (y: margin, left: margin, right: 1 / 3 * full-width),
    background: back,
  )

  // important to margin notes from drafting package
  set-page-properties()

  // font
  let font-size = 9pt
  set text(size: font-size, hyphenate: true, font: "New Computer Modern")

  // math equations
  set math.equation(numbering: "(1)")

  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      // Override equation references.
      link(el.location(), numbering(el.numbering, ..counter(eq).at(el.location())))
    } else {
      // Other references as usual.
      it
    }
  }

  // headings
  show heading: set text(fill: accent-color)

  show heading.where(level: 1): it => {
    set text(size: font-size * 1.1)
    set block(below: font-size * 1.5)
    smallcaps(it)
  }

  show heading.where(level: 2): it => {
    set text(size: font-size * 1)
    set block(above: font-size * 1.5, below: font-size)
    it
  }

  show heading.where(level: 3): it => {
    text(it.body) + [:]
  }

  doc
}

