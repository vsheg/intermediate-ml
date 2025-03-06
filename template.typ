#import "@preview/physica:0.9.3": *
#import "@preview/drafting:0.2.2": (
  margin-note,
  set-page-properties,
  set-margin-note-defaults,
)
#import "@preview/quick-maths:0.2.0": shorthands
#import "defs.typ": *

//////////////
// TEMPLATE //
//////////////

#let draft-pattern = {
  let element = text(size: 2em, fill: gray.opacify(-90%))[*DRAFT*]
  let pattern = tiling(size: (90pt, 40pt), element)
  rotate(-25deg, rect(width: 150%, height: 150%, fill: pattern))
}

#let template(is-draft: true, doc) = {
  // page setup
  let back = none

  if is-draft {
    back = draft-pattern
  }

  let content-width = 180mm
  let content-heigh = 250mm
  let margin-size = 10mm
  let full-width = content-width + 2 * margin-size
  let full-heigh = content-heigh + 2 * margin-size

  set page(
    width: full-width,
    height: full-heigh,
    margin: (y: margin-size, left: margin-size, right: 0.33 * content-width),
    background: back,
  )

  // important to margin notes from drafting package
  set-page-properties()

  // font
  let font-size = 9pt
  set text(
    size: font-size,
    hyphenate: true,
    font: "New Computer Modern",
    costs: (hyphenation: 10%),
  )

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

  // text
  show "i.e.": set text(style: "italic")
  show "e.g.": set text(style: "italic")
  show "etc.": set text(style: "italic")
  show "cf.": set text(style: "italic")
  show "vs.": set text(style: "italic")
  set par(justify: true)

  // lists
  set list(marker: (
    text(font: "Menlo", size: 1.5em, baseline: -0.2em, "✴", fill: accent-color),
    text(size: 0.6em, baseline: +0.2em, "➤", fill: ghost-color),
  ))

  // headings
  show heading: set text(fill: accent-color)
  set heading(numbering: "1")
  let heading_counter = counter("heading_counter")

  show heading.where(level: 1): it => {
    heading_counter.step()
    if heading_counter.get().at(0) > 0 {
      pagebreak(weak: true)
    }
    set text(size: font-size * 1.1)
    set block(below: 1em)
    it
  }

  show heading.where(level: 2): it => {
    set text(size: font-size * 0.9)
    text(it.body) + [.]
  }

  show ref: it => {
    let cite = ref
    let el = it.element
    if el == none {
      footnote(it)
    } else {
      it
    }
  }

  // quick math shorthands
  show: shorthands.with(..replacements)

  // tables
  set table(stroke: none)
  set table(align: center + horizon)
  set table(
    fill: (_, y) => if (y == 0) { accent-color.transparentize(80%) } else { if calc.even(y) { ghost-color.transparentize(80%) } },
  )

  // begin document
  doc
  bibliography(title: [References], style: "ieee", "assets/citations.bib")
}

///////////////////////
// STYLING FUNCTIONS //
///////////////////////

// Math annotation
#set math.cancel(stroke: black.transparentize(50%))

#let margin(title: [], content) = {
  set math.equation(numbering: none)
  set text(size: 0.8em, fill: luma(20%))

  if title != [] { title = strong(title) + [. ] }

  v(0pt, weak: true)
  margin-note(side: right, stroke: none, title + content)
}

#let note(title: [], content) = {
  set math.equation(numbering: none)
  set text(fill: luma(20%), size: 0.9em)

  title = if title != [] { strong(title) + [.] }

  align(center, rect(width: 100%, fill: ghost-color.transparentize(90%), radius: 0.5em, {
    set align(left)
    set text(size: 0.9em)
    title
    content
  }))
}

#let divider = {
  line(
    length: 100%,
    stroke: (paint: ghost-color, thickness: 0.5pt, dash: "loosely-dashed"),
  )
}

#let comment(content) = text(fill: ghost-color, size: 0.8em, $   &$ + content)

#let focus(content) = {
  text(fill: accent-color, content)
}

#let compare(..contents) = {
  let n = contents.pos().len()
  grid(columns: n, column-gutter: 1em, ..contents)
}

// Default color palette
#import "@preview/typpuccino:0.1.0": latte
#let palette = latte
#let colors = (palette.teal, palette.pink, palette.flamingo, palette.mauve, palette.green,)