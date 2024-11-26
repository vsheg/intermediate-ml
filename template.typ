#import "@preview/physica:0.9.3": *
#import "@preview/drafting:0.2.0": margin-note, set-page-properties
#import "defs.typ": *

// Default color palette
#import "@preview/typpuccino:0.1.0": latte
#let palette = latte
#let colors = (palette.teal, palette.pink, palette.flamingo, palette.mauve, palette.green,)

// Math annotation
#import "@preview/mannot:0.1.0": *
#show: mannot-init
#let hg1 = mark.with(color: colors.at(0))
#let hg2 = mark.with(color: colors.at(1))
#let hg3 = mark.with(color: colors.at(2))
#let hg4 = mark.with(color: colors.at(3))
#let hg5 = mark.with(color: colors.at(4))

#let note(title: [], content) = {
  set math.equation(numbering: none)
  set text(size: 0.8em, fill: luma(20%))

  title = strong(title)
  margin-note(side: right, stroke: none, title + content)
}

#let example(title: [], content) = {
  set math.equation(numbering: none)
  set text(fill: luma(20%), size: 0.9em)

  title = if title != [] { strong(title) }

  align(center, rect(width: 100%, fill: ghost-color, radius: 0.5em, {
    set align(left)
    title
    content
  }))
}

#let draft-pattern = {
  let element = text(size: 2em, fill: gray.opacify(-90%))[*DRAFT*]
  let pat = pattern(size: (90pt, 40pt), element)
  rotate(-25deg, rect(width: 150%, height: 150%, fill: pat))
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
    margin: (y: margin, left: margin, right: 1 / 4 * full-width),
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
    set text(size: font-size * 1.2)
    upper(it)
  }

  show heading.where(level: 2): it => {
    set text(size: font-size * 1.2)
    smallcaps(it)
  }

  show heading.where(level: 3): it => {
    set text(size: font-size)
    text(style: "italic", it.body) + [.]
  }

  doc
  pagebreak(weak: true)
}
