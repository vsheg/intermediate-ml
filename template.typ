#import "@preview/drafting:0.2.0": margin-note, set-page-properties

#let note(title: [], content) = {
  set text(size: 0.9em, fill: luma(20%))

  if (title != []) {
    title = strong(title + [.])
  }

  margin-note(side: right, stroke: none, title + content)
}

#let draft-pattern = {
  let element = text(size: 2em, fill: red.opacify(-95%))[*DRAFT*]
  let pat = pattern(size: (90pt, 40pt), element)
  rotate(25deg, rect(width: 150%, height: 150%, fill: pat))
}

#let template(is-draft: true, doc) = {
  // page setup
  let back = none

  if is-draft {
    back = draft-pattern
  }

  set page(
    width: 18cm,
    height: 24cm,
    margin: (y: 0.5cm, left: 0.5cm, right: 8cm),
    background: back,
  )

  // important to margin notes from drafting package
  set-page-properties()

  // font
  set text(size: 9pt, hyphenate: true)

  // math equations
  // set math.equation(numbering: "(1)")

  // headings
  show heading.where(level: 1): it => box(height: 2em, it)
  show heading.where(level: 2): it => box(height: 1em, it)

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
