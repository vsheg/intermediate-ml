all: render

render:
	find . -path "./*/*" -name "*.typ" -exec typst compile {} --root . \;
	find . -path "./*/*" -name "*.pdf" | xargs git add
	git commit -m "Update PDFs"
	git push