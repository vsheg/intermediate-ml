TYPST ?= typst
TYP_FILES := $(shell find content -name '*.typ' -not -path '*/_*' | sort)
HTML_FILES := $(patsubst content/%.typ,html/%.html,$(TYP_FILES))
SHARED_FILES := _template.typ _defs.typ assets/citations.bib
DATA_FILES := $(shell find content -name '*.csv')

.DEFAULT_GOAL := html

html: $(HTML_FILES) assets

html/%.html: content/%.typ $(SHARED_FILES) $(DATA_FILES)
	@mkdir -p "$(@D)"
	$(TYPST) compile --root . --features html --format html "$<" "$@"

assets:
	@mkdir -p html/assets
	cp assets/tufted.css assets/custom.css html/assets/

clean:
	rm -rf html

.PHONY: html assets clean
.DELETE_ON_ERROR:
