TYP_FILES := $(shell find . -path "./*/*" -name "*.typ" | sort)
PDF_ROOT := https://vsheg.github.io/intermediate-ml

all: compile

compile:
	@for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		typst compile $$typ --root .; \
		pdf_file=$${typ%.typ}.pdf; \
		if [ -f $$pdf_file ]; then \
			mv $$pdf_file $$dir/$$parent_dir.pdf; \
			echo "Compiled $$typ to $$dir/$$parent_dir.pdf"; \
		fi; \
		typst compile --format png --pages 1 $$typ --root .; \
		png_file=$${typ%.typ}.png; \
		if [ -f $$png_file ]; then \
			mv $$png_file $$dir/_cover.png; \
			echo "Generated cover $$dir/_cover.png"; \
		fi; \
	done

readme:
	@echo "# ML materials" > README.md
	@echo "" >> README.md
	@echo "> This is a work-in-progress draft of intermediate-level machine learning materials." >> README.md
	@echo "Thanks to LLMs for the high quality; any errors are mine." >> README.md
	@echo "" >> README.md
	@for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		dir=$$(echo $$dir | sed 's|^\./|$(PDF_ROOT)/|'); \
		echo "- [$$parent_dir]($$dir/$$parent_dir.pdf)" >> README.md; \
		echo "Added $$parent_dir/$$parent_dir.pdf to README.md"; \
	done

clean:
	@for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		pdf_file=$$dir/$$parent_dir.pdf; \
		png_file=$$dir/_cover.png; \
		if [ -f $$pdf_file ]; then \
			echo "Removing $$pdf_file"; \
			rm $$pdf_file; \
		fi; \
		if [ -f $$png_file ]; then \
			echo "Removing $$png_file"; \
			rm $$png_file; \
		fi; \
	done