TYP_FILES := $(shell find . -path "./*/*" -name "*.typ" | sort)

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
	done

readme:
	@echo "# ML materials\n" > README.md
	@echo "> This is a work-in-progress draft of intermediate-level machine learning materials." >> README.md
	@echo "Thanks to LLMs for the high quality; any errors are mine." >> README.md
	@echo "\n" >> README.md
	

	@for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		echo "- [$$parent_dir]($$dir/$$parent_dir.pdf)" >> README.md; \
		echo "Added $$parent_dir/$$parent_dir.pdf to README.md"; \
	done

clean:
	@for typ in $(TYP_FILES); do \
		dir=$$(dirname $$typ); \
		parent_dir=$$(basename $$dir); \
		pdf_file=$$dir/$$parent_dir.pdf; \
		if [ -f $$pdf_file ]; then \
			echo "Removing $$pdf_file"; \
			rm $$pdf_file; \
		fi; \
	done