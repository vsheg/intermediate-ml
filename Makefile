all: compile

compile:
	find . -name "*.typ" -exec typst compile {} --root . \;