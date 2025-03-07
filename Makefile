all: compile

compile:
	find . -path "./*" -name "*.typ" -exec typst compile {} --root . \;