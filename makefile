TEX = main.tex  # latex sources
PDF = $(TEX:.tex=.pdf)  # produced PDFs
BBL = $(TEX:.tex=.bbl)  # processed bibliography
BIB = $(wildcard *.bib)  # bibliography
BST = $(wildcard *.bst)  # bibliographic style
STY = $(wildcard *.sty)  # style (package)

all: $(PDF)

figures:
	$(MAKE) -C figures

%.pdf: %.tex $(BIB) $(BST) $(STY) figures
	latexmk -pdf $<

%.bbl: %.tex $(BIB) $(BST) $(STY) figures
	latexmk -pdf $<

clean:
	git clean -Xdf $(addprefix -e !, $(PDF)) -e !arxiv.zip
	# latexmk -c

cleanall: clean
	# $(MAKE) cleanall -C figures
	rm -f $(PDF) arxiv.tar.gz

arxiv.tar.gz: $(TEX) $(BBL) $(STY)
	python arxiv_collector.py

.PHONY: all figures clean cleanall
