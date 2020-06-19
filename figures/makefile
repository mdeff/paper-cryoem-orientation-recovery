EPS = $(wildcard *.eps)
SVG = $(wildcard *.svg)
ODG = $(wildcard *.odg)
PY = $(wildcard *.py)
TEX = $(wildcard *.tex)
PDF = $(EPS:.eps=.pdf) $(SVG:.svg=.pdf) $(ODG:.odg=.pdf) $(PY:.py=.pdf) $(TEX:.tex=.pdf)

all: $(PDF)

%.pdf: %.eps
	-epstopdf $<
	pdfcrop $@ $@

%.pdf: %.svg
	inkscape -D -o $@ $< 
	pdfcrop $@ $@

%.pdf: %.odg
	libreoffice --headless \
		"-env:UserInstallation=file:///tmp/libreoffice_conversion_${USER}" \
		--convert-to pdf $<
	pdfcrop $@ $@

%.pdf: %.py
	python $<
	pdfcrop $@ $@

%.pdf: %.tex
	latexmk -pdf $<
	pdfcrop $@ $@

cleanall:
	rm -f $(PDF)

.PHONY: all cleanall
