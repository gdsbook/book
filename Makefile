.PHONY: all html pdf preview freeze hooks clean lab lablocal pack_site pack_content

# Default: render every configured format (HTML + PDF) from the _freeze cache.
all:
	quarto render

# Enable the versioned git hooks (pre-commit refreshes _freeze on .qmd changes).
hooks:
	git config core.hooksPath .githooks
	@echo "core.hooksPath set to .githooks"

# HTML site only -> docs/
html:
	quarto render --to html

# PDF book only -> docs/
pdf:
	quarto render --to pdf

# Live-reloading local preview.
preview:
	quarto preview

# Re-seed the execution cache by running every chapter. Requires the full
# `gds` conda env (see infrastructure/book_stack.yml) and the datasets in
# data/. Commit the resulting _freeze/ so CI can render without the geo stack.
freeze:
	quarto render

clean:
	rm -rf docs _book .quarto

# Docker dev environments (Jupyter/Lab) -----------------------------------
lab:
	docker run --rm \
               -p 4000:4000 \
               -p 8888:8888 \
               --user root \
               -e NB_UID=$UID \
               -e NB_GID=100 \
               -v ${PWD}:/home/jovyan/work \
               darribas/gds_dev:7.0

lablocal:
	docker run --rm \
               -p 4000:4000 \
               -p 8888:8888 \
               --user root \
               -e NB_UID=1000 \
               -e NB_GID=1000 \
               -v ${PWD}:/home/jovyan/work \
               darribas/gds_dev:7.0

# Distribution archives ----------------------------------------------------
pack_site: html
	rm -f gdsbook_site.zip
	cd docs && zip -r ../gdsbook_site.zip ./

pack_content:
	rm -f gdsbook_content.zip
	rm -rf gdsbook_content
	mkdir gdsbook_content
	mkdir gdsbook_content/notebooks
	cp notebooks/*.qmd gdsbook_content/notebooks
	cp notebooks/references.bib gdsbook_content/notebooks
	cp -r figures gdsbook_content/figures
	cp -r data gdsbook_content/data
	cp README.md gdsbook_content/README.md
	cd gdsbook_content && zip -r ../gdsbook_content.zip ./
	rm -r gdsbook_content
