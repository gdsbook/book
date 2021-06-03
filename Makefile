.PHONY: all html
    
lab:
	docker run --rm \
               -p 4000:4000 \
               -p 8888:8888 \
               --user root \
			   -e NB_UID=$UID \
			   -e NB_GID=100 \
               -v ${PWD}:/home/jovyan/work \
               darribas/gds_dev:6.1
lablocal:
	docker run --rm \
               -p 4000:4000 \
               -p 8888:8888 \
               --user root \
			   -e NB_UID=$UID \
			   -e NB_GID=100 \
               -v ${PWD}:/home/jovyan/work \
               darribas/gds_dev:6.1

    
    
sync: 
	jupytext --sync ./notebooks/*.ipynb
    
html: sync
	echo "Cleaning up existing tmp_book folder..."
	rm -rf docs
	rm -rf tmp_book
	echo "Populating build folder..."
	mkdir tmp_book
	mkdir tmp_book/notebooks
	cp notebooks/*.ipynb tmp_book/notebooks/
	cp notebooks/references.bib tmp_book/notebooks/
	cp -r data tmp_book/data
	cp -r figures tmp_book/figures
	cp -r infrastructure/website_content/* tmp_book/
	cp infrastructure/logo/ico_256x256.png tmp_book/logo.png
	cp infrastructure/logo/favicon.ico tmp_book/favicon.ico
	echo "Starting book build..."
	jupyter-book build tmp_book
	echo "Moving build..."
	mv tmp_book/_build/html docs
	echo "Cleaning up..."
	rm -r tmp_book
	touch docs/.nojekyll
    
reset_docs:
	rm -rf docs/*
	git checkout HEAD docs/*

# Run for example as: `make test_one nb=00_toc`
test_one:
	jupyter nbconvert --to notebook \
                      --execute \
                      --ExecutePreprocessor.timeout=600 \
                      notebooks/$(nb).ipynb
	rm notebooks/$(nb).nbconvert.ipynb

test:
	rm -rf tests
	mkdir tests
	jupyter nbconvert --to notebook \
                      --execute \
                      --output-dir=tests \
                      --ExecutePreprocessor.timeout=600 \
                      --ExecutePreprocessor.ipython_hist_file='' \
                      notebooks/*.ipynb 

	rm -rf tests
	echo "########\n\nAll blocks passed\n\n########"
    
pack_site: html
	rm -f gdsbook_site.zip
	cd docs && zip -r ../gdsbook_site.zip ./

pack_content:
	rm -f gdsbook_content.zip
	rm -rf gdsbook_content
	mkdir gdsbook_content
	mkdir gdsbook_content/notebooks
	cp notebooks/*.ipynb gdsbook_content/notebooks
	cp -r figures gdsbook_content/figures
	cp -r data gdsbook_content/data
	cp README.md gdsbook_content/README.md
	cd gdsbook_content && zip -r ../gdsbook_content.zip ./
	rm -r gdsbook_content
