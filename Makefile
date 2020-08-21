lab:
	docker run --rm -p 4000:4000 -p 8888:8888 -v ${PWD}:/home/jovyan/work darribas/gds_dev:4.1
labosx:
	docker run --rm -p 4000:4000 -p 8888:8888 -v ${PWD}:/home/jovyan/work:delegated darribas/gds_dev:4.1
sync: 
	jupytext --sync ./notebooks/*.ipynb
booksite: sync
	bash ./infrastructure/booksite/build.sh && \
	echo 'Swapping full site for _site'
	cd ./docs && \
	bundle exec jekyll build
	mv ./docs/_site ./tmp && \
	rm -r ./docs && \
	mv ./tmp ./docs && \
	cp ./CNAME ./docs/CNAME
bookserve:
	bash ./infrastructure/booksite/build.sh
	cd ./docs && \
	bundle exec jekyll serve --host 0.0.0.0
jb-gds:
	cp notebooks/*.md book/.
	cp notebooks/references.bib book/.

