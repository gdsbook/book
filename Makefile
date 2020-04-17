container:
	docker build -t gdsbook/stack:3.0 ./infrastructure/docker/
lab:
	docker run --rm -p 4000:4000 -p 8888:8888 -v ${PWD}:/home/jovyan/work gdsbook/stack:3.0
labosx:
	docker run --rm -p 4000:4000 -p 8888:8888 -v ${PWD}:/home/jovyan/work:delegated gdsbook/stack:3.0
sync: 
	jupytext --sync ./notebooks/*.ipynb
booksite: sync
	bash ./infrastructure/booksite/build.sh && \
	echo 'Swapping full site for _site' && \
	mv ./docs/_site ./tmp && \
	rm -r ./docs && \
	mv ./tmp ./docs && \
	cp ./CNAME ./docs/CNAME
bookserve:
	# serve docs folder locally
	bash ./infrastructure/booksite/build.sh
	cd ./docs && \
	bundle exec jekyll serve --host 0.0.0.0

