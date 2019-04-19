container:
	docker build -t gdsbook ./infrastructure/docker/
lab:
	docker run --rm -p 8888:8888 -v ${PWD}:/home/jovyan/host gdsbook
sync: 
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh sh -c "\
		jupytext --sync ./host/notebooks/*.ipynb"
booksite: sync
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh sh -c "\
		sed -i -e 's/\r$$//' host/infrastructure/booksite/build.sh && \
		bash ./host/infrastructure/booksite/build.sh"
bookserve:
	# serve docs folder locally
	docker run --rm -ti -p 4000:4000 -v ${PWD}:/home/jovyan/host gdsbook \
		start.sh sh -c "cd host/docs && bundle exec jekyll serve --host 0.0.0.0"
