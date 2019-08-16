container:
	docker build -t gdsbook/stack:3.0 ./infrastructure/docker/
lab:
	docker run --rm -p 8888:8888 -v ${PWD}:/home/jovyan/host gdsbook/stack:3.0
sync: 
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook/stack:3.0 start.sh sh -c "\
		jupytext --sync ./host/notebooks/*.ipynb"
	#docker run --rm --user root -e NB_UID=1001 -e NB_GID=100 -v ${PWD}:/home/jovyan/host gdsbook start.sh sh -c "\
	#	jupytext --sync ./host/notebooks/*.ipynb"
booksite: sync
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook/stack:3.0 start.sh sh -c "\
		bash ./host/infrastructure/booksite/build.sh && \
		echo 'Swapping full site for _site' && \
		mv host/docs/_site host/tmp && \
		rm -r host/docs && \
		mv host/tmp host/docs"
	#docker run --rm --user root -e NB_UID=1001 -e NB_GID=100 -v ${PWD}:/home/jovyan/host gdsbook/stack:3.0 start.sh sh -c "\
		#sed -i -e 's/\r$$//' host/infrastructure/booksite/build.sh && \
		#bash ./host/infrastructure/booksite/build.sh"
bookserve:
	# serve docs folder locally
	docker run --rm -ti -p 4000:4000 -v ${PWD}:/home/jovyan/host gdsbook/stack:3.0 \
		start.sh sh -c "\
		bash ./host/infrastructure/booksite/build.sh && \
		cd host/docs && \
		bundle exec jekyll serve --host 0.0.0.0"
	#docker run --rm -ti --user root -e NB_UID=1001 -e NB_GID=100 -p 4000:4000 -v ${PWD}:/home/jovyan/host gdsbook \
		#start.sh sh -c "cd host/docs && bundle exec jekyll serve --host 0.0.0.0"
