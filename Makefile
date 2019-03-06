booksite:
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh sh -c "\
		sed -i -e 's/\r$//' host/infrastructure/booksite/build.sh && \
		./host/infrastructure/booksite/build.sh"
bookserve:
	# serve docs folder locally
	docker run --rm -ti -p 4000:4000 -v ${PWD}:/home/jovyan/host gdsbook \
		start.sh sh -c "cd host/docs && make serve"
