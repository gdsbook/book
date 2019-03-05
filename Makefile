booksite:
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh \
		./host/infrastructure/booksite/build.sh
bookserve:
	# serve docs folder locally
