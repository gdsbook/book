booksite:
	# Clean up earlier leftovers
	rm -rf tmp_book
	# Set up new jupyter-book project
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh \
		jupyter-book create host/tmp_book
	# Copy _config.ym
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh \
		cp host/infrastructure/booksite/_config.yml host/tmp_book/_config.yml
	# Copy notebooks/content
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh \
		cp host/notebooks/ host/tmp_book/content/notebooks/
	# Copy other contents of the book (intro, etc.)
	# TOC
	docker run --rm -v ${PWD}:/home/jovyan/host gdsbook start.sh \
		cp host/infrastructure/booksite/toc.yml host/tmp_book/_data/toc.yml
	# LICENSE
	# Code requirements
	# Bibliography
	# Build book
	# Move over to docs folder to be served online
	mv tmp_book docs
bookserve:
	# serve docs folder locally
