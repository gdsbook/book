execute: notebooks/$(chapter).ipynb
	jupyter nbconvert --execute --to html notebooks/$(chapter).ipynb --output-dir docs/
	cd notebooks; pwd; jupyter nbconvert --execute --to pdf $(chapter).ipynb --output-dir ../pdf/

show: notebooks/$(chapter).ipynb
	jupyter nbconvert --to html notebooks/$(chapter).ipynb --output-dir docs/
	cd notebooks; pwd; jupyter nbconvert --to pdf $(chapter).ipynb --output-dir ../pdf/

executebook: # execute the notebooks and build the full book 
	jupyter nbconvert --execute --to html --ExecutePreprocessor.timeout=-1 --allow-errors notebooks/*.ipynb --output-dir docs/
	# since it's broken when building from git root
	cd notebooks; pwd; jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --allow-errors --to pdf *.ipynb --output-dir ../pdf/;
	pdfunite pdf/*.pdf book.pdf

showbook: # only build the book, don't execute the notebooks
	jupyter nbconvert --to html notebooks/*.ipynb --output-dir docs/
	# since it's broken when building from git root
	cd notebooks; pwd; jupyter nbconvert --to pdf *.ipynb --output-dir ../pdf/;
	pdfunite pdf/*.pdf book.pdf

test_stack:
	# Test data stack
	curl https://raw.githubusercontent.com/darribas/gds_env/master/gds_py/check_py_stack.ipynb -o check_py_stack.ipynb
	docker run --rm -ti -v `pwd`:/home/jovyan/host gdsbook start.sh jupyter nbconvert --execute /home/jovyan/host/check_py_stack.ipynb
	rm check_py_stack.ipynb check_py_stack.html
	# Test jupyter-book
	docker run --rm -ti gdsbook start.sh jupyter-book create mybookname --demo
	# Test Jekyll
	docker run --rm -ti gdsbook jekyll new myblog
	# Success
	echo "\nThings seem to work out!\n"

launch_container:
	docker run --rm -ti -v `pwd`:/home/jovyan/host -p 8888:8888 gdsbook
