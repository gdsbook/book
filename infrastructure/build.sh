#!/bin/bash

# Clean up earlier leftovers
echo "Cleaning up existing tmp_book folder..."
rm -rf tmp_book
# Set up new jupyter-book project
echo "Starting new jupyter-book project..."
jupyter-book create tmp_book \
                    --license ./infrastructure/booksite/LICENSE \
                    --toc ./infrastructure/booksite/toc.yml \
                    --config ./infrastructure/booksite/_config.yml
echo "Copying extra files over book folder..."
# Copy TOC
cp ./infrastructure/booksite/toc.yml ./tmp_book/_data/toc.yml
# Copy notebooks/content
mkdir ./tmp_book/content/notebooks/
cp -r ./notebooks/*.ipynb ./tmp_book/content/notebooks/
cp ./notebooks/00_toc.md ./tmp_book/content/notebooks/00_toc.md
# Copy data/content
mkdir ./tmp_book/content/data/
cp -r ./data ./tmp_book/content/
# Copy other contents of the book (intro, etc.)
cp ./infrastructure/booksite/intro.md ./tmp_book/content/intro.md
cp ./infrastructure/booksite/intro_part_*.md ./tmp_book/content/
#---------------------------
# Fill in
#---------------------------
# Logo
cp ./infrastructure/booksite/logo/ico_256x256.png ./tmp_book/content/images/logo/logo.png
cp ./infrastructure/booksite/logo/favicon.ico ./tmp_book/content/images/logo/favicon.ico
# Code requirements
# Bibliography
#---------------------------
# Build book
echo "Building book..."
jupyter-book build tmp_book
echo "Building site HTML..."
cd tmp_book && make build && cd ..
# Move over to docs folder to be served online
echo "Moving build over to docs folder..."
rm -rf ./docs
mv tmp_book/ docs
rm -rf tmp_book
echo "All done!"

