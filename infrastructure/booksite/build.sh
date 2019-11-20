#!/bin/bash

# Clean up earlier leftovers
echo "Cleaning up existing tmp_book folder..."
rm -rf tmp_book
# Set up new jupyter-book project
echo "Starting new jupyter-book project..."
jupyter-book create tmp_book 
echo "Copying files over book folder..."
# Copy _config.ym
cp ./infrastructure/booksite/_config.yml ./tmp_book/_config.yml
# Copy notebooks/content
mkdir ./tmp_book/content/notebooks/
cp -r ./notebooks/*.ipynb ./tmp_book/content/notebooks/
cp ./notebooks/00_toc.md ./tmp_book/content/notebooks/00_toc.md
# Copy other contents of the book (intro, etc.)
cp ./infrastructure/booksite/intro.md ./tmp_book/content/intro.md
cp ./infrastructure/booksite/intro_part_*.md ./tmp_book/content/
# TOC
cp ./infrastructure/booksite/toc.yml ./tmp_book/_data/toc.yml
#---------------------------
# Fill in
#---------------------------
# LICENSE
cp ./infrastructure/booksite/LICENSE ./tmp_book/LICENSE
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
cd tmp_book && make build
# Move over to docs folder to be served online
echo "Moving build over to docs folder..."
cd /home/jovyan/work
rm -r ./docs
mv tmp_book/ docs
rm -r tmp_book
echo "All done!"

