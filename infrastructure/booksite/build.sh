#!/bin/bash

# Clean up earlier leftovers
echo "Cleaning up existing tmp_book folder..."
rm -rf host/tmp_book
# Set up new jupyter-book project
echo "Starting new jupyter-book project..."
jupyter-book create host/tmp_book 
echo "Copying files over book folder..."
# Copy _config.ym
cp host/infrastructure/booksite/_config.yml host/tmp_book/_config.yml
# Copy notebooks/content
cp -r host/notebooks/ host/tmp_book/content/notebooks/
# Copy other contents of the book (intro, etc.)
cp host/infrastructure/booksite/intro.md host/tmp_book/content/intro.md
# TOC
cp host/infrastructure/booksite/toc.yml host/tmp_book/_data/toc.yml
#---------------------------
# Fill in
#---------------------------
# LICENSE
# Code requirements
# Bibliography
# Build book
#---------------------------
echo "Building book..."
jupyter-book build host/tmp_book
echo "Building site HTML..."
cd host/tmp_book && make site
# Move over to docs folder to be served online
echo "Moving build over to docs folder..."
cd /home/jovyan
rm -r host/docs
mv host/tmp_book host/docs
echo "All done!"

