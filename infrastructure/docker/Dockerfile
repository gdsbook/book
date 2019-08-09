FROM darribas/gds_py:3.0

LABEL maintainer="Dani Arribas-Bel <d.arribas-bel@liverpool.ac.uk>"

#--- Jupyter-book ---#
RUN pip install -U --no-deps \
                    bs4 \
                    git+https://github.com/jupyter/jupyter-book \
                    nbclean \
                    nbgrader \
                    pytest \
                    ruamel.yaml
RUN jupyter-book create testbook --demo

#--- Jekyll ---#
# Taken inspiration from:
# https://github.com/jupyter/jupyter-book/blob/master/jupyter_book/Dockerfile

USER root
RUN apt-get update \
 && apt-get install -y ruby-full build-essential zlib1g-dev libc-dev
USER $NB_UID
# Install Ruby Gems to ~/gems
ENV GEM_HOME="/home/$NB_USER/gems"
ENV PATH="/home/$NB_USER/gems/bin:$PATH"
RUN cd /home/$NB_USER/testbook \
#&& mv Gemfile.lock original_Gemfile.lock \
 && gem install bundler -v '2.0.1' \
 && bundle install

# JupyText

RUN pip install jupytext --upgrade
USER root
RUN jupyter nbextension install --py jupytext \
 && jupyter nbextension enable --py jupytext \
 && jupyter labextension install jupyterlab-jupytext
USER $NB_UID
RUN echo "c.NotebookApp.contents_manager_class = "\
         "'jupytext.TextFileContentsManager'" \
 >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py

# Clean up
RUN rm -rf $GEM_HOME/cache \
 && rm -rf /usr/local/bundle/cache \
 && rm -r /home/$NB_USER/testbook \
 && conda clean --all

EXPOSE 4000

#--- Jupyter config ---#

RUN echo "c.NotebookApp.default_url = '/lab'" \
 >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
