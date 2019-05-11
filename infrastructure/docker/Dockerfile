FROM darribas/gds_py:2.0

LABEL maintainer="Dani Arribas-Bel <d.arribas-bel@liverpool.ac.uk>"

#--- Jekyll ---#

USER root

RUN apt-get update \
 && apt-get install -y ruby-full build-essential zlib1g-dev 

# Install Ruby Gems to ~/gems
ENV GEM_HOME="/home/$NB_USER/gems"
ENV PATH="/home/$NB_USER/gems/bin:$PATH"

USER $NB_UID

RUN gem install jekyll bundler jekyll-scholar

# Clean up
RUN rm -rf $GEM_HOME/cache
RUN rm -rf /usr/local/bundle/cache

#--- Jupyter-book ---#
RUN pip install -U --no-deps \
                    bs4 \
                    git+https://github.com/jupyter/jupyter-book \
                    nbclean \
                    nbgrader \
                    pytest \
                    ruamel.yaml
RUN jupyter-book create testbook --demo
RUN cd /home/$NB_USER/testbook \
 && gem install bundler -v 1.17.2 \
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
 && conda clean -tipsy

EXPOSE 4000

#--- Jupyter config ---#

RUN echo "c.NotebookApp.default_url = '/lab'" \
 >> /home/$NB_USER/.jupyter/jupyter_notebook_config.py

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID
