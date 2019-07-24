FROM darribas/gds_py:2.0

# Local docs
RUN rm -R work/
COPY ./README.md ${HOME}/README.md
RUN mkdir ${HOME}/content
COPY ./notebooks ${HOME}/content/notebooks
COPY ./figures ${HOME}/content/figures
COPY ./data ${HOME}/content/data
# Fix permissions
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
