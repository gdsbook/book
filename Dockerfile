FROM darribas/gds_py:9.0

# Local docs
RUN rm -R work/
COPY ./README.md ${HOME}/README.md
COPY ./notebooks ${HOME}/notebooks
COPY ./figures ${HOME}/figures
COPY ./data ${HOME}/data
# Fix permissions
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
# Remove paired markdowns
RUN rm ${HOME}/notebooks/*.md
