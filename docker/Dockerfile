FROM mambaorg/micromamba:latest

USER root
# Install system utilities and ARM64 Quarto CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates git \
    && wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v1.5.57/quarto-1.5.57-linux-arm64.deb \
    && dpkg -i quarto-1.5.57-linux-arm64.deb \
    && rm quarto-1.5.57-linux-arm64.deb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

# Install native ARM64 geospatial stack + JupyterLab + Jupytext
RUN micromamba install -y -n base -c conda-forge \
    python=3.11 \
    geopandas \
    libpysal \
    esda \
    mgwr \
    spreg \
    contextily \
    mapclassify \
    rasterio \
    xarray \
    rioxarray \
    seaborn \
    scikit-learn \
    jupyterlab \
    jupytext \
    ipykernel \
    && micromamba clean --all --yes

EXPOSE 8888
WORKDIR /home/mambauser/work

CMD ["micromamba", "run", "-n", "base", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
