---
title: Geographic Data Science
subtitle: A Second Edition
author: Sergio J. Rey, Daniel Arribas-Bel, and Levi John Wolf
date: today
format:
  html: 
    theme: litera
---

## Introduction

We'd like to make a 2nd edition of *Geographic Data Science with Python.* We'd like to drop the "with Python" in the title, and revise it to be a subtitle "with applications in Python." We'd aim to have a 2nd edition ready for fall of 2029.

In terms of general approach, we would tend to hide a bit more of the code generating incidental illustrations or tables, enabling a more traditional book-style presentation of material. We would also seek to ensure that rasters and point patterns are included as "first class citizens" in the analysis of data; every chapter would aim to provide examples across point, polygon, and raster data support. We would aim for a similar level of mathematical sophistication as the 1st edition: it's not a math book, and derivations would not be in the main text.
This fits more clearly into the portfolio of the *Statistical Science* series, since it will be revised to be more general/less about the use of *Python* as an integrated learning environment. This will also make the text more general, but retain its legibility to the largest computing language for data science. 

For us to do this, we'd like to know:
- What, exactly is our word limit/page limit? 
- How are the images/tables assessed, and is it as a single image or facets each count as one image? 
- Can we use Tufte margin notes? If we can hack it into Krantz, would it be allowed?
- Can we drop *In Python* from the title? If not, can we replace it with a subtitle, "with applications in Python?"

## New Table of Contents

★ means new or substantially revised chapter
**Section I: Representation**
1. Geographic Thinking for Data Scientists
2. Statistical Thinking for Geographers ★
3. Spatial Data
4. Spatial Queries and Data Processing ★
5. Spatial Graphs
6. Geovisualiation ★
**Section II: Exploration**
7. Global Autocorrelation
8. Local Autocorrelation
9. Multivariate Autocorrelation ★
10. Spatial Disparity (beyond Autocorrelation)
11. So you've found Autocorrelation, now what? ★
**Section III: Estimation**
12. Spatial Regression
13. Beyond Regression ★
14. Local Learning ★
15. Spatial Interaction Modelling ★
16. Clustering and Regionalization
17. Embeddings and Dimension Reduction ★
**Section IV: Refinement**
18. Model Fit and Spatial Model Assessment ★
19. Crossvalidating Geographical Models ★
20. Spatial Feature Engineering ★
21. Simulating Geographical Systems ★

## Annotated Table of Contents

This contains new chapters, as well as plans for revisions/removal of old chapters. 

### Section I: Representation

#### Geographic Thinking for Data Scientists

Will remain largely the same

#### Statistical Thinking for Geographers

This will outline a few different philosophical positions about how inference ought to be done in modelling. We'll cover the range from hardcore empiricists (only use randomization, no such thing as "process") through frequentists (what is observed is just one realisation of what could have been seen) and Bayesian (what is observed is intertwined with prior beliefs).

#### ~~Computational Notebooks (removed)~~

Relevant components of this will be put in the appendix, but generally the material will not be re-included. It is outdated and does not need serve a useful purpose any longer. 

#### Spatial Data

Will remain largely the same. 

#### Spatial Queries & Data Processing (new)

This chapter will show how to use geopandas indices to answer spatial queries. In addition, we'll talk a little bit about projection, and show how haversine tree vs. kdewill 
- use polars/duckdb as alternatives
- basics of interpolation

#### Spatial Graphs 

Largely the same, but will be updated to reflect ecosystem drift. 

#### Geovisualization (revised)

This extends original Choropleth Mapping chapter. It will segue into autocorrelation chapters with components and map complexity measures. It will cover static and responsive mapping using matplotlib, folium, and lonboard

### Section II:  Exploration

#### Global Spatial Autocorrelation (revised)

This will incorporate the elements from the point pattern chapter, and cover the idea of correlograms.

#### Local Spatial Autocorrelation (revised)

New material on point pattern local autocorrelation will get added here (local K function).

#### Multivariate Spatial Autocorrelation (new)

We'll cover multivariate autocorrelation/colocation statistics on lattices (multivariate join count) and point patterns (cross-K)

#### Spatial Disparity beyond Autocorrelation (segregation, inequality, polarization) (revised)

This will edit down the inequality chapter, and augment with other measures of spatial disparity, such as segregation (existing) and polarization (new)

#### ~~Point Pattern Analysis (removed)~~

Elements of this will be folded into the global spatial autocorrelation chapter and the simulation chapter. 

#### So you've found spatial autocorrelation: now what? (new)

Show how spatial structure in prediction error indicates degraded model performance, possible omitted variables, etc. Also maybe talk about leakage between folds within a cv setup? we don't have models yet by here, but maybe we can rely on people being vaguely familiar with prediction methods.

### Section III: Modelling

#### Spatial Regression (revised)

This will cover linear regression in a spatial context, including fitting standard linear regression, spatial fixed effects, SLX, and SAR. 

#### Beyond Regression (new)

This will cover nonlinear predictors common in data science, but apply them in a spatial context. This will include trees/forests and basic neural networks. 

#### Local Learning (new)

This will cover the basic idea of local predictive models: why do we fit them, and how can we make them useful? We'll probably cover `gwlearn`-type local ensembles, regimes regression, and [supervised spatial indices.](https://github.com/ljwolf/gwlearn/tree/spatial-index)

#### Spatial Interaction Models

This covers the basic of interaction models, and how they differ from predictions at points/locations or over fields. As a dyadic prediction, we talk about strategies for making good predictions, and cover the basic production/attraction/double constraint. 

#### Clustering and Regionalization 

This will be revised to reflect updates in the ecosystem, but will remain largely the same. 

#### Embeddings & Dimension Reduction (new)

This will cover dimension reduction methods that result in scores/embeddings, rather than classification decisions. We'll cover a little bit of PCA/nonlinear methods (ISOMAP or autoencoder), and then talk a little about embedding products like [Google's AlphaEarth](https://spatialthoughts.github.io/projects/satellite-embedding-project/)

### Section IV: Refinement

#### Model Fit & Assessment Plots/Metrics (new)

Will cover explicit aspatial and spatial measures for model fit, like R2, local R2, Moran statistic on residuals, (Geo)silhouettes, Join Count on Classifier output, v-measure region concordance measures, and spatially-aware rand index

*should we cover performance in terms of ram/timings?*

#### Spatial Feature Engineering (revised)

This will revise the chapter to focus on using the spatial query and graph methods from chapter 1 and constructing new features to integrate into predictive models. We'll explicitly cover distance-band/knn features, cluster features (both categorical and predict_proba/fuzzy), and centrality/eigenvectors.

#### Spatial CV (new)

This will cover the basic ideas of blocking and dispersion-based CV methods.

#### Simulation (new)

This will cover the basics of data generating processes, from basic hypothesis test simulation to SAR processes, and the use of local bootstraps/permutations for model assessment.

### We will need to build the following functionality from scratch:

In terms of code required to finish the second edition, we would need the following:

- local k in `pointpats`
- cross k in `pointpats`
- spatially aware rand index `esda`/`gwlearn.metrics`/`geovalidate`
- polarization measures `inequality`
- [supervised spatial indices](https://github.com/ljwolf/) `gwlearn`
- [spatial CV methods](https://ljwolf.org/geovalidate) `geovalidate`
