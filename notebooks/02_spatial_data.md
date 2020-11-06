---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Geographic thinking for data scientists

## Introduction to geographic thinking

[Cova & Goodchild (2002)](https://doi.org/10.1080/13658810210137040) as a useful canonical reference. 

- all models are wrong, some are useful
- all maps are wrong, some are useful

spatial data is data with a locational component
sales pitch "what does geography give you?" argument

spatial data is ubiquitous and XXX

it gives you:
- location
- relation

Here, we'll refer to "models" of geography as a conceptual representation of geographic phenomena. We'll use "data structure" to refer to how these concepts are implemented in computer programs. 

## Conceptual Representations: Models

explain the ideas behind common geographical processes

The world is too complex to fully capture, a model as "dumbing down" reality that distills the essence of what you're interested in.
(theory versus measurement)

systems
    - object, agents that do things (movement modellers, demographers, )
    - field, fields of relationships (geostatisticians, remote sensors)
    - interaction, relationships as entities themselves (economists, mobility in cities)

scale/matched filter/tobler's scale invariance

explain object vs. field models of geographical processes more explicitly

This is about storing measurement

objects
    - have discrete boundary
    - have attributes measured about that are constant within their bounds
    
field model
    - doesn't necessarily have a discrete partitioning
    - assumes continuity where not observed

interactions
    - relations between one (or more) agents that themselves are of interest
    - more than the straight line between two objects

Scale (mis)match and how it can be mediated by the representation choice + information content (6m hexes from a single census tract contain the same amount of information...)


## Computational represenations: Data Structures

Tech as implementation of abstract concepts. But also as an entity with a "life of its own" (--> pick up later on barriers breaking up)

all maps are wrong, but some are useful

data cubes

postgis vector table

each cell is an "object" and fits into that paradigm, but it's not necessarily the same

GALs and other serializations of graphs

## Connecting the conceptual to the computational

### Conventional matching of raster -> field and vector -> object

"desktop view"

### This categorization is now breaking up (data is data)

hexagons/universalism 
- support as a data problem, good interpolators mean we don't need to worry about support. 
- common scales/formats for visualization (h3, rasterization to webmerc zooms, etc. )
    
examples of
- mapping between
- changes in "typical" representations of
  - population
  - ??? 


---
## Ideas
- all models are wrong, some are useful
- all maps are wrong, some are useful
- theory versus measurement
- letting the data speak for itself (ml? esda?)
- data mining from bad to good to ??
- process vs pattern
- matched filter (rodgerson)
    - https://en.wikipedia.org/wiki/Matched_filter

    - scale invariance (tobler)
- target-> data scientisits
- information content
    - effective sample size
    - different narrative from that for GIScientists
- target-> GIScientists
- spatial data is data with locational information encoded within
- how that is done 
- conceptual linking of representation/encoding to process
- there is a division between conceptualization and data models
- raster versus vector, field versus object, hybrid
- technical has driven things more so than conceptualization
- population versus sample (stochastic process)
    - Summerfield, M., 1983. "Populations, Samples and Statistical Inference in Geography." The Professional Geographer, 35, 143-149.
- integral (goodchild), spatial probability versus probability distribution
- how does this relate to a surface
- conceptual/represetnational model versus data model
- scale 
- are there issues we want to be opinionated about versus those we want to present different views on
