# Chapter 1: Geographic Thinking for Data Scientists
The chapter introduces the main conceptual data models for
geographical processes and their typical implementations in
computational data structures. Objects, which are generally used to
represent distinct “bounded” agents, are typically represented by
“vector” data structures through a combination of points, lines,
polygons. Fields, representations of the continuous smooth surfaces,
are typically represented by “raster” data structures that look like
images, with pixels recording values at each possible site and bands
indicating the kind of data recorded.  Networks, which reflect
relationships between objects, have typically been core to the data
models of many geographical processes, but have not historically been
represented easily in many of the data structures common in geographic
information systems.  This chapter provides a grounding for the
concepts and language used in the rest of the book.

# Chapter 2: Computational Tools for Geographic Data Science
This chapter provides an overview of the scientific and computational
context in which the book is framed. Many of the ideas discussed here
apply beyond geographic data science but, since they have been a
fundamental pillar in shaping the character of the book, they need to
be addressed. First, we will explore debates around “Open Science,”
its origins, and how the computational community is responding to
contemporary pressures to make science more open and accessible to
all. In particular, we will discuss three innovations in open science:
computational notebooks, open-source packages, and reproducible
science platforms. Having covered the conceptual background, we will
turn to a practical introduction of the key infrastructure this book
relies on: Jupyter Notebooks and JupyterLab, Python packages, and a
containerized platform to run the Python code in this book.


# Chapter 3: Spatial Data
This chapter grounds the ideas discussed in the previous two chapters
into a practical context. We consider how data structures, and the
data models they represent, are implemented in Python. We also cover
how to interact with these data structures. This will happen alongside
the code used to manipulate the data in a single computational
laboratory notebook. This, then, unites the two concepts of open
science and geographical thinking.  Further, we will spend most of the
chapter discussing how Python represents data once read from a file or
database, rather than focusing on specific file formats used to store
data. This is because the libraries we use will read any format into
one of a few canonical data structures that we discuss in
Chapter 1. We take this approach because these data structures are
what we interact with during our data analysis: they our interface
with the data. File formats, while useful, are secondary to this
purpose. Indeed, part of the benefit of Python (and other computing
languages) is abstraction: the complexities, particularities and
quirks associated with each file format are removed as Python
represents all data in a few standard ways, regardless of
provenance.

# Chapter 4: 
“Spatial weights” are one way to represent graphs in geographic data
science and spatial statistics. They are widely used constructs that
represent geographic relationships between the observational units in
a spatially referenced dataset. Implicitly, spatial weights connect
objects in a geographic table to one another using the spatial
relationships between them. By expressing the notion of geographical
proximity or connectedness, spatial weights are the main mechanism
through which the spatial relationships in geographical data is
brought to bear in the subsequent analysis.
In this chapter, we first consider different approaches to construct
spatial weights, distinguishing between those based on
contiguity/adjacency relations from weights obtained from distance
based relationships. We then discuss the case of hybrid weights which
combine one or more spatial operations in deriving the neighbor
relationships between observations. We illustrate all of these
concepts through the spatial weights class in pysal, which provides a
rich set of methods and characteristics for spatial weights.


# Chapter 5:  Choropleth Mapping
Choropleths are geographic maps that display statistical information
encoded in a color palette. Choropleth maps play a prominent role in
geographic data science as they allow us to display non-geographic
attributes or variables on a geographic map.  Choropleth mapping thus
revolves around: first, selecting a number of groups smaller than into
which all values in our dataset will be mapped to; second, identifying
a classification algorithm that executes such mapping, following some
principle that is aligned with our interest; and third, once we know
into how many groups we are going to reduce all values in our data,
which color is assigned to each group to ensure it encodes the
information we want to reflect. In broad terms, the classification
scheme defines the number of classes and rules for assignment; while a
good symbolization conveys information about the value differentiation
across classes.  In this chapter we first discuss the approaches used
to classify attribute values. This is followed by an overview of color
theory and the implications of different color schemes for effective
map design. We combine theory and practice by exploring how these
concepts are implemented in different Python packages, including
geopandas, and the Pysal federation of packages.

