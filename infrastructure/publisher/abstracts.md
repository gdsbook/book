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

# Chapter 6: Global Spatial Autocorrelation
Spatial autocorrelation has to do with the degree to which the
similarity in values between observations in a dataset is related to
the similarity in locations of such observations. This is similar to
the traditional idea of correlation between two variables, which
informs us about how the values in one variable change as a function
of those in the other, albeit with some key differences discussed in
this chapter. In a similar fashion, spatial autocorrelation is also
related (but distinct) to temporal counterpart, temporal
autocorrelation, which relates the value of a variable at a given
point in time with those in previous periods. In contrast to these
other ideas of correlation, spatial autocorrelation relates the value
of the variable of interest in a given location, with values of the
same variable in other locations.

# Chapter 7: Local Spatial Autocorrelation
In this chapter, we introduce local measures of spatial
autocorrelation. Local measures of spatial autocorrelation focus on
the relationships between each observation and its surroundings,
rather than providing a single summary of these relationships across
the map. In this sense, they are not summary statistics but scores
that allow us to learn more about the spatial structure in our
data. The general intuition behind the metrics however is similar to
that of global ones. Some of them are even mathematically connected,
where the global version can be decomposed into a collection of local
ones. One such example are Local Indicators of Spatial Association
(LISAs) [Ans95], which we use to build the understanding of local
spatial autocorrelation, and on which we spend a good part of the
chapter. Once such concepts are firmed, we introduce a couple
alternative statistics that present complementary information or allow
us to obtain similar insights for categorical data.

# Chapter 8: Point Pattern Analysis
Points are spatial entities that can be understood in two
fundamentally different ways. On the one hand, points can be seen as
fixed objects in space, which is to say their location is taken as
given (exogenous). In this interpretation, the location of an observed
point is considered as secondary to the value observed at the
point. Think of this like measuring the number of cars traversing a
given road intersection; the location is fixed, and the data of
interest comes from the measurement taken at that location. The
analysis of this kind of point data is very similar to that of other
types of spatial data such as polygons and lines. On the other hand,
an observation occurring at a point can also be thought of as a site
of measurement from an underlying geographically-continuous
process. In this case, the measurement could theoretically take place
anywhere, but was only carried out or conducted in certain
locations. Think of this as measuring the length of birds’ wings: the
location at which birds are measured reflects the underlying
geographical process of bird movement and foraging, and the length of
the birds’ wings may reflect an underlying ecological process that
varies by bird. This kind of approach means that both the location and
the measurement matter.

# Chapter 9: Spatial Inequality Dynamics
This chapter uses economic inequality to illustrate how the study of
the evolution of social disparities can benefit from an explicitly
spatial treatment. Social and economic inequality is often at the top
of policy makers’ agendas. Its study has always drawn considerable
attention in academic circles. Much of the focus has been on
interpersonal income inequality, on differences between individuals
irrespective of the geographical area where they leave. Yet there is a
growing recognition that the question of inter-regional income
inequality requires further attention as the growing gaps between poor
and rich regions have been identified as key drivers of civil unrest
and political polarization in developing and developed countries.
After discussing the data we employ, we begin with an introduction to
classic methods for interpersonal income inequality analysis and how
they have been adopted to the question of regional inequalities. These
include a number of graphical tools alongside familiar indices of
inequality. As we discuss more fully, the use of these classical
methods in spatially referenced data, while useful in providing
insights on some of the aspects of spatial inequality, fails to fully
capture the nature of geographical disparities and their
dynamics. Thus, we next move to spatially explicit measures for
regional inequality analysis. The chapter closes with some recent
extensions of some classical measures to more fully examine the
spatial dimensions of regional inequality dynamics.

# Chapter 10: Clustering and Regionalization
In this chapter we consider clustering techniques and regionalization
methods. In the process, we will explore the socioeconomic
characteristics of neighborhoods in San Diego.  We begin with an
exploration of the multivariate nature of our dataset by suggesting
some ways to examine the statistical and spatial distribution before
carrying out any clustering. We then consider
geodemographic approaches to clustering—the application of
multivariate clustering to spatially referenced demographic data. Two
popular clustering algorithms are employed: k-means and Ward’s
hierarchical method. As we will see, mapping the spatial distribution
of the resulting clusters reveals interesting insights on the
socioeconomic structure of the San Diego metropolitan area. We also
see that in many cases, clusters are spatially fragmented.  This will
illustrate why connectivity might be important when building insight
about spatial data, since these clusters will not at all provide
intelligible regions. With this insight in mind, we will move on to
regionalization, exploring different approaches that incorporate
geographical constraints into the exploration of the social structure
of San Diego. Applying a regionalization approach is not always
required but it can provide additional insights into the spatial
structure of the multivariate statistical relationships that
traditional clustering is unable to articulate.

# Chapter 11: Spatial Regression
Regression (and prediction more generally) provides us a perfect case
to examine how spatial structure can help us understand and analyze
our data. In this chapter, we discuss how spatial structure can be
used to both validate and improve prediction algorithms, focusing on
linear regression specifically. In this chapter, we build space into
the traditional regression framework. We begin with a standard linear
regression model, devoid of any geographical reference. From there, we
formalise space and spatial relationships in three main ways: first,
encoding it in exogenous variables; second, through spatial
heterogeneity, or as systematic variation of outcomes across space;
third, as dependence, or through the effect associated to the
characteristics of spatial neighbors. Throughout, we focus on the
conceptual differences each approach entails rather than on the
technical details.

# Chapter 12 Spatial Feature Engineering
At its core, spatial feature engineering is the process of developing
additional information from raw data using geographic knowledge. This
distilling of information can occur between datasets, where geography
is used to link information in separate datasets together; or within
datasets, where geography can be used to augment the information
available for one sample by borrowing from nearby ones. This chapter
is structured following that distinction: for cases where geography
connects different datasets, we adopt the term “Map Matching”, often
used in industry; while we use the mirroring concept of “Map
Synthesis” describing the use of geographical structure to derive new
features from a given dataset. Technically speaking, some of the
methods we review are similar across these two cases, or even the
same; however they can be applied in the context of “matching” or
“synthesis”, and we consider those conceptually different, hence their
inclusion in both sections. Throughout the chapter, we use the AirBnB
nightly rental prices in San Diego, as well as auxiliary datasets such
as elevation or Census demographics.

