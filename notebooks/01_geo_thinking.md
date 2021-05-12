---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python tags=["remove-cell"]
import warnings
warnings.filterwarnings("ignore")
```

# Geographic thinking for data scientists

Data scientists have long worked with geographical data. Maps, particularly, are a favored kind of "infographic" in the age of responsive web cartography. While we discuss a bit about computational cartography in the chapter on choropleth mapping, it's useful to think a bit more deeply about how geography works in the context of data science. So, this chapter delves a bit into "geographic thinking," which represents the collected knowledge geographers have about why geographical information deserves special care and attention, especially when using geographic data in computation. 

## Introduction to geographic thinking

Geographical data has two very useful traits. First, geographic data is ubiquitous. Everything has a *location* in space-time, and this location can be used directly to make better predictions or inferences. In addition, this location allows you to understand the *relations* between observations. It is often *relations* that are useful in data science because they let us *contextualize* our analysis, building links within our existing data and beyond to other relevant data. As argued by the geographer Waldo Tobler, near things are likely to be more related than distant things, both in space and in time. Therefore, if learn from surroundings appropriately, we may be able to build better models.

Speaking of models, it is important to discuss how "location" and "relation" are represented. As the classic saying about statistical models,

> All models are wrong, but some are useful {cite}`Box_1976`

In this, the author (statistician George Box) suggests that models are simplified representations of reality. Despite the fact that these representations are *not* exactly correct in some sense, they *are* useful in understanding what is important about a statistical process. Reality is so complex that, if we simply could not capture all of the interactions and feedback loops that exist in our model. And, indeed, even if we could, the model would not be *useful*, since it would be so complex that it would be unlikely that any individual could understand it in totality. 

In a similar fashion, we paraphrase geographer Keith Ord in suggesting:

> All maps are wrong, but some are useful {cite}`Ord_2010` (p. 167)

Like a statistical model, a *map* is only a representation of the underlying geographical process. Despite the fact that these representations are *not* exactly correct in some sense, they *are* useful in understanding what is important about a geographical process. In this sense, we will use the term "data model" to refer to how we represent a geographical process conceptually. We'll use "data structure" in later sections to refer to how geographic data is represented in a computer. Below, we discuss a few common geographic data models and then move to presenting their links to typical geographic data structures. 

## Conceptual Representations: Models

The conceptual representation of a geographic process is often not straightforward to represent. For example, maps of population density generally require that we count the number of people that live within some specified "enumeration area," and then we divide by the total area. This represents the density of the area as a constant value over the entire enumeration unit. But, people are (in fact) discrete: we each exist only at one specific point in space and time. So, at a sufficiently fine-grained scale of measurement (in both time and space), the density is zero in most places and times! And, in a typical person's day, they may move from work to home, possibly moving through a few points in space-time in the process. For instance, most shopping malls have *zero* residents, but their population density is very high at a specific point in time, and draws its population from elsewhere.

This example of population density helps illustrate the classic data models in geographic information science. Geographic processes are represented using *objects*, *fields*, and *networks*. 
- *Objects* are discrete entities that occupy a specific position in space and time. 
- *Fields* are continuous surfaces that could, in theory, be measured at any location in space and time.
- *Networks* reflect a set of *connections* between objects or between positions in a field. 

In our population density example, an "enumeration unit" is an object, as is a person. The field representation would conceptualize density simply as a smooth, continuous surface that reflects the total number of persons at each possible location. The network representation would represent the inter-related system of densities that arise from people moving around. A helpful reference on the topic of common models in geographic information science is {cite}`Goodchild_2007`, who focus establishing a very general framework with which geographic processes can be described. 

The differences between these are important to understand, because they affect what kinds of *relations* are appropriate. For instance, the geographical processes with objects can be computed directly from the distance between them, or by considering or constructing a network that relates the objects. Geographical processes with networks must account for this *topology*, or structure of the connections between the "nodes" (i.e. the origins or destinations). We cannot assume that every node is connected, and these connections also cannot be directly determined from the nodes alone. In a field, measurement can occur *anywhere*, so models generally must account for the fact that the process *also* exists in the unobserved space between points of measurement. 

This structure, in turn, arises from how processes are conceptualized and what questions the analyst seeks to answer. And, since the measurement of a process is often beyond our control, it is useful to recognize that how a geographical process actually operates can be different from how it can actually be measured. In the subsequent sections, we discuss the common frames of measurement you may encounter, and the traditional linkages between data model and data structure that are found in classical geographic information systems.

## Computational representations: Data Structures


Above, we have discussed how data models are abstractions of reality that allow us to focus on the aspects we are interested in and measure them in a way that helps us answer the questions we care about. In this context, models are one piece of a broader process of simplification and operationalization that turns reality into representations suitable for computers and statistics. This is necessary for us to tap into their analytical capabilities. Data models clarify our thinking about which parts of the real world are relevant and which we can discard in each particular case; sometimes, they even guide how we could record or measure those aspects.

However, most of data and GI science is *empirical*, in that it consists of the analysis of measurements. To get to analysis, we require a further step of operationalization and simplification. The data models discussed in the previous section are usually still too abstract to help us in this context. So, we pair these data models with other constructs that tell us how quantities can be stored in computers and made amenable for analysis. This is the role that "data structures" play. 

Data structures are computer representations that organize different types of data in alignment with both the model they represent and the purpose such data fulfill. They form the middle layer that connects conceptual models to technology. At best, they accommodate the data model's principles as well as is technologically possible. In doing so, data structures enable data models to guide the computation. 

This relationship of influence however can also run in the opposite direction: once established, a technology or data structure can exert an important influence about how we see and model the world. This is not necessarily a bad thing. Embedding complex ideas in software helps widen the reach of a discipline. For example, desktop GIS software in the 90s and 00s made geographic information useful to a much wider audience. It made it so that geographic data *users* did not necessarily require specific training to work with geographic data or conduct spatial analysis. 

However, making conceptual decisions based on technological implementations can also be limiting. As a metaphor, we can think of technology as a pair of eyeglasses and data models as the "instructions" to build lenses: if all we use to look at the world is the one pair we already have, we miss all the other ways of looking at the world that arise if we built different lenses. In the 1990s, Mark Gahegan proposed the concept of ["disabling technology"](http://www.geocomputation.org/what.html) to express this notion {cite}`Gahegan_1999`.

So, "*what main data structures should the geographic data scientist care about, we hear you say?*" Of course, as with anything in technology, this is an evolving world. In fact, as we will see below in this chapter, much is changing rapidly and redefining how we translate conceptual models into computational constructs to hold data. However, there are a few key standards that have been around for a long time and proven their usefulness. In particular, we will cover three of them: geographic tables, surfaces (and cubes), and spatial graphs. We have decided to discuss these as each can be seen as the technological mirror of the concepts discussed in the previous section.

*Geographic tables* store information about discrete *objects*. Tables are two dimensional structures made up of rows and columns; each row represents an independent object, while each column stores an attribute of those objects. Geographic tables are standard tables where one column stores geographic information. The tabular structure fits well with the object model because it clearly partitions space into discrete entities, and assigns a geometry to each entity according to their spatial nature. More importantly, geographic tables allow to seamlessly combine geographic and non-geographic information. It is almost as if Geography is simply "one more attribute", when it comes to storage and computer representation. This is powerful because there is wide support in the world of databases for tabular formats. Geographic tables integrate spatial data into this non-spatial world and allow it to leverage much of its power. Technically speaking, geographic tables are widely supported in a variety of platforms. Popular examples include: PostGIS tables (as a geographic extension of PostgreSQL), R's `sf` data frames or, more relevant for this book, Python's `GeoDataFrame` objects, provided by `geopandas` (shown in the figure below). Although each of them has their own particularities, they all represent implementations of an object model.

![Figure 1: geographic tables](../figures/02_spatial_data_geodataframe.png)

*Surfaces* record empirical measurements of *fields*. Fields are a continuous representation of space. In principle, there is an infinite set of locations for which a field has a different value. In practice, fields are measured at a discrete set of locations.  This aim for continuity in space (and potentially time) is important because it feeds directly into how data are structured. In practice, fields are recorded and stored in uniform grids, or arrays whose dimension is closely linked to the geographic extent of the area they represent. Arrays are matrices made up of, at least, two dimensions. Unlike geographic tables, surface arrays use both rows and columns to signify location, and use cell values to store information about that location. A surface for a given phenomenon (e.g. air pollution) will be represented as an array where each row will be linked to different latitudes, and each column will represent longitudes. If we want to represent more than one phenomenon (e.g. air pollution *and* elevation), or the same phenomenon at different points int time we will need different arrays, possibly connected. These multi-dimensional arrays are sometimes called *data cubes*. An emerging standard in Python to represent surfaces and cubes is that provided by the `xarray` library, shown in the figure below.

![Figure 2: data cube](../figures/02_spatial_data_xarray.png)

*Spatial graphs* capture relations relationships between objects that are mediated through space. In a sense, they can be considered geographic *networks*, or a data structure to store topologies. There are several ways to define spatial relationships between features, and we explore many of them in Chapter XXX. The important thing to note for now is that, whichever spatial rule we follow, spatial graphs provide a way to encode them into a data structure that can support analytics. As we will see throughout the book, the range of techniques that rely on these topologies is pretty large, spanning from exploratory statistics of spatial autocorrelation (Ch. XXX), to regionalization (Ch. XXX) to spatial econometrics (Ch. XXX). Ironically, each of these fields, and others in computer science and mathematics, have come up with their own terminology to describe similar structures. Hence, when we talk or read about spatial weights matrices, adjacency matrices, geo-graphs, or spatial networks, we are thinking of very similar fundamental structures deployed in different contexts. Spatial graphs record information about how a given observation is spatially connected to others in the dataset. For this reason, they are an obvious complement to geographic tables, which store information about individual observations in isolation. Spatial graphs can also be derived from surfaces but here the situation is slightly different because, although surfaces record discrete measurements, they usually relate to a continuous phenomena. In theory, one could take these measurements at any point in space, so spatial graph of a surface should have an infinite number of observations. In practice however spatial graphs *are* sometimes used with grids because, as we will see in the following section, the connections and barriers between data models and structures are melting very quickly. Since many fields have theoretical constructs that resemble spatial graphs, there exist several slightly different data  structure that store them both in memory and on disk. In this book, we will focus on graph objects provided by the `networkX` library and, specially, on spatial weights matrices in PySAL which rely to a great extent on sparse matrix data structures. 

![Figure 3: networks](../figures/02_spatial_data_network.png)

The term spatial graph is sometimes interchangeably used with that of spatial network. This is of course a matter of naming conventions and, to the extent it is only that, it is not very important. However, the confusion can sometimes reflect a more profound misconception of what is being represented. Take the example of the streets in a city or, of the interconnected system of rivers in a catchment area. Both are usually referred to as networks, although in many cases what is being recorded is actually a collection of objects stored in a geographic table. To make the distinction clear, we need to think about what aspect of the street layout or the river system we want to record. If it is the exact shape, length and location of each segment or stream, this resembles much more a collection of independent lines or polygons that happen to "touch each other" at their ends. If what we are interested in is to understand how each segment or river *is related* to each other, which is connected with which and how this set of individual connections grow into a broader interconnected system, then a spatial graph is a more helpful structure to use. This dichotomy of the object versus the graph is only one example of a larger point about how we use data models and structures and how what the *right* one is does not depend on the phenomenon we are trying to capture only, but also on why we want to represent it, what the goal is.


## Connecting the conceptual to the computational

Now that the distinction between the *conceptual* data model and *computational* data structure is clear, we should explore the ways in which these are traditionally aligned. In presenting this traditional relationship between data model and data structure, we also seek to highlight the recent developments where this traditional mapping is breaking down. 

First, the main conceptual mapping of data model to data structure is inherited from advances made in computer graphics. This traditional view represents fields as rasters and objects as vector-based tables. In this mode of analysis, there is generally no space for *networks* as a first-class geographic data structure. They are instead computed on the fly from a given set of objects or fields. 

The separation of raster/image from vector/table and general omission of networks both stem from implementation decisions made in the one of the first commercially-successful geographic information systems, the Environmental Research Software Institute's ARC/INFO package. This was a command-line precursor to modern graphical information processing systems, such as the free and open source *QGIS* or the Environmental Research Software Institute (ESRI)'s *ArcGIS* packages. This association between field-and-raster, object-and-vector is sometimes called the "desktop view" of geographic information due to the dominance of these graphical GIS packages, although some argue that this  is fundamental to geographic representation {cite}`Goodchild_2007a` and cannot be transcended.

### This categorization is now breaking up (data is data)

Current trends in geographic data science suggest that this may not necessarily be the case, though. 
Indeed, contemporary geographic data science is moving beyond this in two very specific ways.

First, statistical learning methods are getting very good at efficiently translating data between different representations. 
The rise of machine learning, if nothing else, has generated extremely efficient "black box" prediction algorithms across a wide class of problems. 
If you can handle the fact that these algorithms generally are not *explainable* in their results, then these methods can generally provide significant improvements in prediction problems. 
Change of support problems, which arise when attempting to move data from one geographical support to another, are wholly oriented towards *accuracy*; the interpretation of a change of support algorithm is generally never of substantive interest. 
Therefore, machine learning has reduced the relevance of picking the "right" representation from the outset. 

Second, this means that there are an increasingly large number of attempts to find a "fundamental" underlying scale or representation that can be used for interchange between geographic data representations.
This has largely grown out of corporate data science environments, where transferring all geographic data to a single underlying representation has significant benefits for data storage, computation, and visualization. 
Projects such as Uber's "Hierarchical Hexagonal Geospatial Index," or "h3" for short, provide something like this, and many other internal proprietary systems serve the same purpose. 
In addition, projects like WorldPop are also shifting the traditional associations between "types" of data and the representations in which they are typically made available. 
Population data is generally presented in "object-based" representations, with the census enumeration units as "objects" and the counts as features of that object.
Now, though, population data is increasingly provided in world-wide rasters at varying resolutions, conceptualizing population *distribution* as a continuous field over which population *counts* (both modeled and known) can be presented. 

These are only two examples of the drive towards field-based representations in contemporary geographic data science, and will no doubt change as rapidly as the field itself. 
However, the incentives to create new representations will likely only intensify, as standard, shared architectures increasingly dominate the free and open source scientific software ecosystem. 


## Conclusion

This chapter has discussed the main conceptual *data models* for geographical process and their typical implementations in computational *data structures*. 
Objects, which are generally used to represent distinct "bounded" agents, are typically represented by "vector" data structures through a combination of points, lines, polygons. 
Fields, representations of the continuous smooth surfaces, are typically represented by "raster" data structures that look like images, with *pixels* recording values at each possible site and *bands* recording the kind of data recorded. 
Networks, which reflect relationships between objects, have typically been core to the data models of many geographical processes, but have not historically been represented easily in many of the data *structures* common in geographic information systems. 
This is changing, however, as networks become more central to our computational and conceptual understanding of geographical systems. 

By recognizing that the conceptual data model can be distinct from the computational data structure, we can move more easily between different data structures. 
Further, recent developments in computation and data storage are breaking down the traditional connections between data models and data structures. 
These have involved focusing much more intently on *change of support problems* that arise when converting between geographic data structures and attempts at finding a single "canonical" geographic data structure. 
Despite these trends, the main choice about the *data model* can be made independently of the *data structure*, so it is important to be very clear about the "entities" involved in your problem and how they are related.
Be attentive to what the entities you're interested in analyzing or predicting *are*, and be aware of when that may be at odds with *how they are measured*. 
While it is increasingly common to use hexagonal or square grids to represent data, be attentive to how this obscures the *actual* process you seek to measure or behavior you aim to model. 
Be prepared to answer: yes, we can grid this, but should we? The answer will depend on your goal. 

Overall, this chapter provides a grounding for the concepts and language used in the rest of the book to describe geographic processes, the datasets that describe them, and the models that analyze them. 
In some subsequent chapters, we will integrate information from many different data structures to study some processes; in other chapters, this will not be necessary.
Recognizing these conceptual differences will be important throughout the book. 
