---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: analysis
    language: python
    name: analysis
---

# Spatial Data

```python
import osmnx
import geopandas
import rioxarray
import xarray
import contextily as cx
```

## Fundamentals


Geographic data is generally stored in a few distinct formats. As we discussed in the last chapter, these *data structures* are typically associated with *data models* that represent geographic processes. In what follows, we will get a little more familiar with the data structures that are involved in geographic data science. For each format, we will present the main concepts behind the data structure as well as the main patterns for working with data in that format. 


### Geographic Tables

To start, one common representation of geographic objects is the *geographic table* that represents a single geographic object as a row of a table. Each column in this table records information about the object. Typically, there is a special column in this table that records the *geometry* of the object. Computer systems that use this data structure are intended to add geography into a *relational database*, such as PostGreSQL (through its PostGIS extension) or sqlite (through its spatialite extension). Beyond this, however, many data science languages (such as R, Julia, and Python), have packages that adopt this data structure as well (such as sf, ArchGDAL, and geopandas), and it is rapidly becoming the main data structure for object-based geographic data. 

Before proceeding, though, it helps to mention a quick clarification on terminology. Throughout this book, regardless of the data structure used, we will refer to a measurement about an observation as a *feature*. This is consistent with other work in data science and machine learning. Then, one set of measurements is a *sample*. For tables, this means a feature is a column and a sample is a row. Historically, though, geographic information scientists have used the word "feature" to mean an individual observation, since a "feature" in cartography is an entity on a map. Thus, being clear about this terminology is important: for this book, a *feature* is one measured trait pertaining to an observation (a column), and a *sample* is one set of measurements (a row). 

To understand the structure of these datasets, it will help to read in the `countries_clean.gpkg` dataset included in this book that describes countries in the world. To read in this data, we can use the `read_file()` method in `geopandas`:

```python
gt_polygons = geopandas.read_file("../data/countries/countries_clean.gpkg")
```

```python
gt_polygons.head()
```

Each row of this table is a single country. This table shows only two features: the administrative name of the country and the geometry of the country's boundary. The name of the country is encoded in the `ADMIN` column using the Python `str` type, which is used to store text-based data. The geometry of the country's boundary is stored in the `geometry` column, and is encoded using a special class in Python that is used to represent geometric objects. 

```python
type(gt_polygons.geometry[0])
```

In `geopandas` (as well as other packages representing geographic data), sometimes the `geometry` column has special traits. For example, when we plot the dataframe, the `geometry` column is used as the main shape to use in the plot. 

```python
gt_polygons.plot()
```

Changing geometries must be done carefully: since the `geometry` column is special, there are special functions to adjust the geometry. For example, if we were to represent each country using its *centroid*, a point in the middle of the shape, then we must take care to set the geometry again. For example, when we compute the centroid, we can use the `gt_polygons.geometry.centroid` property and add a new column containing the centroid:

```python
gt_polygons['centroid'] = gt_polygons.geometry.centroid
```

```python
gt_polygons.head()
```

We can switch to the centroid column using the `set_geometry()` method. This can be useful when you want to work with two different geometric representations of the same underlying sample. For example, we can plot the centroid and the boundary of each country by switching the geometry column with `set_geometry`: 

```python
ax = gt_polygons.set_geometry('centroid').plot('ADMIN', markersize=5)
gt_polygons.plot('ADMIN', ax=ax, facecolor='none', 
                 edgecolor='k', linewidth=.2)
```

Thus, as should now be clear, nearly any kind of geometric object can be represented in one (or more) geometry columns. Thinking about the number of different kinds of shapes or geometries one could draw quickly boggles the mind. Fortunately the Open Geospatial Consortium (OGC) has defined a set of "abstract" types that can be used to define any kind of geometry. This specification, codified in ISO 19125-1, the "simple features" specification, specifies the formal relationships between these types: a `Point` is a zero-dimensional location with an x and y coordinate; a `LineString` is a path composed of a set of more than one `Point`, and a `Polygon` is a surface that has  at least one LineString that starts and stops with the same coordinate. All of these types *also* have "Multi-" variants that indicate a collection of multiple geometries of the same type. So, for instance, Indonesia is a `MultiPolygon` containing  many `Polygons` for each individual island in the country:

```python
gt_polygons.query('ADMIN == "Indonesia"')
```

```python
gt_polygons.query('ADMIN == "Indonesia"').plot()
```

Normally, geographic tables will only have geometries of a single type; records will *all* be `Point` or `LineString`, for instance. However, there is no formal requirement that a *geographic table* has geometries that all have the same type. 

In addition to polygons...

```python
graph = osmnx.graph_from_place("Adachi, JP")
gt_intersections, gt_lines = osmnx.graph_to_gdfs(graph)
```

```python
gt_points = geopandas.read_file("../data/tokyo/tokyo_clean.csv")
```

### Surfaces


### Spatial graphs

```python
graph
```

## Hybrids


Technology is changing and sometimes objects may be best represented as surfaces, surfaces can be analysed as tables (+vectorization)


### Surfaces as tables


- Read raster file
- Convert into table
- Show how to build the geometries if of interest
- Go back to raster data structure


### Tables as surfaces


Discuss cases where, for example you have so many points that there are more than pixels in the screen. Here it makes sense, computationally and conceptually, to aggregate in fine geographies such as grids and store/represent them as such.

Use Tokyo photographs as example for a gridding (e.g. using datashader, which returns `DataArray` objects).


### Networks as graphs *and* tables


Pick up on the discussion in the last paragraph of [this](https://geographicdata.science/book/notebooks/02_spatial_data.html#computational-represenations-data-structures) and illustrate how a street network can be represented as a graph or a table, what are the data structures in Python for each and how to go from one to the other.


---
# Questions

1. One way to convert from `Multi-`type geometries into many individual geometries is using the `explode()` method of a GeoDataFrame. Using the `explode()` method, how many islands are in Indonesia?

```python
gt_polygons.query('ADMIN == "Indonesia"').explode()
```
