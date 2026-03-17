```python
import warnings, osmnx

warnings.filterwarnings("ignore")
osmnx.settings.overpass_settings = '[out:json][timeout:90][date:"2021-10-07T00:00:00Z"]'
```

# Spatial Data

This chapter grounds the ideas discussed in the previous two chapters into a practical context. We consider how data structures, and the data models they represent, are implemented in Python. We also cover how to interact with these data structures. This will happen alongside the code used to manipulate the data in a single computational laboratory notebook. This, then, unites the two concepts of open science and geographical thinking. 

Further, we will spend most of the chapter discussing how Python represents data
*once read* from a file or database, rather than focusing on specific *file*
formats used to store data. This is because the libraries we use will read any
format into one of a few canonical data structures that we discuss in Chapter 1.
We take this approach because these data structures are what we interact with
during our data analysis: they are our interface with the data. File formats, while useful, are secondary to this purpose. Indeed, part of the benefit of Python (and other computing languages) is *abstraction*: the complexities, particularities and quirks associated with each file format are removed as Python represents all data in a few standard ways, regardless of provenance. We take full advantage of this feature here. 

We divide the chapter in two main parts. The first part looks at each of the
three main data structures reviewed in Chapter 1 (*Geographic Thinking*):
geographic tables, surfaces and spatial graphs. Second, we explore combinations
of different data structures that depart from the traditional data
model/structure matchings discussed in Chapter 2. We cover how one data in one
structure can be effectively transferred to another, but also we discuss why that might (or might not) be a good idea in some cases. A final note before we delve into the content of this book is in order: this is not a comprehensive account of *everything* that is possible with each of the data structures we present. Rather, you can think of it as a preview that we will build on throughout the book to showcase much of what is possible with Python.


```python
import pandas
import osmnx
import geopandas
import rioxarray
import xarray
import datashader
import contextily as cx
from shapely import geometry
import matplotlib.pyplot as plt
```

## Fundamentals of geographic data structures

As outlined in Chapter 1, there are a few main data structures that are used in geographic data science: geographic tables (which are generally matched to an object data model), rasters or surfaces (which are generally matched to a field data model), and spatial networks (which are generally matched to a graph data model). We discuss these in turn throughout this section. 

### Geographic tables

Geographic objects are usually matched to what we called the *geographic table*. Geographic tables can be thought of as a tab in a spreadsheet where one of the columns records geometric information. This data structure represents a single geographic object as a row of a table; each column in the table records information about the object, its attributes or features, as we will see below. Typically, there is a special column in this table that records the *geometry* of the object. Computer systems that use this data structure are intended to add geography into a *relational database*, such as PostgreSQL (through its PostGIS extension) or sqlite (through its spatialite extension). Beyond this, however, many data science languages (such as R, Julia, and Python), have packages that adopt this data structure as well (such as `sf`, `GeoTables.jl`, and `geopandas`), and it is rapidly becoming the main data structure for object-based geographic data. 

Before proceeding, though, it helps to mention a quick clarification on
terminology. Throughout this book, regardless of the data structure used, we
will refer to a measurement about an observation as a *feature*. This is
consistent with other work in data science and machine learning. Then, one set
of measurements is a *sample*. For tables, this means a feature is a column, and
a sample is a row. Historically, though, geographic information scientists have
used the word "feature" to mean an individual observation, since a "feature" in
cartography is an entity on a map, and "attribute" to describe characteristics
of that observation. Elsewhere, a feature may be called a "variable," and a
sample is referred to as a "record." So, consistent terminology is important: for this book, a *feature* is one measured trait pertaining to an observation (column), and a *sample* is one set of measurements (row). 

To understand the structure of geographic tables, it will help to read in the `countries_clean.gpkg` dataset included in this book that describes countries in the world. To read in this data, we can use the `read_file()` method in `geopandas`:[^package-v-function]


```python
gt_polygons = geopandas.read_file(
    "../data/countries/countries_clean.gpkg"
)
```

And we can examine the top of the table with the `.head()` method:


```python
gt_polygons.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ADMIN</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Indonesia</td>
      <td>MULTIPOLYGON (((13102705.696 463877.598, 13102...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Malaysia</td>
      <td>MULTIPOLYGON (((13102705.696 463877.598, 13101...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chile</td>
      <td>MULTIPOLYGON (((-7737827.685 -1979875.500, -77...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bolivia</td>
      <td>POLYGON ((-7737827.685 -1979875.500, -7737828....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Peru</td>
      <td>MULTIPOLYGON (((-7737827.685 -1979875.500, -77...</td>
    </tr>
  </tbody>
</table>
</div>



Each row of this table is a single country. Each country only has two features:
the administrative name of the country and the geometry of the country's
boundary. The name of the country is encoded in the `ADMIN` column using the
Python `str` type, which is used to store text-based data. The geometry of the
country's boundary is stored in the `geometry` column, and it is encoded using a special class in Python that is used to represent geometric objects. As with other table-based data structures in Python, every row and column have an index that identifies them uniquely and is rendered in bold on the left-hand side of the table. This geographic table is an instance of the `geopandas.GeoDataFrame` object, used throughout Python's ecosystem to represent geographic data.

Geographic tables store geographic information as an additional column. But, how is this information encoded? To see, we can check the type of the object in the first row:


```python
type(gt_polygons.geometry[0])
```




    shapely.geometry.multipolygon.MultiPolygon



In `geopandas` (as well as other packages representing geographic data), the `geometry` column has special traits which a "normal" column, such as `ADMIN`, does not. For example, when we plot the dataframe, the `geometry` column is used as the main shape to use in the plot, as shown in Figure 1. 


```python
gt_polygons.plot();
```


    
![png](03_spatial_data_files/03_spatial_data_10_0.png)
    


Changing the geometric representation of a sample must be done carefully: since the `geometry` column is special, there are special functions to adjust the geometry. For example, if we wanted to represent each country using its *centroid*, a point in the middle of the shape, then we must take care to make sure that a new geometry column was set properly using the `set_geometry()` method. This can be useful when you want to work with two different geometric representations of the same underlying sample. 

Let us make a map of both the boundary and the centroid of a country. First, to compute the centroid, we can use the `gt_polygons.geometry.centroid` property. This gives us the point that minimizes the average distance from all other points on the boundary of the shape. Storing that back to a column, called `centroid`:


```python
gt_polygons["centroid"] = gt_polygons.geometry.centroid
```

We now have an additional feature:


```python
gt_polygons.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ADMIN</th>
      <th>geometry</th>
      <th>centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Indonesia</td>
      <td>MULTIPOLYGON (((13102705.696 463877.598, 13102...</td>
      <td>POINT (13055431.810 -248921.141)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Malaysia</td>
      <td>MULTIPOLYGON (((13102705.696 463877.598, 13101...</td>
      <td>POINT (12211696.493 422897.505)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Chile</td>
      <td>MULTIPOLYGON (((-7737827.685 -1979875.500, -77...</td>
      <td>POINT (-7959811.948 -4915458.802)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bolivia</td>
      <td>POLYGON ((-7737827.685 -1979875.500, -7737828....</td>
      <td>POINT (-7200010.945 -1894653.148)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Peru</td>
      <td>MULTIPOLYGON (((-7737827.685 -1979875.500, -77...</td>
      <td>POINT (-8277554.831 -1032942.536)</td>
    </tr>
  </tbody>
</table>
</div>



Despite the fact that `centroid` is a geometry (you can tell because each cell starts with `POINT`), it is not currently set as the active geometry for our table. We can switch to the `centroid` column using the `set_geometry()` method. Finally, we can plot the centroid and the boundary of each country after switching the geometry column with `set_geometry()`:


```python
# Plot centroids
ax = gt_polygons.set_geometry("centroid").plot("ADMIN", markersize=5)
# Plot polygons without color filling
gt_polygons.plot(
    "ADMIN", ax=ax, facecolor="none", edgecolor="k", linewidth=0.2
);
```


    
![png](03_spatial_data_files/03_spatial_data_16_0.png)
    


Note again how we can create a map by calling `.plot()` on a `GeoDataFrame`. We can thematically color each feature based on a column by passing the name of that column to the plot method (as we do on with `ADMIN` in this case), and that the current geometry is used.

Thus, as should now be clear, nearly any kind of geographic object can be represented in one (or more) geometry column(s). Thinking about the number of different kinds of shapes or geometries one could use quickly boggles the mind. Fortunately the Open Geospatial Consortium (OGC) has defined a set of "abstract" types that can be used to define any kind of geometry. This specification, codified in ISO 19125-1---the "simple features" specification---defines the formal relationships between these types: a `Point` is a zero-dimensional location with an x and y coordinate, a `LineString` is a path composed of a set of more than one `Point`, and a `Polygon` is a surface that has at least one `LineString` that starts and stops with the same coordinate. All of these types *also* have `Multi` variants that indicate a collection of multiple geometries of the same type. So, for instance, Bolivia is represented as a single polygon:


```python
gt_polygons.query('ADMIN == "Bolivia"')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ADMIN</th>
      <th>geometry</th>
      <th>centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Bolivia</td>
      <td>POLYGON ((-7737827.685 -1979875.500, -7737828....</td>
      <td>POINT (-7200010.945 -1894653.148)</td>
    </tr>
  </tbody>
</table>
</div>




```python
gt_polygons.query('ADMIN == "Bolivia"').plot();
```


    
![png](03_spatial_data_files/03_spatial_data_19_0.png)
    


while Indonesia is a `MultiPolygon` containing many `Polygons` for each individual island in the country:


```python
gt_polygons.query('ADMIN == "Indonesia"')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ADMIN</th>
      <th>geometry</th>
      <th>centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Indonesia</td>
      <td>MULTIPOLYGON (((13102705.696 463877.598, 13102...</td>
      <td>POINT (13055431.810 -248921.141)</td>
    </tr>
  </tbody>
</table>
</div>




```python
gt_polygons.query('ADMIN == "Indonesia"').plot();
```


    
![png](03_spatial_data_files/03_spatial_data_22_0.png)
    


In many cases, geographic tables will have geometries of a single type; records will *all* be `Point` or `LineString`, for instance. However, there is no formal requirement that a *geographic table* has geometries that all have the same type. 

Throughout this book, we will use geographic tables extensively, storing polygons, but also points and lines. We will explore lines a bit more in the second part of this chapter but, for now, let us stop on points for a second. As mentioned above, these are the simplest type of feature in that they do not have any dimension, only a pair of coordinates attached to them. This means that points can sometimes be stored in a non-geographic table, simply using one column for each coordinate. We find an example of this on the Tokyo dataset we will use more later. The data is stored as a comma-separated value table, or `.csv`:


```python
gt_points = pandas.read_csv("../data/tokyo/tokyo_clean.csv")
```

Since we have read it with `pandas`, the table is loaded as a `DataFrame`, with no explicit spatial dimension:


```python
type(gt_points)
```




    pandas.core.frame.DataFrame



If we inspect the table, we find there is not a `geometry` column:


```python
gt_points.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>date_taken</th>
      <th>photo/video_page_url</th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10727420@N00</td>
      <td>139.700499</td>
      <td>35.674000</td>
      <td>2010-04-09 17:26:25.0</td>
      <td>http://www.flickr.com/photos/10727420@N00/4545...</td>
      <td>1.555139e+07</td>
      <td>4.255856e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8819274@N04</td>
      <td>139.766521</td>
      <td>35.709095</td>
      <td>2007-02-10 16:08:40.0</td>
      <td>http://www.flickr.com/photos/8819274@N04/26503...</td>
      <td>1.555874e+07</td>
      <td>4.260667e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62068690@N00</td>
      <td>139.765632</td>
      <td>35.694482</td>
      <td>2008-12-21 15:45:31.0</td>
      <td>http://www.flickr.com/photos/62068690@N00/3125...</td>
      <td>1.555864e+07</td>
      <td>4.258664e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49503094041@N01</td>
      <td>139.784391</td>
      <td>35.548589</td>
      <td>2011-11-11 05:48:54.0</td>
      <td>http://www.flickr.com/photos/49503094041@N01/6...</td>
      <td>1.556073e+07</td>
      <td>4.238684e+06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40443199@N00</td>
      <td>139.768753</td>
      <td>35.671521</td>
      <td>2006-04-06 16:42:49.0</td>
      <td>http://www.flickr.com/photos/40443199@N00/2482...</td>
      <td>1.555899e+07</td>
      <td>4.255517e+06</td>
    </tr>
  </tbody>
</table>
</div>



Many point datasets are provided in this format. To make the most of them, it is convenient to convert them into `GeoDataFrame` tables. There are two steps involved in this process. First, we turn the raw coordinates into geometries:


```python
pt_geoms = geopandas.points_from_xy(
    x=gt_points["longitude"],
    y=gt_points["latitude"],
    # x,y are Earth longitude & latitude
    crs="EPSG:4326",
)
```

Second, we create a `GeoDataFrame` object using these geometries:


```python
gt_points = geopandas.GeoDataFrame(gt_points, geometry=pt_geoms)
```

And now `gt_points` looks and feels exactly like the one of countries we have seen before, with the difference the `geometry` column stores `POINT` geometries:


```python
gt_points.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>longitude</th>
      <th>latitude</th>
      <th>date_taken</th>
      <th>photo/video_page_url</th>
      <th>x</th>
      <th>y</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10727420@N00</td>
      <td>139.700499</td>
      <td>35.674000</td>
      <td>2010-04-09 17:26:25.0</td>
      <td>http://www.flickr.com/photos/10727420@N00/4545...</td>
      <td>1.555139e+07</td>
      <td>4.255856e+06</td>
      <td>POINT (139.70050 35.67400)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8819274@N04</td>
      <td>139.766521</td>
      <td>35.709095</td>
      <td>2007-02-10 16:08:40.0</td>
      <td>http://www.flickr.com/photos/8819274@N04/26503...</td>
      <td>1.555874e+07</td>
      <td>4.260667e+06</td>
      <td>POINT (139.76652 35.70909)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62068690@N00</td>
      <td>139.765632</td>
      <td>35.694482</td>
      <td>2008-12-21 15:45:31.0</td>
      <td>http://www.flickr.com/photos/62068690@N00/3125...</td>
      <td>1.555864e+07</td>
      <td>4.258664e+06</td>
      <td>POINT (139.76563 35.69448)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49503094041@N01</td>
      <td>139.784391</td>
      <td>35.548589</td>
      <td>2011-11-11 05:48:54.0</td>
      <td>http://www.flickr.com/photos/49503094041@N01/6...</td>
      <td>1.556073e+07</td>
      <td>4.238684e+06</td>
      <td>POINT (139.78439 35.54859)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40443199@N00</td>
      <td>139.768753</td>
      <td>35.671521</td>
      <td>2006-04-06 16:42:49.0</td>
      <td>http://www.flickr.com/photos/40443199@N00/2482...</td>
      <td>1.555899e+07</td>
      <td>4.255517e+06</td>
      <td>POINT (139.76875 35.67152)</td>
    </tr>
  </tbody>
</table>
</div>



### Surfaces

Surfaces are used to record data from a field data model. In theory, a field is
a continuous surface and thus has an infinite number of locations at which it
could be measured. In reality, however, fields are measured at a finite sample
of locations that, to provide a sense of continuity and better conform with the
field model, are uniformly structured across space. Surfaces thus are
represented as grids where each cell contains a sample. A grid can also be
thought of as a table with rows and columns but, as we discussed in the previous
chapter, both of them are directly tied to a geographic location. This is in sharp contrast with geographic tables, where geography is confined to a single column.

To explore how Python represents surfaces, we will use an extract for the Brazilian city of Sao Paulo of a [global population dataset](../data/ghsl/build_ghsl_extract). This dataset records population counts in cells of the same dimensions uniformly covering the surface of the Earth. Our extract is available as a GeoTIF file, a variation of the TIF image format that includes geographic information. We can use the `open_rasterio()` method from the `rioxarray` package to read in the GeoTIF:


```python
pop = rioxarray.open_rasterio("../data/ghsl/ghsl_sao_paulo.tif")
```

This reads the data into a `DataArray` object:


```python
type(pop)
```




    xarray.core.dataarray.DataArray



`xarray` is a package to work with multi-dimensional labeled arrays. Let's
unpack this: we can use arrays of not only two dimensions as in a table with
rows and columns, but also with an arbitrary number of them; each of these dimensions is "tracked" by an index that makes it easy and efficient to manipulate. In `xarray`, these indices are called coordinates, and they can be retrieved from our `DataArray` through the `coords` attribute:


```python
pop.coords
```




    Coordinates:
      * band         (band) int64 1
      * x            (x) float64 -4.482e+06 -4.482e+06 ... -4.365e+06 -4.365e+06
      * y            (y) float64 -2.822e+06 -2.822e+06 ... -2.926e+06 -2.926e+06
        spatial_ref  int64 0



Interestingly, our surface has *three* dimensions: `x`, `y`, and `band`. The former two track the latitude and longitude that each cell in our population grid covers. The third one has a single value (1) and, in this context, it is not very useful. But it is easy to imagine contexts where a third dimension would be useful. For example, an optical color image may have three bands: red, blue, and green. More powerful sensors may pick up additional bands, such as near infrared (NIR) or even radio bands. Or, a surface measured over time, like the geocubes that we discussed in Chapter 2, will have bands for each point in time at which the field is measured. A geographic surface will thus have two dimensions recording the location of cells (`x` and `y`), and at least one `band` that records other dimensions pertaining to our data.

An `xarray.DataArray` object contains additional information about the values stored under the `attrs` attribute:


```python
pop.attrs
```




    {'AREA_OR_POINT': 'Area',
     '_FillValue': -200.0,
     'scale_factor': 1.0,
     'add_offset': 0.0}



In this case, we can see this includes information required to convert pixels in the array into locations on the Earth surface (e.g., `transform`, and `crs`), the spatial resolution (250 meters by 250 meters), and other metadata that allows us to better understand where the data comes from and how it is stored.

Thus, our `DataArray` has three dimensions:


```python
pop.shape
```




    (1, 416, 468)



A common operation will be to reduce this to only the two geographic ones. We can do this with the `sel` operator, which allows us to select data by the value of their coordinates:


```python
pop.sel(band=1)
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (y: 416, x: 468)&gt;
[194688 values with dtype=float32]
Coordinates:
    band         int64 1
  * x            (x) float64 -4.482e+06 -4.482e+06 ... -4.365e+06 -4.365e+06
  * y            (y) float64 -2.822e+06 -2.822e+06 ... -2.926e+06 -2.926e+06
    spatial_ref  int64 0
Attributes:
    AREA_OR_POINT:  Area
    _FillValue:     -200.0
    scale_factor:   1.0
    add_offset:     0.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>y</span>: 416</li><li><span class='xr-has-index'>x</span>: 468</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-089ba4e5-3789-4663-8b75-d7e54804a6f7' class='xr-array-in' type='checkbox' checked><label for='section-089ba4e5-3789-4663-8b75-d7e54804a6f7' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>...</span></div><div class='xr-array-data'><pre>[194688 values with dtype=float32]</pre></div></div></li><li class='xr-section-item'><input id='section-c08e7b3d-bffe-4c25-8f23-e1d64a39382d' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c08e7b3d-bffe-4c25-8f23-e1d64a39382d' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>band</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1</div><input id='attrs-a60de97f-0b61-420f-b339-57b331d38feb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-a60de97f-0b61-420f-b339-57b331d38feb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6fb3077c-9985-4912-9cd1-6d30a1c6f82a' class='xr-var-data-in' type='checkbox'><label for='data-6fb3077c-9985-4912-9cd1-6d30a1c6f82a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(1)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-4.482e+06 ... -4.365e+06</div><input id='attrs-7fcdf8b5-cd52-4a46-a5d8-d1e85de738a9' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7fcdf8b5-cd52-4a46-a5d8-d1e85de738a9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-78449baf-0b13-42de-8083-9de1d5c98607' class='xr-var-data-in' type='checkbox'><label for='data-78449baf-0b13-42de-8083-9de1d5c98607' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-4481875., -4481625., -4481375., ..., -4365625., -4365375., -4365125.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.822e+06 ... -2.926e+06</div><input id='attrs-f82b5c05-6932-4909-9aaa-31624669e63e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-f82b5c05-6932-4909-9aaa-31624669e63e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-ddce6f61-4105-473b-ad6d-cc5cebe5ed3c' class='xr-var-data-in' type='checkbox'><label for='data-ddce6f61-4105-473b-ad6d-cc5cebe5ed3c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-2822125., -2822375., -2822625., ..., -2925375., -2925625., -2925875.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>spatial_ref</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-c47ab01d-491a-42df-8cea-38d0f6d925c5' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-c47ab01d-491a-42df-8cea-38d0f6d925c5' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e76211b7-ca35-4d1c-8f94-33bbdefcda2a' class='xr-var-data-in' type='checkbox'><label for='data-e76211b7-ca35-4d1c-8f94-33bbdefcda2a' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>spatial_ref :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>GeoTransform :</span></dt><dd>-4482000.0 250.0 0.0 -2822000.0 0.0 -250.0</dd></dl></div><div class='xr-var-data'><pre>array(0)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-bdb13cf3-b465-4444-a18a-8b1b157fcdb9' class='xr-section-summary-in' type='checkbox'  ><label for='section-bdb13cf3-b465-4444-a18a-8b1b157fcdb9' class='xr-section-summary' >Indexes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>x</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-928621d4-ad11-4213-9be7-f257b4b86d22' class='xr-index-data-in' type='checkbox'/><label for='index-928621d4-ad11-4213-9be7-f257b4b86d22' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([-4481875.0, -4481625.0, -4481375.0, -4481125.0, -4480875.0, -4480625.0,
       -4480375.0, -4480125.0, -4479875.0, -4479625.0,
       ...
       -4367375.0, -4367125.0, -4366875.0, -4366625.0, -4366375.0, -4366125.0,
       -4365875.0, -4365625.0, -4365375.0, -4365125.0],
      dtype=&#x27;float64&#x27;, name=&#x27;x&#x27;, length=468))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>y</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-677fede5-0aaf-4e49-b3e6-165e4f8885f3' class='xr-index-data-in' type='checkbox'/><label for='index-677fede5-0aaf-4e49-b3e6-165e4f8885f3' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([-2822125.0, -2822375.0, -2822625.0, -2822875.0, -2823125.0, -2823375.0,
       -2823625.0, -2823875.0, -2824125.0, -2824375.0,
       ...
       -2923625.0, -2923875.0, -2924125.0, -2924375.0, -2924625.0, -2924875.0,
       -2925125.0, -2925375.0, -2925625.0, -2925875.0],
      dtype=&#x27;float64&#x27;, name=&#x27;y&#x27;, length=416))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-ebeae69b-6ec0-46d3-8087-2b92e7873dc9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ebeae69b-6ec0-46d3-8087-2b92e7873dc9' class='xr-section-summary' >Attributes: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>AREA_OR_POINT :</span></dt><dd>Area</dd><dt><span>_FillValue :</span></dt><dd>-200.0</dd><dt><span>scale_factor :</span></dt><dd>1.0</dd><dt><span>add_offset :</span></dt><dd>0.0</dd></dl></div></li></ul></div></div>



The resulting object is thus a two-dimensional array. Similar to geographic tables, we can quickly plot the values in our dataset:


```python
pop.sel(band=1).plot();
```


    
![png](03_spatial_data_files/03_spatial_data_48_0.png)
    


This gives us a first overview of the distribution of population in the Sao Paulo region. However, if we inspect the map further, we can see that the map includes negative counts! How could this be? As it turns out, missing data is traditionally stored in surfaces not as a class of its own (e.g., `NaN`) but with an impossible value. If we return to the `attrs` printout above, we can see how the `nodatavals` attribute specifies missing data recorded with -200. With that in mind, we can use the `where()` method to select only values that are *not* -200:


```python
pop.where(pop != -200).sel(band=1).plot(cmap="RdPu");
```


    
![png](03_spatial_data_files/03_spatial_data_50_0.png)
    


The colorbar now looks more sensible, and indicates *real* counts, rather than including the missing data placeholder values.

### Spatial graphs

Spatial graphs store connections between objects through space. These
connections may derive from geographical topology (e.g., contiguity), distance,
or more sophisticated dimensions such as interaction flows (e.g., commuting,
trade, communication). Compared to geographic tables and surfaces, spatial
graphs are rather different. First, in most cases they do not record
measurements about given phenomena, but instead focus on *connections*, on
storing relationships between objects as they are facilitated (or impeded in
their absence) by space. Second, because of this relational nature, the data are
organized in a more unstructured fashion: while one sample may be connected to
only one other sample, another one can display several links. This is in stark contrast to geographic tables and surfaces, both of which have a clearly defined structure, shape and dimensionality in which data are organized. 

These particularities translate into a different set of Python data structures. Unlike the previous data structures we have seen, there are quite a few data structures to represent spatial graphs, each optimized for different contexts. One such case is the use of spatial connections in statistical methods such as exploratory data analysis or regression. For this, the most common data structure are spatial weights matrices, to which we devote the next chapter. 

In this chapter, we briefly review a different way of representing spatial graphs that is much closer to the mathematical concept of a graph. A graph is composed of *nodes* that are linked together by *edges*. In a spatial network, *nodes* may represent geographical places, and thus have a specific location; likewise, *edges* may represent geographical paths between these places. Networks require both *nodes* and *edges* to analyze their structure. 

For illustration, we will rely on the `osmnx` library, which can query data from OpenStreetMap. For example, we extract the street-based graph of Yoyogi Park, near our earlier data from Tokyo:


```python
graph = osmnx.graph_from_place("Yoyogi Park, Shibuya, Tokyo, Japan")
```


```python
osmnx.save_graphml(graph, "../data/cache/yoyogi_park_graph.graphml")
```

The code snippet above sends the query to the OpenStreetMap server to fetch the data. Note that the cell above _requires_ internet connectivity to work. If you are working on the book _without_ connectivity, a cached version of the graph is available on the data folder and can be read as:


```python
graph = osmnx.load_graphml("../data/cache/yoyogi_park_graph.graphml")
```

Once the data is returned to `osmnx`, it gets processed into the `graph` Python representation:


```python
type(graph)
```




    networkx.classes.multidigraph.MultiDiGraph



We can have a quick inspection of the structure of the graph with the `plot_graph` method:


```python
osmnx.plot_graph(graph);
```


    
![png](03_spatial_data_files/03_spatial_data_59_0.png)
    


The resultant `graph` object is actually a `MultiDiGraph` from `networkx`, a graph library written in Python. The graph here is stored as a collection of 106 nodes (street intersections):


```python
len(graph.nodes)
```




    102



and 287 edges (streets) that connect them:


```python
len(graph.edges)
```




    275



Each of these elements can be queried to obtain more information such as the location and ID of a node:


```python
graph.nodes[1520546819]
```




    {'y': 35.6711267, 'x': 139.6925951, 'street_count': 4}



The characteristics of an edge:


```python
graph.edges[(1520546819, 3010293622, 0)]
```




    {'osmid': 138670840,
     'highway': 'footway',
     'oneway': False,
     'reversed': False,
     'length': 59.113,
     'geometry': <LINESTRING (139.693 35.671, 139.693 35.671, 139.693 35.671)>}



Or how the different components of the graph relate to each other. For example, what other nodes are directly connected to node `1520546819`?


```python
list(graph.adj[1520546819].keys())
```




    [3010293622, 5764960322, 1913626649, 1520546959]



Thus, networks are easy to represent in Python, and are one of the three main data structures in geographic data science. 



## Hybrids

We have just seen how geographic tables, surfaces, and networks map onto `GeoDataFrame`, `DataArray` and `Graph` objects in Python, respectively. These represent the conventional pairings that align data models to data structures with Python representations. However, while the conventional pairings are well-used, there are others in active use and many more to yet be developed. Interestingly, many new pairings are driven by new developments in technology, enabling approaches that were not possible in the past or creating situations (e.g., large datasets) that make the conventional approach limiting. Therefore, in this second section of the chapter, we step a bit "out of the box" to explore cases in which it may make sense to represent a dataset with a data structure that might not be the most obvious initial choice. 


### Surfaces as tables

The first case we explore is treating surfaces as (geo-)tables. In this context, we shift from an approach where each dimension has a clear mapping to a spatial or temporal aspect of the dataset, to one where each sample, cell of the surface/cube is represented as a row in a table. This approach runs contrary to the general consensus that fields are best represented as surfaces or rasters because that allows us to index space and time "by default" based on the location of values within the data structure. Shifting to a tabular structure implies either losing that space-time reference, or having to build it manually with auxiliary objects (e.g., a spatial graph). In almost any case, operating on this format is less efficient than it *could* be if we had bespoke algorithms built around surface structures. Finally, from a more conceptual point of view, treating pixels as independent realizations of a process that we *know* is continuous can be computationally inefficient and statistically flawed. 

This perspective, however, also involves important benefits. First, sometimes we *don't* need location for our particular application. Maybe we are interested in calculating overall descriptive statistics; or maybe we need to run an analysis that is entirely atomic in the sense that it operates on each sample in isolation from all the other ones.  Second, by "going tabular" we recast our specialized, spatial data into the most common data structure available, for which a large amount of commodity technology is built. This means many new tools can be used for analysis. So-called "big data" technologies, such as distributed systems, are much more common, robust, and tested for tabular data than for spatial surfaces. *If* we can translate our spatial challenge into a tabular challenge, we can immediately plug in technology that is more optimized and, in some cases, reliable. Further, some analytic toolboxes common in (geographic) data science are entirely built around tabular structures. Machine learning packages such as `scikit-learn`, or some spatial analytics (such as most methods in the Pysal family of packages) are designed around this data structure. Converting our surfaces into tables thus allows us to plug into a much wider suite of (potentially) efficient tools and techniques.

We will see two ways of going from surfaces to tables: one converts every pixel into a table row, and another aggregates pixels into pre-determined polygons.

#### One pixel at a time

Technically, going from surface to table involves traversing from `xarray` to `pandas` objects. This is actually a well established bridge. To illustrate it with an example, let's revisit the population counts in [Sao Paulo](../data/ghsl/build_ghsl_extract) used earlier. We can read the surface into a `DataArray` object with `rioxarray`, a special package designed to work with raster data in `xarray`. We can use its `open_rasterio()` method to read in the data:


```python
surface = rioxarray.open_rasterio("../data/ghsl/ghsl_sao_paulo.tif")
```

Transferring to a table is as simple as calling the `DataArray`'s `to_series()` method:


```python
t_surface = surface.to_series()
```

The resulting object is a `pandas.Series` object indexed on each of the dimensions of the original `DataArray`:


```python
t_surface.head()
```




    band  y           x         
    1     -2822125.0  -4481875.0   -200.0
                      -4481625.0   -200.0
                      -4481375.0   -200.0
                      -4481125.0   -200.0
                      -4480875.0   -200.0
    dtype: float32



At this point, everything we know about `pandas` and tabular data applies! For example, it might be more convenient to express it as a `DataFrame`:


```python
t_surface = t_surface.reset_index().rename(columns={0: "Value"})
```

With the power of a tabular library, some queries and filter operations become much easier. For example, finding cells with more than 1,000 people can be done with the usual `query()` method.[^xarray-query]


```python
t_surface.query("Value > 1000").info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 7734 entries, 3785 to 181296
    Data columns (total 4 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   band    7734 non-null   int64  
     1   y       7734 non-null   float64
     2   x       7734 non-null   float64
     3   Value   7734 non-null   float32
    dtypes: float32(1), float64(2), int64(1)
    memory usage: 271.9 KB


The table we have built has no geometries associated with it, only rows representing pixels. It takes a bit more effort, but it is possible to convert it, or a subset of it, into a full-fledged geographic table, where each pixel includes the grid geometry it represents. For this task, we develop a function that takes a row from our table and the resolution of the surface, and returns its geometry:


```python
def row2cell(row, res_xy):
    res_x, res_y = res_xy  # Extract resolution for each dimension
    # XY Coordinates are centered on the pixel
    minX = row["x"] - (res_x / 2)
    maxX = row["x"] + (res_x / 2)
    minY = row["y"] + (res_y / 2)
    maxY = row["y"] - (res_y / 2)
    poly = geometry.box(
        minX, minY, maxX, maxY
    )  # Build squared polygon
    return poly
```

For example:


```python
row2cell(t_surface.loc[0, :], surface.rio.resolution())
```




    
![svg](03_spatial_data_files/03_spatial_data_86_0.svg)
    



One of the benefits of this approach is that we do not require entirely filled surfaces and can only record pixels where we have data. For the example above or cells with more than 1,000 people, we could create the associated geo-table as follows:


```python
max_polys = (
    t_surface.query(
        "Value > 1000"
    )  # Keep only cells with more than 1k people
    .apply(  # Build polygons for selected cells
        row2cell, res_xy=surface.rio.resolution(), axis=1
    )
    .pipe(  # Pipe result from apply to convert into a GeoSeries
        geopandas.GeoSeries, crs=surface.rio.crs
    )
)
```

And generate a map with the same tooling that we use for any standard geo-table:


```python
# Plot polygons
ax = max_polys.plot(edgecolor="red", figsize=(9, 9))
# Add basemap
cx.add_basemap(
    ax, crs=surface.rio.crs, source=cx.providers.CartoDB.Voyager
);
```


    
![png](03_spatial_data_files/03_spatial_data_90_0.png)
    


Finally, once we have operated on the data as a table, we may want to return to a surface-like data structure. This involves taking the same journey in the opposite direction as how we started. The sister method of `to_series` in `xarray` is `from_series`:


```python
new_da = xarray.DataArray.from_series(
    t_surface.set_index(["band", "y", "x"])["Value"]
)
new_da
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;Value&#x27; (band: 1, y: 416, x: 468)&gt;
array([[[-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.],
        ...,
        [-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.]]], dtype=float32)
Coordinates:
  * band     (band) int64 1
  * y        (y) float64 -2.926e+06 -2.926e+06 ... -2.822e+06 -2.822e+06
  * x        (x) float64 -4.482e+06 -4.482e+06 ... -4.365e+06 -4.365e+06</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'Value'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>band</span>: 1</li><li><span class='xr-has-index'>y</span>: 416</li><li><span class='xr-has-index'>x</span>: 468</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-ead57fe2-181b-4232-b844-393aa449811b' class='xr-array-in' type='checkbox' checked><label for='section-ead57fe2-181b-4232-b844-393aa449811b' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>-200.0 -200.0 -200.0 -200.0 -200.0 ... -200.0 -200.0 -200.0 -200.0</span></div><div class='xr-array-data'><pre>array([[[-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.],
        ...,
        [-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.],
        [-200., -200., -200., ..., -200., -200., -200.]]], dtype=float32)</pre></div></div></li><li class='xr-section-item'><input id='section-9eb2934b-04cc-47df-bdc1-47cefb367952' class='xr-section-summary-in' type='checkbox'  checked><label for='section-9eb2934b-04cc-47df-bdc1-47cefb367952' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>band</span></div><div class='xr-var-dims'>(band)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1</div><input id='attrs-df84949b-1b7a-47dc-98d7-805c805d3c8c' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-df84949b-1b7a-47dc-98d7-805c805d3c8c' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3dcb9d10-dc31-40d9-9548-3e58fa51a2a8' class='xr-var-data-in' type='checkbox'><label for='data-3dcb9d10-dc31-40d9-9548-3e58fa51a2a8' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-2.926e+06 ... -2.822e+06</div><input id='attrs-89312346-ad22-4d6e-965d-0dc5646936e1' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-89312346-ad22-4d6e-965d-0dc5646936e1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-e94b9d34-84a7-4707-99ff-2cd687b77899' class='xr-var-data-in' type='checkbox'><label for='data-e94b9d34-84a7-4707-99ff-2cd687b77899' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-2925875., -2925625., -2925375., ..., -2822625., -2822375., -2822125.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-4.482e+06 ... -4.365e+06</div><input id='attrs-57c67d6c-af2a-4a43-b018-af8705162375' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-57c67d6c-af2a-4a43-b018-af8705162375' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c9be0e7a-3a46-4f3a-8506-3221241fe24b' class='xr-var-data-in' type='checkbox'><label for='data-c9be0e7a-3a46-4f3a-8506-3221241fe24b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-4481875., -4481625., -4481375., ..., -4365625., -4365375., -4365125.])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-54a82a74-6d6a-404c-b26d-6684cb2b1a4d' class='xr-section-summary-in' type='checkbox'  ><label for='section-54a82a74-6d6a-404c-b26d-6684cb2b1a4d' class='xr-section-summary' >Indexes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-index-name'><div>band</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-71e19c0a-1c24-45a7-93e3-28f4314a6556' class='xr-index-data-in' type='checkbox'/><label for='index-71e19c0a-1c24-45a7-93e3-28f4314a6556' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([1], dtype=&#x27;int64&#x27;, name=&#x27;band&#x27;))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>y</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-018a923a-7d80-4d17-ad39-5c1d99885a6f' class='xr-index-data-in' type='checkbox'/><label for='index-018a923a-7d80-4d17-ad39-5c1d99885a6f' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([-2925875.0, -2925625.0, -2925375.0, -2925125.0, -2924875.0, -2924625.0,
       -2924375.0, -2924125.0, -2923875.0, -2923625.0,
       ...
       -2824375.0, -2824125.0, -2823875.0, -2823625.0, -2823375.0, -2823125.0,
       -2822875.0, -2822625.0, -2822375.0, -2822125.0],
      dtype=&#x27;float64&#x27;, name=&#x27;y&#x27;, length=416))</pre></div></li><li class='xr-var-item'><div class='xr-index-name'><div>x</div></div><div class='xr-index-preview'>PandasIndex</div><div></div><input id='index-d8b487b3-cdc5-4167-b207-31b9651a2c48' class='xr-index-data-in' type='checkbox'/><label for='index-d8b487b3-cdc5-4167-b207-31b9651a2c48' title='Show/Hide index repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-index-data'><pre>PandasIndex(Index([-4481875.0, -4481625.0, -4481375.0, -4481125.0, -4480875.0, -4480625.0,
       -4480375.0, -4480125.0, -4479875.0, -4479625.0,
       ...
       -4367375.0, -4367125.0, -4366875.0, -4366625.0, -4366375.0, -4366125.0,
       -4365875.0, -4365625.0, -4365375.0, -4365125.0],
      dtype=&#x27;float64&#x27;, name=&#x27;x&#x27;, length=468))</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-70a50459-eda1-4913-9be8-471a7ffe304a' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-70a50459-eda1-4913-9be8-471a7ffe304a' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



#### Pixels to polygons

A second use case involves moving surfaces directly into geographic tables by
aggregating pixels into pre-specified geometries. For this illustration, we will
use the digital elevation model [(DEM)](../data/nasadem/build_nasadem_sd) surface containing elevation for the San Diego (US) region, and the set of [census tracts](../data/sandiego/sandiego_tracts_cleaning). For an example, we will investigate the average altitude of each neighborhood.

Let's start by reading the data. First, the elevation model:


```python
dem = rioxarray.open_rasterio("../data/nasadem/nasadem_sd.tif").sel(
    band=1
)
dem.where(dem > 0).plot.imshow();
```


    
![png](03_spatial_data_files/03_spatial_data_94_0.png)
    


And the neighborhood areas (tracts) from the census:


```python
sd_tracts = geopandas.read_file(
    "../data/sandiego/sandiego_tracts.gpkg"
)
sd_tracts.plot();
```


    
![png](03_spatial_data_files/03_spatial_data_96_0.png)
    


There are several approaches to compute the average altitude of each neighborhood. We will use `rioxarray`to clip parts of the surface *within* a given set of geometries. By this, we mean that we will cut out the part of the raster that falls within each geometry, and then we can summarize the values in that sub-raster. This is sometimes called computing a "zonal statistic" from a raster, where the "zone" is the geometry.

Since this is somewhat complicated, we will start with a single polygon. For the illustration, we will use the largest one, located on the eastern side of San Diego. We can find the ID of the polygon with:


```python
largest_tract_id = sd_tracts.query(
    f"area_sqm == {sd_tracts['area_sqm'].max()}"
).index[0]

largest_tract_id
```




    627



And then pull out the polygon itself for the illustration:


```python
largest_tract = sd_tracts.loc[largest_tract_id, "geometry"]
```

Clipping the section of the surface that is within the polygon in the DEM can be achieved with the `rioxarray` extension to clip surfaces based on geometries:


```python
# Clip elevation for largest tract
dem_clip = dem.rio.clip(
    [largest_tract.__geo_interface__], crs=sd_tracts.crs
)

# Set up figure to display against polygon shape
f, axs = plt.subplots(1, 2, figsize=(6, 3))
# Display elevation of largest tract
dem_clip.where(dem_clip > 0).plot(ax=axs[0], add_colorbar=True)

# Display largest tract polygon
sd_tracts.loc[[largest_tract_id]].plot(
    ax=axs[1], edgecolor="red", facecolor="none"
)
axs[1].set_axis_off()
# Add basemap
cx.add_basemap(
    axs[1], crs=sd_tracts.crs, source=cx.providers.Esri.WorldTerrain
);
```


    
![png](03_spatial_data_files/03_spatial_data_102_0.png)
    


Once we have elevation measurements for all the pixels within the tract, the average one can be calculated with `mean()`:


```python
dem_clip.where(dem_clip > 0).mean()
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body[data-theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-index-preview {
  grid-column: 2 / 5;
  color: var(--xr-font-color2);
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data,
.xr-index-data-in:checked ~ .xr-index-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-index-name div,
.xr-index-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data,
.xr-index-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2,
.xr-no-icon {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray ()&gt;
array(585.11346, dtype=float32)
Coordinates:
    band         int64 1
    spatial_ref  int64 0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-e5ba1864-463e-4b79-9737-3f71144561f4' class='xr-array-in' type='checkbox' checked><label for='section-e5ba1864-463e-4b79-9737-3f71144561f4' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>585.1</span></div><div class='xr-array-data'><pre>array(585.11346, dtype=float32)</pre></div></div></li><li class='xr-section-item'><input id='section-580fd496-91b2-4449-b35e-a50ae93f0fad' class='xr-section-summary-in' type='checkbox'  checked><label for='section-580fd496-91b2-4449-b35e-a50ae93f0fad' class='xr-section-summary' >Coordinates: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>band</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1</div><input id='attrs-9bff7f11-59ea-45a3-8d11-f99aa0c0ba74' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9bff7f11-59ea-45a3-8d11-f99aa0c0ba74' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9c024ee7-d76d-4bd9-bacc-61d73a756435' class='xr-var-data-in' type='checkbox'><label for='data-9c024ee7-d76d-4bd9-bacc-61d73a756435' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array(1)</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>spatial_ref</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-13240df4-4929-470e-9b39-843a57db8364' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-13240df4-4929-470e-9b39-843a57db8364' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f477c3ef-38e6-4857-8f8c-a8bf07775766' class='xr-var-data-in' type='checkbox'><label for='data-f477c3ef-38e6-4857-8f8c-a8bf07775766' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>semi_major_axis :</span></dt><dd>6378137.0</dd><dt><span>semi_minor_axis :</span></dt><dd>6356752.314245179</dd><dt><span>inverse_flattening :</span></dt><dd>298.257223563</dd><dt><span>reference_ellipsoid_name :</span></dt><dd>WGS 84</dd><dt><span>longitude_of_prime_meridian :</span></dt><dd>0.0</dd><dt><span>prime_meridian_name :</span></dt><dd>Greenwich</dd><dt><span>geographic_crs_name :</span></dt><dd>WGS 84</dd><dt><span>horizontal_datum_name :</span></dt><dd>World Geodetic System 1984</dd><dt><span>grid_mapping_name :</span></dt><dd>latitude_longitude</dd><dt><span>spatial_ref :</span></dt><dd>GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0,AUTHORITY[&quot;EPSG&quot;,&quot;8901&quot;]],UNIT[&quot;degree&quot;,0.0174532925199433,AUTHORITY[&quot;EPSG&quot;,&quot;9122&quot;]],AXIS[&quot;Latitude&quot;,NORTH],AXIS[&quot;Longitude&quot;,EAST],AUTHORITY[&quot;EPSG&quot;,&quot;4326&quot;]]</dd><dt><span>GeoTransform :</span></dt><dd>-116.61847222222222 0.0002777777777777799 0.0 33.42902777777778 0.0 -0.0002777777777777784</dd></dl></div><div class='xr-var-data'><pre>array(0)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-49fd5cd7-e207-4157-8d41-e8bce7c7b24f' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-49fd5cd7-e207-4157-8d41-e8bce7c7b24f' class='xr-section-summary'  title='Expand/collapse section'>Indexes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'></ul></div></li><li class='xr-section-item'><input id='section-f136223c-d917-44c5-a32e-7814a1724d63' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-f136223c-d917-44c5-a32e-7814a1724d63' class='xr-section-summary'  title='Expand/collapse section'>Attributes: <span>(0)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'></dl></div></li></ul></div></div>



Now, to scale this to the entire geo-table, there are several approaches. Each has its benefits and disadvantages. We opt for applying the method above to each row of the table. We define an auxiliary function that takes a row containing one of our tracts and returns its elevation:


```python
def get_mean_elevation(row, dem):
    # Extract geometry object
    geom = row["geometry"].__geo_interface__
    # Clip the surface to extract pixels within `geom`
    section = dem.rio.clip([geom], crs=sd_tracts.crs)
    # Calculate mean elevation
    elevation = float(section.where(section > 0).mean())
    return elevation
```

Applied to the same tract, it returns the same average elevation:


```python
get_mean_elevation(sd_tracts.loc[largest_tract_id, :], dem)
```




    585.1134643554688



This method can then be run on each polygon in our series using the `apply()` method:


```python
elevations = sd_tracts.head().apply(
    get_mean_elevation, dem=dem, axis=1
)
elevations
```




    0      7.144268
    1     35.648491
    2     53.711388
    3     91.358780
    4    187.312027
    dtype: float64



This simple approach illustrates the main idea well: find the cells that pertain to a given geometry and summarize their values in some manner. This can be done with any kind of geometry. Further, this simple method plays well with `xarray` surface structures and is scalable in that it is not too involved to run in parallel and distributed form using libraries like `dask`. Further, it can be extended using arbitrary Python functions, so it is simple to extend.

However, this approach can be quite slow in big data. A more efficient
alternative for our example uses the `rasterstats` library. This is a
purpose-built library to construct so-called "zonal statistics" from
surfaces. Here, the "zones" are the polygons and the "surface" is our
`DataArray`. Generally, this library will be faster than the simpler approach
used above, but it may be more difficult to extend or adapt:


```python
from rasterstats import zonal_stats

elevations2 = zonal_stats(
    sd_tracts.to_crs(dem.rio.crs),  # Geotable with zones
    "../data/nasadem/nasadem_sd.tif",  # Path to surface file
)
elevations2 = pandas.DataFrame(elevations2)
```


```python
elevations2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-12.0</td>
      <td>18.0</td>
      <td>3.538397</td>
      <td>3594</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.0</td>
      <td>94.0</td>
      <td>35.616395</td>
      <td>5709</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-5.0</td>
      <td>121.0</td>
      <td>48.742630</td>
      <td>10922</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31.0</td>
      <td>149.0</td>
      <td>91.358777</td>
      <td>4415</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-32.0</td>
      <td>965.0</td>
      <td>184.284941</td>
      <td>701973</td>
    </tr>
  </tbody>
</table>
</div>



To visualize these results, we can make an elevation map:


```python
# Set up figure
f, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot elevation surface
dem.where(  # Keep only pixels above sea level
    dem
    > 0
    # Reproject to CRS of tracts
).rio.reproject(
    sd_tracts.crs
    # Render surface
).plot.imshow(
    ax=axs[0], add_colorbar=False
)

# Plot tract geography
sd_tracts.plot(ax=axs[1])

# Plot elevation on tract geography
sd_tracts.assign(  # Append elevation values to tracts
    elevation=elevations2["mean"]
).plot(  # Plot elevation choropleth
    "elevation", ax=axs[2]
);
```


    
![png](03_spatial_data_files/03_spatial_data_115_0.png)
    


### Tables as surfaces

The case for converting tables into surfaces is perhaps less controversial than that for turning surfaces into tables. This is an approach we can take in cases where we are interested in the *overall* distribution of objects (usually points) and we have so many that it is not only technically more efficient to represent them as a surface, but conceptually it is also easier to think about the points as uneven measurements from a continuous field. To illustrate this approach, we will use the dataset of [Tokyo photographs](../data/tokyo/tokyo_cleaning) we loaded above into `gt_points`.

From a purely technical perspective, for datasets with too many points, representing every point in the data on a screen can be seriously overcrowded:


```python
gt_points.plot();
```


    
![png](03_spatial_data_files/03_spatial_data_117_0.png)
    


In this figure, it is hard to tell anything about the density of points in the center of the image due to *overplotting*: while points *theoretically* have no width, they *must* have some dimension in order for us to see them! Therefore, point *markers* often plot on top of one another, obscuring the true pattern and density in dense areas. Converting the dataset from a geo-table into a surface involves laying out a grid and counting how many points fall within each cell. In one sense, this is the reverse operation to what we saw when computing zonal statistics in the previous section: instead of aggregating cells into objects, we aggregate objects into cells. Both operations, however, involve aggregation that reduces the amount of information present in order to make the (new) data more manageable. 

In Python, we can rely on the `datashader` library, which does all the computation in a very efficient way. This process involves two main steps. First, we set up the grid (or canvas, `cvs`) into which we want to aggregate points:


```python
cvs = datashader.Canvas(plot_width=60, plot_height=60)
```

Then we "transfer" the points into the grid:


```python
grid = cvs.points(gt_points, x="longitude", y="latitude")
```

The resulting `grid` is a standard `DataArray` object that we can then manipulate as we have seen before. When plotted below, the amount of detail that the resampled data allows for is much greater than when the points were visualized alone. This is shown in Figure 14. 


```python
f, axs = plt.subplots(1, 2, figsize=(14, 6))
gt_points.plot(ax=axs[0])
grid.plot(ax=axs[1]);
```


    
![png](03_spatial_data_files/03_spatial_data_123_0.png)
    


### Networks as graphs *and* tables

In the previous chapter, we saw networks as data structures that store *connections* between objects. We also discussed how this broad definition includes many interpretations that focus on different aspects of the networks. While spatial analytics may use graphs to record the topology of a table of objects such as polygons, transport applications may treat the network representation of the street layout as a set of objects itself, in this case lines. In this final section we show how one can flip back and forth between one representation and another, to take advantage of different aspects.

We start with the `graph` object from the [previous section](#Spatial-graphs). Remember this captures the street layout around Yoyogi park in Tokyo. We have seen how, stored under this data structure, it is easy to query which node is connected to which, and which ones are at the end of a given edge. 

However, in some cases, we may want to convert the graph into a structure that allows us to operate on each component of the network independently. For example, we may want to map streets, calculate segment lengths, or draw buffers around each intersection. These are all operations that do not require topological information, that are standard for geo-tables, and that are irrelevant to the graph structure. In this context, it makes sense to convert our `graph` to two geo-tables, one for intersections (graph nodes) and one for street segments (graph edges). In `osmnx`, we can do that with the built-in converter:


```python
gt_intersections, gt_lines = osmnx.graph_to_gdfs(graph)
```

Now each of the resulting geo-tables is a collection of geographic objects:


```python
gt_intersections.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
      <th>x</th>
      <th>street_count</th>
      <th>highway</th>
      <th>geometry</th>
    </tr>
    <tr>
      <th>osmid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>886196069</th>
      <td>35.670087</td>
      <td>139.694333</td>
      <td>3</td>
      <td>NaN</td>
      <td>POINT (139.69433 35.67009)</td>
    </tr>
    <tr>
      <th>886196073</th>
      <td>35.669725</td>
      <td>139.699508</td>
      <td>3</td>
      <td>NaN</td>
      <td>POINT (139.69951 35.66972)</td>
    </tr>
    <tr>
      <th>886196100</th>
      <td>35.669442</td>
      <td>139.699708</td>
      <td>3</td>
      <td>NaN</td>
      <td>POINT (139.69971 35.66944)</td>
    </tr>
    <tr>
      <th>886196106</th>
      <td>35.670422</td>
      <td>139.698564</td>
      <td>4</td>
      <td>NaN</td>
      <td>POINT (139.69856 35.67042)</td>
    </tr>
    <tr>
      <th>886196117</th>
      <td>35.671256</td>
      <td>139.697470</td>
      <td>3</td>
      <td>NaN</td>
      <td>POINT (139.69747 35.67126)</td>
    </tr>
  </tbody>
</table>
</div>




```python
gt_lines.info()
```

    <class 'geopandas.geodataframe.GeoDataFrame'>
    MultiIndex: 275 entries, (886196069, 1520546857, 0) to (7684088896, 3010293702, 0)
    Data columns (total 9 columns):
     #   Column    Non-Null Count  Dtype   
    ---  ------    --------------  -----   
     0   osmid     275 non-null    object  
     1   highway   275 non-null    object  
     2   oneway    275 non-null    bool    
     3   reversed  275 non-null    object  
     4   length    275 non-null    float64 
     5   geometry  275 non-null    geometry
     6   bridge    8 non-null      object  
     7   name      9 non-null      object  
     8   access    2 non-null      object  
    dtypes: bool(1), float64(1), geometry(1), object(6)
    memory usage: 28.3+ KB


If we were in the opposite situation, where we had a set of street segments and their intersections in geo-table form, we can generate the graph representation with the `graph_from_gdfs` sister method:


```python
new_graph = osmnx.graph_from_gdfs(gt_intersections, gt_lines)
```

The resulting object will behave in the same way as our original `graph`.

## Conclusion

In conclusion, this chapter provides an overview of the mappings between data models, presented in Chapter 2, and data structures that are common in Python. Beyond the data structures discussed here, the Python ecosystem is vast, deep, and ever-changing. Part of this is the ease with which you can create your own representations to express different aspects of a problem at hand. However, by focusing on our shared representations and the interfaces between these representations, you can generally conduct any analysis you need. By creating unique, bespoke representations, your analysis might be more efficient, but you can also inadvertently isolate it from other developers and render useful tools inoperable. Therefore, a solid understanding of the basic data structures (the `GeoDataFrame`, `DataArray`, and `Graph`) will be sufficient to support nearly any analysis you need to conduct. 

## Questions

1. One way to convert from `Multi-`type geometries into many individual geometries is using the `explode()` method of a GeoDataFrame. Using the `explode()` method, can you find out how many islands are in Indonesia?

2. Using `osmnx`, are you able to extract the street graph for your hometown?

3. As you have seen with the `osmnx.graph_to_gdfs()` method, it is possible to convert a graph into the constituent nodes and edges. Graphs have many other kinds of non-geographical representations. Many of these are provided in `networkx` methods that start with `to_`. How many representations of graphs are currently supported?

4. Using `networkx.to_edgelist()`, what "extra" information does `osmnx` include when building the dataframe for edges?

5. Instead of computing the average elevation for each neighborhood in San Diego, can you answer the following queries?
  - What neighborhood (or neighborhoods) have the *the highest average elevation*?
  - What neighborhood (or neighborhoods) have *the highest point single point*?
  - Can you find the neighborhood (or neighborhoods) with *the largest elevation change*?

[^package-v-function]: We will generally use two curved brackets (such as `method_name()`) to denote a *function*, and will omit them (such as `package`) when referring to an object or package.]
[^xarray-query]: Although, if all you want to do is this type of query, `xarray` is well equipped for this kind of task too.
