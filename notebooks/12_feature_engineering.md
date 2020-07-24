---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Spatial feature engineering


Ways to "stick" space into models that are not necessarily spatial.

<!-- #region -->
**spatial feature engineering**: synthesizing information using spatial relationships either within the data or across data. 

This is one way of "spatializing" data that is included in models. This is not about fitting *spatial models* that use
> the kohonen quote about spatially-correlated learning in SOMs

it is about figuring out representations of geographical relationships and using them in typical non-spatial models. 



Geographying

Spatializing

*(note: fit the distinction between using spatialized data vs. using spatial models into the regression chapter, ch. 11)*. 
<!-- #endregion -->

```python
import geopandas, pandas, libpysal.weights as weights, contextily
import matplotlib.pyplot as plt
import numpy
import osmnx
import rasterio
from rasterio.plot import show as rioshow
```

Throughout this chapter, we will use a common dataset to which we want to append more information through geography. For the illustration, we will use the set of [AirBnb properties](../data/airbnb/regression_cleaning). Let's read it:

```python
airbnbs = geopandas.read_file('../data/airbnb/regression_db.geojson')\
                   .set_index("id")
```

## Feature Engineering Using Map Matching
*Using spatial relationships between two datasets to transfer information from one to another for a model.*
"Space is the ultimate linkage key." - DAB



### Counting *nearby* features


A first, conceptually straightforward, approach is to augment our dataset by counting how many points of a different dataset are in the vicinity of each observation. For example, we might want to know how many bars and restaurants each AirBnb has within a given radious. This count can then become an additional  feature of our dataset, stored in a new column of `airbnbs`.

To obtain information on the location of restaurants and bars, we can download it from OpenStreetMap directly using `osmnx`. We first query all the points of interest (POIs) within the area our points cover, and then filter out everything except restaurants and bars. For that, we require to get a polygon that covers all our `airbnbs` points; an easy approach is a convex hull:

```python
airbnbs_ch = airbnbs.unary_union.convex_hull
airbnbs_ch
```

Now we can use this polygon as query for OpenStreetMap (note this step requires internet connection as it is querying a remote server):

```python
%%time
pois = osmnx.pois_from_polygon(airbnbs_ch,
                               tags={"amenity": ['restaurant', 'bar']}
                              )
```

This returns every location where the words "restaurant" and "bar" are in its label. However, this includes more options:

```python
pois["amenity"].unique()
```

To retain only those under `restaurant` and `bar`, we can query further our table:

```python
pois = pois.query('amenity in ("restaurant", "bar")')
```

Once loaded into `pois` as a `GeoDataFrame`, let's take a peak at their location, as compared with AirBnb spots:

```python
f,ax = plt.subplots(1,figsize=(12, 12))
airbnbs.plot(ax=ax, marker='.')
pois.plot(ax=ax, color='r')
contextily.add_basemap(ax, 
                       crs=airbnbs.crs.to_string(), 
                       source=contextily.providers.Stamen.Toner
                      )
```

Now our intention is to create a new feature for the AirBnb dataset --a new column in `airbnbs`-- that incorporates information about how many POIs are *nearby* each property. Let us first walk through what we need to do conceptually:

1. Decide what is *nearby*, which will dictate how far we go from each AirBnb to find POIs and count them. For this example, we will consider 500m
2. Find, for each AirBnb, POIs *within* that radious
3. Count how many POIs are withing the specified radious of each AirBnb

Let us now translate the list above into code. For 1., we need to be able to measure distances in metres. `airbnbs` is originally expressed in degrees:

```python
airbnbs.crs
```

So are the POIs, so we need to convert both into a projection on metres, NAD83 for example:

```python
airbnbs_nad83 = airbnbs.to_crs("EPSG:6350")
pois_nad83 = pois.to_crs("EPSG:6350")
```

We can create the radious of 500m around each AirBnb by drawing a buffer of that length:

```python
abb_buffer = airbnbs_nad83.buffer(500)
```

These buffers can be intersected with our POIs through a spatial join, which links geometries based on spatial relationships (or predicates). What we are aiming to is linking,  to every POI, the ID of the AirBnb for which the buffer contains the POI:

```python
j = geopandas.sjoin(pois_nad83,
                    airbnbs_nad83.set_geometry(abb_buffer)\
                                 .reset_index()\
                                 [["id", "geometry"]],
                    op="within"
                   )
```

The resulting joined object `j` contains a row for every pair of POI and AirBnb that are linked. From there, we can apply a group-by operation, using the AirBnb ID, and count how many POIs each AirBnb has within 500m of distance:

```python
poi_count = j.groupby("id")["osmid"].count()
poi_count.head()
```

The resulting `Series` is indexed on the AirBnb IDs, so we can assign it to the original `airbnbs` table. In this case, we know by construction that missing AirBnbs in `poi_count` do not have any POI within 500m, so we can fill missing values in the column with zeros.

```python
airbnbs_w_counts = airbnbs.assign(poi_count=poi_count)\
                          .fillna({"poi_count": 0})
```

We can visualise now the distribution of counts to get a sense of how "well-served" AirBnb properties are arranged over space (for good measure, we'll also add a legendgram):

```python
f, ax = plt.subplots(1, figsize=(9, 9))
airbnbs_w_counts.plot(column="poi_count",
                      scheme="quantiles",
                      alpha=0.5,
                      legend=True,
                      ax=ax
                     )
contextily.add_basemap(ax, 
                       crs=airbnbs.crs.to_string(), 
                       source=contextily.providers.Stamen.Toner
                      )
```

---

```python
f, axs = plt.subplots(1, 3, figsize=(18, 6))

airbnbs.plot(ax=axs[0], markersize=0.5)

pois.plot(ax=axs[1], color="green", markersize=0.5)

airbnbs_w_counts.plot(column="poi_count",
                      scheme="quantiles",
                      markersize=0.5,
                      legend=True,
                      ax=axs[2]
                     )

axs[1].set_xlim(axs[0].get_xlim())
axs[1].set_ylim(axs[0].get_ylim())

plt.show()
```

### Assigning point values from surfaces: elevation of AirBnbs


We have just seen how to count points around each observation in a point dataset. In other cases, we might be confronted with a related but different challenge: transfering the value of a particular point in a surface to a point in a different dataset. 

To make this more accessible, let us illustrate the context with an example question: *what is the elevation of each AirBnb property?* To answer it, we require, at least, the following:

1. A sample of AirBnb property locations.
1. A dataset of elevation. We will use here the [NASA DEM](../data/nasadem/README.md) surface for the San Diego area.

Let us bring the elevation surface:

```python
dem = rasterio.open("../data/nasadem/nasadem_sd.tif")
rioshow(dem)
```

Let's first check the CRS is aligned with our sample of point locations:

```python
dem.crs
```

We have opened the file with `rasterio`, which has not read the entire dataset just yet. This feature allows us to use this approach with files that are potentially very large, as only requested data is read into memory.

To extract a discrete set of values from the elevation surface in `dem`, we can use `sample`. For a single location, this is how it works:

```python
list(dem.sample([(-117.24592208862305, 32.761619109301606)]))
```

Now, we can take this logic and apply it to a sequence of coordinates. For that, we need to extract them from the `geometry` object:

```python
abb_xys = pandas.DataFrame({"X": airbnbs.geometry.x, 
                            "Y": airbnbs.geometry.y
                           }).to_records(index=False)
```

```python
elevation = pandas.DataFrame(dem.sample(abb_xys),
                             columns=["Elevation"],
                             index=airbnbs.index
                            )
elevation.head()
```

Now we have a table with the elevation of each  AirBnb property, we can plot it on a map for visual inspection:

```python
f, ax = plt.subplots(1, figsize=(9, 9))
airbnbs.join(elevation)\
       .plot(column="Elevation",
             scheme="quantiles",
             legend=True,
             alpha=0.5,
             ax=ax
            )
contextily.add_basemap(ax, 
                       crs=airbnbs.crs.to_string(), 
                       source=contextily.providers.Stamen.TerrainBackground,
                       alpha=0.5
                      )
```

---

```python
f, axs = plt.subplots(1, 3, figsize=(18, 6))

rioshow(dem, ax=axs[0])

airbnbs.plot(ax=axs[1], markersize=0.5)

airbnbs.join(elevation)\
       .plot(column="Elevation",
             scheme="quantiles",
             markersize=0.5,
             legend=True,
             ax=axs[2]
            )

axs[0].set_xlim(axs[1].get_xlim())
axs[0].set_ylim(axs[1].get_ylim())

plt.show()
```

### distance banding counts & distance-to a secondary feature
- osmnx pois
- flicker data

### Point Interpolation using sklearn 
- (streetscore averaging from nearest sites)
- air quality?

### spatial join, but really don't focus too much on the structure/GIS theory of it
- census data 

### tobler? area to area interpolation

- census geographies vs. h3

### raster engineering to vector features

- Elevation: https://blog.mapbox.com/global-elevation-data-6689f1d0ba65
- DEM from USGS? Public domain? https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1-arc?qt-science_center_objects=0#qt-science_center_objects
- air quality
- night light - served through nasa, using - contextily


## Feature Engineering using Map Synthesis
*Using spatial relationships within a single dataset to synthesize new features for a model.*
### generalize distanceband/buffer counting into a re-explanation of WX models under different weights
### KNN-engineering, adding features by distances
### distance-banding
### eigenvectors feature engineering
### preclustering points into groups for group-based regressions
### use spatially-constrained clustering to build categorical variables for regression


---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.


---

**[To remove eventually]**

But we might want to stuff in a blog post or somewhere else...

```python
mapbox_access_token = ""
demraw, demext = contextily.bounds2img(*airbnbs.to_crs(epsg=3857).total_bounds, zoom=11,
                                       url='https://api.mapbox.com/v4/mapbox.terrain-rgb/'
                                           '{z}/{x}/{y}.pngraw?access_token='+mapbox_access_token) 
```

```python
demraw = demraw.astype(numpy.uint64)
```

```python
plt.imshow(demraw)
```

```python
h,w,b = demraw.shape
```

```python
dem = -1e4 + (demraw[:,:,0]*256**2 + 
              demraw[:,:,1]*256 + 
              demraw[:,:,2])*.1
```

```python
plt.imshow(dem)
```

```python
airbnbs_webmerc_coords = numpy.column_stack((airbnbs.to_crs(epsg=3857).geometry.x, 
                                             airbnbs.to_crs(epsg=3857).geometry.y))
```

```python
dembounds = (demext[0], demext[2], demext[1], demext[3])
```

```python
transform = rasterio.transform.from_bounds(*dembounds, w,h)
```

```python
with rasterio.MemoryFile() as vfile:
    with vfile.open(driver='GTiff', height=h, width=w, count=1, 
                    dtype=dem.dtype, 
                    crs='epsg:3857', transform=transform) as dem_tmp:
        dem_tmp.write(dem, 1)
        elevation_generator = dem_tmp.sample(airbnbs_webmerc_coords)
        elevation_at_airbnb = numpy.row_stack(list(elevation_generator))
```

```python
f,ax = plt.subplots(1,1,figsize=(10,10))
ax = airbnbs.assign(elev = elevation_at_airbnb)\
            .to_crs(epsg=3857).plot('elev', marker='.',ax=ax)
ax.imshow(basemap, extent=basemap_ext)
```
