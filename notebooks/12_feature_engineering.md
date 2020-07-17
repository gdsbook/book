---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.4
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
```

Throughout this chapter, we will use a common dataset to which we want to append more information through geography. For the illustration, we will use the set of [AirBnb properties](../data/airbnb/regression_cleaning). Let's read it:

```python
airbnbs = geopandas.read_file('../data/airbnb/regression_db.geojson')
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
                               amenities=['restaurant', 'bar']
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

`DAB` stop point

---


Once loaded into `pois` as a `GeoDataFrame`, let's take a peak at their location, as compared with AirBnb spots:

```python jupyter={"outputs_hidden": true}
f,ax = plt.subplots(1,figsize=(12, 12))
airbnbs.plot(ax=ax, marker='.')
pois.plot(ax=ax, color='r')
contextily.add_basemap(ax, 
                       crs=airbnbs.crs.to_string(), 
                       source=contextily.providers.Stamen.Toner
                      )
```

To connect AirBnb properties to nearby restaurant and bars

```python
pois_per_listing = geopandas.sjoin(airbnbs.set_geometry(airbnbs.buffer(.01)), 
                                   pois, op='intersects').groupby('id').id.count()
```

```python
airbnbs.merge(pois_per_listing.to_frame('n_pois'), left_on='id', right_index=True).plot('n_pois')
```

# DEM sampling for a regression covariate



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
