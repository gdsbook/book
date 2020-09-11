---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: analysis
    language: python
    name: analysis
---

# Geographic feature engineering


In machine learning and data science, we are often equipped with *tons* of data. Indeed, given the constellation of packages to query data services, free and open source data sets, and the rapid and persistent collection of geographical data, there is simply too much data to even represent coherently in a single, tidy fashion. However, we often need to be able to construct useful *features* from this rich and deep sea of data. 

Where data is available, but not yet directly *usable*, *feature engineering* helps to construct useful data for modelling a given phenomenon of interest. In traditional machine learning, feature engineering involves applying additional *domain knowledge* to raw information in order to structure it in a manner that is meaningful for a model. Often, this involves some sort of *transformation* of the original dataset, which is a well-studied concept in both classical statistics and remains so in machine learning methods. While *feature engineering* always relies on this implicit domain knowledge, it is an extremely important part of adapting general-purpose algorithms to unique or distinctive problems facing the every-day data scientist. 

Geography is one of the most high-quality, ubiquitous ways to introduce *domain knowledge* into a problem: everything has a position in both *space* and *time*. And, while things that happen near to one another *in time* do not necessarily have a fundamental relationship, things that are *near* one another are often related. Thus, space is the ultimate *linkage key*, allowing us to connect different datasets together in order to improve our models and our predictions. This means that, even for *aspatial*, "non-geographic" data, you can use *spatial feature engineering* to create useful, highly-relevant features for your analysis. 

At its core, *spatial feature engineering* is the process of developing additional information from raw data using *geographic knowledge*. This distilling of information can occur *between* datasets, where geography is used to link information in separate datasets together; or *within* datasets, where geography can be used to augment the information available for one sample by borrowing from nearby ones. This chapter is structured following that distinction: for cases where geography connects *different* datasets, we adopt the term "Map Matching", often used in industry; while we use the mirroring concept of "Map Synthesis" describing the use of geographical structure to derive new features from a given dataset. Technically speaking, some of the methods we review are similar across these two cases, or even the same; however they can be applied in the context of "matching" or "synthesis", and we consider those conceptually different, hence their inclusion in both sections. Throughout the chapter, we use the Airbnb nightly rental prices in San Diego, as well as auxilliary datasets such as elevation or Census demographics.

```python
import geopandas, pandas, libpysal.weights as weights, contextily
from tobler.area_weighted import area_interpolate
import matplotlib.pyplot as plt
import cenpy
import numpy
import osmnx
import seaborn
import rasterio
from rasterio.plot import show as rioshow
```

Throughout this chapter, we will use a common dataset to which we want to append more information through geography. For the illustration, we will use the set of [AirBnb properties](../data/airbnb/regression_cleaning). Let's read it:

```python
airbnbs = geopandas.read_file('../data/airbnb/regression_db.geojson')
```

# What is spatial feature engineering? 

At its core, *spatial feature engineering* is the process of developing additional information from raw data using *geographic knowledge*. This synthesis could occur *between* datasets, where geography is used to link samples in separate datasets together; or *within* datasets, where geography can be used to borrow information from nearby samples. Building linkages *between* datasets is often called "Map Matching", while we use the term "Map Synthesis" to describe the use of geographical structure to derive new features from existing data. Both kinds of geographic feature engineering will be covered in this chapter, starting first with various methods for Map Matching when modelling Airbnb nightly rental prices in San Diego.

To help us discuss this, a vocabulary is helpful. We will cover a few different kinds of features in this chapter, both of which can be constructed in either Map Synthesis or Map Matching operations: 
- A *spatial summary feature* measures the attributes of observations that have some pre-specified spatial relationship with our target observations. This includes 
    - taking the average or median value of features within a neighborhood of each target observation. 
    - the *spatial lag*, used in previous chapters of this book (e.g. Chapters 3, 6, and 11), is a kind of spatial summary feature, since it reflects the average value of the data in the neighborhood around each point. 
    - Other kinds of spatial summary features might include the count of observations within a given distance or the standard deviation of ten nearest observations. 
    - Summary features generally include *interpolated features* which involve a transfer of information from one spatial support to another, such as when the target locations are not the same as the locations in our anciliary data. Interpolated features become significantly more complex as well when the data is *areal*, as will be discussed later in this chapter.
- A *proximity feature* measures the distance from a target observation to some other observation or position in the map. This might be done in a map matching context, as we did before with the distance to Balboa Park, or it might be done in a map synthesis context by measuring the distance to the nearest other observation. 

## Feature Engineering Using Map Matching

Space is the ultimate linkage key; map matching is all about exploiting this asset. Geographic information is collected at different scales, aggregated using bespoke geographic delineations, and ultimately stored in different datasets. Modelling and machine learning usually require the use of data aligned and coherently structured; to bring disparate geo-datasets together, the data scientist needs to be able to transfer information expressed for one geography into another. This is where map matching comes to rescue. By using geographical location, we can connect datasets that have no common key or that are otherwise completely unrelated. This is a "magic trick" that we can pull off because of location that would be much harder, or impossible, with data of other nature.

The details, and difficulty, of this transfer of information from one geography to another one depend on the nature of the "source" and "target", and on the precision with which we want to perform such transfer. In this context, there is always an inevitable trade-off between precission and sophistication: more precise transfer is usually possible at the expense of more advanced and involved techniques. Their additional cost in applying them, both in terms of cognitive load on the data scientist's part or in terms of computation, must be weighted in on an individual basis: sometimes we need the best possible estimates, at all costs; sometimes "good enough" is, well, good enough. 

In this section, we cover a few cases that we consider represent the most common and widely used approaches. We begin with situations where we have two point datasets and all we need to calculate are descriptive statistics of one dataset to the observations in the other. We swith the type of data and consider how to attach information from a continuous grid, stored in a raster file, to a set of points. Then we show the equivalent case for "moving" information for a polygon geography to a point dataset. These are all cases that, in their simplest form, involve mostly traditional GIS operations (e.g. buffer construction, spatial joins) and little in the way of statistical modelling. As an example of a case that is more involved, we cover the transfer of information from a polygon geography to another, different polygon geography. For this final case, but also as a more general comment, we try to include examples that capture the essence of the method, but keep the technique as simple as possible. For almost any of these cases we cover, the reader can find more sophisticated techniques that usually yield more accurate estimates. Where possible, we try to signpost these.

### Counting *nearby* features


A first, conceptually straightforward, approach is to augment our dataset by counting how many points of a different dataset are in the vicinity of each observation. For example, we might want to know how many bars and restaurants each AirBnb has within a given radious. This count can then become an additional  feature of our dataset, stored in a new column of `airbnbs`.

To obtain information on the location of restaurants and bars, we can download it from OpenStreetMap directly using `osmnx`. We first query all the points of interest (POIs) within the area our points cover, and then filter out everything except restaurants and bars. For that, we require to get a polygon that covers all our `airbnbs` points. From Chapter 8, we can recall that there are a few different hulls that can be used. We'll use the Convex Hull here, which is the smallest convex polygon that covers all of the points in the set. 

```python
airbnbs_ch = airbnbs.unary_union.convex_hull
airbnbs_ch
```

Using this polygon, we can use the `osmnx` package to fetch points of interest (POIs) from OpenStreetMap. We can make our request more manageable by only requesting points of interest that fall within specific categories. Below, we'll request POIs within San Diego that are "restaurants" or "bars," according to their metadata stored in OpenStreetMap. (*note: this step requires internet connection as it is querying a remote server*):

```python
%%time
pois = osmnx.pois_from_polygon(airbnbs_ch,
                               tags={"amenity": ['restaurant', 'bar']}
                              )
```

This provides us with every location within our convex hull that is tagged as a "restaurant" or "bar" its metadata on OpenStreteMap. Overall, this provides us with about 1300 points of interest: 

```python
pois.groupby('amenity').amenity.count()
```

Once loaded into `pois` as a `GeoDataFrame`, let's take a peek at their location, as compared with AirBnb spots:

```python
f,ax = plt.subplots(1,figsize=(12, 12))
airbnbs.plot(ax=ax, marker='.')
pois.plot(ax=ax, color='r')
contextily.add_basemap(ax, 
                       crs=airbnbs.crs.to_string(), 
                       source=contextily.providers.Stamen.Toner
                      )
```

Now, for some feature engineering, it may be extremely useful to know whether an Airbnb is located in a "hot" location, with a lot of restaurants and bars to choose from. Alternatively, if Airbnbs are very remote, they might not be as lucrative for short, expensive "city-break" reservations. That is, Airbnb users may decide to reserve stays where there are a lot of dining and drinking opportunities, and thus may be *willing to pay more* for the same accommodation. We might be able to predict prices better if we know about the drinking and dining scene near the Airbnb. 

Thus, we can *engineer features* in the Airbnb data using the nearby POIs. To do this, we can create a new feature for the AirBnb dataset --that is, a new column in `airbnbs`-- which incorporates information about how many POIs are *nearby* each property. This kind of "feature counting" is useful in applications where the mere presence of nearby features can affect the quantity we are modelling. 

To do this kind of feature engineering, let us first walk through what we need to do at a conceptual level: 

1. Decide what is *nearby*. This will dictate how far we go from each AirBnb when counting the number of "nearby" bars & restaurants. For this example, we will use 500 meter buffer, which is approximately the distance of a leisurely ten-minute walk.
2. For each AirBnb, determine whether POIs are *within* a leisurely 10-minute walk. 
3. Count how many POIs are withing the specified radius of each AirBnb.

At the end of this procedure, we have the number of bars & restuarants that are within a leisurely walk of the AirBnb, which might be useful in predicting the price of each AirBnb. 

With this, let us now translate the list above into code. For part 1., we need to be able to measure distances in metres. However, `airbnbs` is originally expressed in degrees, since it is provided in terms of locations in latitude and longitude:

```python
airbnbs.crs
```

In addition, the `pois` are also provided in terms of their latitude & longitude:

```python
pois.crs
```

Therefore, we need to convert this into a coordinate system that is easier to work with. Here, we will use a projection common for mapping in California, the California Albers projection:

```python
airbnbs_albers = airbnbs.to_crs(epsg=3311)
pois_albers = pois.to_crs(epsg=3311)
```

```python
pois_albers.crs
```

With this, we can create the radius of 500m around each AirBnb. This is often called *buffering*, where a shape is dilated by a given radius.

```python
airbnbs_albers['buffer_500m'] = airbnbs_albers.buffer(500)
```

Now, `abb_buffer` contains a 500-meter circle around each Airbnb.

Using these, we can count the number of POIs that are within these areas using a *spatial join*. Spatial joins link geometries based on spatial relationships (or predicates). Here, we need to know the relationship: `pois within airbnb_buffers`, where `within` is the predicate relating `pois` to `airbnb_buffers`. Predicates are not always *reversible*: no `airbnb_buffer` can be `within` a `poi`. In `geopandas`, we can compute all pairs of relations between the `pois` and `airbnb_buffers` efficiently using the `sjoin` function, which takes a `predicate` argument defining the requested relationship between the first & second argument. 

```python
joined = geopandas.sjoin(pois_albers,
                    airbnbs_albers.set_geometry('buffer_500m')[['id', 'buffer_500m']],
                    op="within"
                   )
```

The resulting joined object `joined` contains a row for every pair of POI and AirBnb that are linked. From there, we can apply a group-by operation, using the AirBnb ID, and count how many POIs each AirBnb has within 500m of distance:

```python
poi_count = joined.groupby("id")\
                  ["osmid"]\
                  .count()\
                  .to_frame('poi_count')
poi_count.head()
```

The resulting `Series` is indexed on the AirBnb IDs, so we can assign it to the original `airbnbs` table. In this case, we know by construction that missing AirBnbs in `poi_count` do not have any POI within 500m, so we can fill missing values in the column with zeros.

```python
airbnbs_w_counts = airbnbs_albers.merge(poi_count, left_on='id', right_index=True)\
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
                       crs=airbnbs_albers.crs.to_string(), 
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

### Point Interpolation using sklearn 

In the previous example, we have transfered information from a surface (stored in a raster layer) to a set of points; in other words, we have gone from surface to points. Sometimes, however, we do not have the luxury of a ready-made surface. Instead, all we have available is a set of points with measurements for the variable of interest that do not match the points we want the information for. In this situation, a solution we can rely on is "spatial interpolation". 

- (streetscore averaging from nearest sites)
- air quality?



### Polygon to point

We now move on to a case where the information we are interested in matching to our set of points is stored for a polygon geography. For example, we would like to know the population density of the neighborhood in which each AirBnb is located. To that, we will download population estimates at the Census tract level, and "transfer" those estimates over to each AirBnb point. Geographically, the only challenge here is finding within which tract every point falls, and the performing what is spatial databases parlance is called a "spatial join", by which we connect the two layers through their spatial connection.

Let us pull down the number of inhabitants from the American Community Survey for tracts in San Diego:

```python
%%time
acs = cenpy.products.ACS()
sd_pop = acs.from_msa("San Diego, CA",
                      level = "tract",
                      variables=['B02001_001E']
                     )
```

And calculate population density:

```python
sd_pop["density"] = sd_pop["B02001_001E"] / sd_pop.to_crs(epsg=3311).area
```

Now, to "transfer" density estimates to each AirBnb, we can rely on the spatial join in `geopandas`:

```python
j = geopandas.sjoin(airbnbs, sd_pop.to_crs(airbnbs.crs))
```

The result is a table with one row per AirBnb and one column for each attribute we originally had for properties, *as well as* those of the tract where the area is located:

```python
j.info()
```

---

```python
f, axs = plt.subplots(1, 3, figsize=(18, 6))

sd_pop.plot(column="density", scheme="quantiles", ax=axs[0])

airbnbs.plot(ax=axs[1], markersize=0.5)

j.plot(column="density",
       scheme="quantiles",
       markersize=0.5,
       ax=axs[2]
      )

plt.show()
```

### Area to area interpolation

The final case of map matchin we consider is transfer of information from one polygon/areal geography to a different one. This is a common use-case when an analysis requires data that is provided at different levels of aggregation and different boundary delineations.

There is a large literature around this problem under the umbrella of dasymetric mapping (REF). The conceptual idea is relatively straight-forward: we want to apportion values from one set of polygons to the other based on how much "geography" is shared. In its simplest case, we can do this based on area. In this case, we will assign values from the source geography to the target based on how much they share. Let us illustrate this with an example. We will call the geography for which we have data the "source", and that to which we want to transfer data the "target". If polygon A in the target is made up of 50% of polygon 1 in the source, 30% of polygon 2, and 20% of polygon 3, the estimate for A will be a weighted average between the values in 1, 2, and 3, where the weights are 0.5, 0.3, and 0.2, respectively. Of course, underlying this exercise is the implicit assumption that the values we are interested in are uniformly distributed within each polygon in the source and target. In some cases, this is a valid assumption or, at least, it does not introduce critical errors; in others, this is not acceptable. Dasymetric mapping has proposed a large amount of sophistications that try to come up with more realistic estimates and that can incorporate additional information.

To implement dasymetric mapping in Python, the best option is `tobler`, a package from the PySAL federation designed exactly for this goal. We will show here the simplest case, that of areal interpolation where apportioning is estimated based on area, but the package provides also more sophisticated approaches.

For the example, we need two polygon layers. We will stick with San Diego and use the set of Census Tracts and the [H3 hexagonal grid layer](../data/h3_grid/README.md). Our goal will be to create population estimates for each hexagon.

First, let us load the H3 grid:

```python
h3 = geopandas.read_file("../data/h3_grid/sd_h3_grid.gpkg")
```

We are ready to interpolate:

```python
interpolated = area_interpolate(source_df=sd_pop.to_crs(epsg=3311),
                                target_df=h3.to_crs(epsg=3311),
                                extensive_variables=["B02001_001E"],
                                intensive_variables=["density"]
                               )
```

There is quite a bit going on in the cell above, let us unpack it:

- Remember this method apportions data values based on area, so it makes sense to have an accurate estimate for the extent of each polygon. To do that, we convert each geography to Albers Equal (`EPSG:3311`), which is expressed in metres, using `to_crs`.
- The method `area_interpolate` then takes the source and the target `GeoDataFrame` objects using the same naming convention we have in our explanation. 
- In addition, we need to specify which variables we would like to interpolate. here, Tobler makes a distinction:
    - *Extensive* variables, or absolute values such as counts, aggregates, etc. (which we use for population, `B02001_001E`)
    - *Intensive* variables, such as rates, ratios, etc. (which we select for density as it is the ratio of population over area)

A good first approach to examine the output is by comparing the source and the target visually. Here is total population:

```python
f, axs = plt.subplots(1, 3, figsize=(18, 6))

minX, minY, maxX, maxY = interpolated.total_bounds
sd_pop.to_crs(epsg=3311)\
      .cx[minX:maxX, minY:maxY]\
      .plot(column="B02001_001E", 
            scheme="quantiles", 
            k=10,
            ax=axs[0]
           )

h3.to_crs(epsg=3311)\
  .plot(ax=axs[1], markersize=0.5)

interpolated.plot(column="B02001_001E",
                  scheme="quantiles",
                  k=10,
                  ax=axs[2]
                 )

axs[0].set_xlim(minX, maxX)
axs[0].set_ylim(minY, maxY)

f.suptitle("Population")

plt.show()
```

And density:

```python
f, axs = plt.subplots(1, 3, figsize=(18, 6))

minX, minY, maxX, maxY = interpolated.total_bounds
sd_pop.to_crs(epsg=3311)\
      .cx[minX:maxX, minY:maxY]\
      .plot(column="density", 
            scheme="quantiles", 
            k=10,
            ax=axs[0]
           )

h3.to_crs(epsg=3311)\
  .plot(ax=axs[1], markersize=0.5)

interpolated.plot(column="density",
                  scheme="quantiles",
                  k=10,
                  ax=axs[2]
                 )

axs[0].set_xlim(minX, maxX)
axs[0].set_ylim(minY, maxY)

f.suptitle("Population Density")

plt.show()
```

## Feature Engineering using Map Synthesis

Feature engineering with map matching is most helpful when you have additional information to use in the analysis. And, with the wealth of freely available data from censuses, satellites, and open volunteered geographic information vendors such as OpenStreetMap, map matching can be a very powerful tool for enriching and improving your analyses. However, it is sometimes *also* useful to only examine the data you've got, and use the spatial structure within to build better features or better models. While this might be done using spatially-explicit models (like those covered in Chapter 11), it is also possible to use map synthesis to build spatial feature and improve your predictions.

There is an extensive amount of map synthesis features. In addition to the two kinds of features discussed in map matching, there are other kinds of useful features that can be used in map synthesis that will be discussed below. First, we will return to spatial summary features. Second, we will examine some *regionalization features*, which detect and leverage geographical clusters in the data to improve prediction. 

### Spatial Summary Features in Map Synthesis


Just like in map matching, you can use spatial summary features in map synthesis to make better predictions. One clear method involves constructing spatial summary measures of your training data. This is done in the same manner as in map matching, except we can now refer only to the data on hand. Thus, we may want to determine whether nearby Airbnbs are "competing" with each airbnb. We might do this by finding the distance to the nearest Airbnb with the same number of bedrooms, since two nearby listings that *also* sleep the same number of people likely will compete with one another for tenants. 

We might do this by building a `DistanceBand` weight object, which considers Airbnb as "neighbors" if they are within the distance threshold. 

```python
d500_w = weights.DistanceBand.from_dataframe(airbnbs_albers, threshold=500, silence_warnings=True)
```

Now, we can get the average size of surrounding Airbnbs directly as the spatial lag:

```python
d500_w.transform = 'r'
```

```python
local_average_bedrooms = weights.lag_spatial(d500_w, airbnbs_albers[['bedrooms']].values)
```

While related, these features contain quite distinct pieces of information, and both may prove useful in modelling: 

```python
plt.scatter(airbnbs_albers[['bedrooms']].values, local_average_bedrooms)
plt.xlabel("Number of bedrooms")
plt.ylabel("Average of nearby\n listings' bedrooms")
```

If we were instead interested in the most common number of bedrooms, rather than the average, we could use the `lag_categorical` function:

```python
local_mode = weights.lag_categorical(d500_w, airbnbs_albers[['bedrooms']].values)
```

Since we are now treating these features as discrete, it helps to use `pandas.crosstab` to visualize their distribution:

```python
crosstab = pandas.crosstab(airbnbs_albers.bedrooms, 
                           local_mode.flatten())
crosstab.columns.name = "nearby"
crosstab
```

If more complicated statistics are required. it can help to re-express the construction of summary statistics as a *reduction* of the *adjacency list* representation of our weights, as done in Chapter 3. For instance, constructing the local median involves building the adjacency list:

```python
adjlist = d500_w.to_adjlist()
```

```python
adjlist.head()
```

merging in the relevant information:

```python
adjlist = adjlist.merge(airbnbs_albers[['bedrooms']], left_on='neighbor', right_index=True, how='left')
```

and then grouping by the `focal` and summarizing:

```python
adjlist.groupby("focal").bedrooms.median()
```

In most cases, however, `lag_spatial` and `lag_categorical` can construct most of the common features used in day-to-day analysis. 

Sometimes, analysts might build multiple "bands" of features by increasing the `threshold` in a `DistanceBand` or the `k` in a K-nearest neighbors application. That is to build another average of the area within 1 kilometer, you can repeat the above analysis with a different threshold:

```python
average_within_500 = weights.lag_spatial(d500_w, airbnbs_albers[['bedrooms']].values)

d1k_w = weights.DistanceBand.from_dataframe(airbnbs_albers, threshold=1000, silence_warnings=True)
d1k_w.transform = 'r'
average_within_1km = weights.lag_spatial(d1k_w, airbnbs_albers[['bedrooms']].values)
```

In some cases, an *exclusive* average is preferred. That is, we may want the 1 kilometer average to include only the data from 500m to 1km and *disregard* the values before it. 

While this is somewhat more challenging, it can be done efficiently using the `weights.set_operations` module:

```python
d1k_no_500_w = weights.set_operations.w_difference(d1k_w, d500_w)
```

```python
average_between = weights.lag_spatial(d1k_no_500_w, airbnbs_albers[['bedrooms']].values)
```

```python

```

### preclustering points into groups for group-based regressions
### use spatially-constrained clustering to build categorical variables for regression


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
