---
redirect_from:
  - "/notebooks/10-clustering-and-regionalization"
interact_link: content/notebooks/10_clustering_and_regionalization.ipynb
kernel_name: python3
has_widgets: false
title: 'Clustering & Regionalization'
prev_page:
  url: /notebooks/09_spatial_inequality
  title: 'Spatial Inequality'
next_page:
  url: /notebooks/11_regression
  title: 'Regression'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


# Clustering and Regionalization
<!--
**NOTE**: parts of this notebook have been
borrowed from [GDS'17 - Lab
6](http://darribas.org/gds17/content/labs/lab_06.html)
-->



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from pysal.explore.esda.moran import Moran
from pysal.lib.api import Queen, KNN
from pysal.lib.weights import Wsets
from booktools import choropleth
import seaborn 
import pandas
import geopandas 
import data
import numpy
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt


```
</div>

</div>



## Introduction

The world's hardest questions are complex and multi-faceted.
Effective methods to learn from data should recognize this. Many questions
and challenges are inherently multidimensional; they are affected, shaped, and
defined by many different components all acting simultaneously. In statistical
terms, these processes are called *multivariate processes*, as opposed to 
*univariate processes*, where only a single variable acts at once.
Clustering is a fundamental method of geographical analysis that draws insights
from large, complex multivariate processes. It works by finding similarities among the
many dimensions in a multivariate process, condensing them down into a simpler representation
Thus, through clustering, a complex and difficult to understand process is recast into a simpler one
that even non-technical audiences can look at and understand. 

Often, clustering involves sorting observations into groups. For these groups to be more
meaningful than any single initial dimension, members of a group should be more
similar to one another than they are to members of a different group.
Each group is referred to as a *cluster* while the process of assigning
objects to groups is known as *clustering*. If done well, these clusters can be
characterized by their *profile*, a simple summary of what members of a group
are like in terms of the original multivariate process.

Since a good cluster is more
similar internally than it is to any other cluster, these cluster-level profiles
provide a convenient shorthand to describe the original complex multivariate process.
Observations in one group may have consistently high 
scores on some traits but low scores on others. 
The analyst only needs to look at the profile of a cluster in order to get a
good sense of what all the observations in that cluster are like, instead of
having to consider all of the complexities of the original multivariate process at once. 
Throughout data science, and particularly in geographic data science, 
clustering is widely used to provide insights on the
geographic structure of complex multivariate spatial data. 

In the context of explicitly spatial questions, a related concept, the *region*,
is also instrumental. A *region* is similar to a *cluster*, in the sense that
all members of a region have been grouped together, and the region should provide 
a shorthand for the original data. 
Further, for a region to be analytically useful, its members also should
display stronger similarity to each other than they do to the members of other regions. 
However, regions are more complex than clusters because they combine this
similarity in profile with additional information about the geography of their members.
In short, regions are like clusters (since they have a coherent profile), but they
also have a coherent geography&mdash;members of a region should also be
located near one another.

The process of creating regions is called regionalization.
A regionalization is a special kind of clustering where the objective is 
to group observations which are similar in their statistical attributes,
but also in their spatial location. In this sense, regionalization embeds the same
logic as standard clustering techniques, but also applies a series of
spatial and/or geographical constraints. Often, these
constraints relate to connectivity: two candidates can only be grouped together in the
same region if there exists a path from one member to another member
that never leaves the region. These paths often model the spatial relationships
in the data, such as contiguity or proximity. However, connectivity does not
always need to hold for all regions, and in certain contexts it makes
sense to relax connectivity or to impose different types of spatial constraints. 

In this chapter we consider clustering techniques and regionalization methods which will
allow us to do exactly that. In the process, we will explore the characteristics
of neighborhoods in San Diego.
We will extract common patterns from the
cloud of multidimensional data that the Census Bureau produces about small areas
through the American Community Survey. We begin with an exploration of the
multivariate data about San Diego by suggesting some ways to examine the 
statistical and spatial distribution of the data before carrying out any
 clustering. Focusing on the individual variables, as well as their pairwise
associations, can help guide the subsequent application of clusterings or regionalizations. 

We then consider geodemographic approaches to clustering&mdash;the application
of multivariate clustering to spatially referenced demographic data.
Two popular clustering algorithms are employed: k-means and Ward's hierarchical method.
Mapping the spatial distribution of the resulting clusters 
reveals interesting insights on the socioeconomic structure of the San Diego
metropolitan area. We also see that in many cases, clusters are spatially 
fragmented. That is, a cluster may actually consist of different areas that are not
spatially connected. Indeed, some clusters will have their members strewn all over the map. 
This will illustrate why connectivity might be important when building insight
about spatial data, since these clusters will not at all provide intelligible regions. 
So, we then will move on to regionalization, exploring different approaches that
incorporate geographical constraints into the exploration of the social structure of San Diego.

## Data

The dataset we will use in this chapter comes from the American Community Survey
(ACS). In particular, we examine data at the Census Tract level in San Diego,
California in 2016. Let us begin by reading in the data as a GeoDataFrame and
exploring the attribute names.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Read file
db = geopandas.read_file(data.san_diego_tracts())
# Print column names
db.columns

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Index(['GEOID', 'GEOID_x', 'Total Popu', 'White', 'Black', 'Hispanic',
       'Med Age', 'Pop < 18', 'Total Hous', 'In househo', 'Male house',
       'Female hou', 'High Sch D', 'GED', 'Bachelor's', 'Travel tim',
       'Income bel', 'Income Mal', ' Income Fe', 'Median HH', 'Cash publi',
       'Lowest qui', 'Second qui', 'Third  qui', 'Fourth  qu', 'Lower limi',
       'Gini index', 'Total Empl', 'In Labor F', 'Civilian L', 'Employed',
       'Total Ho_1', 'Vacant', 'Owner Occu', 'Renter Occ', 'Median Num',
       'Median Str', 'Median Gro', 'Median Val', 'state', 'county', 'tract',
       'AREALAND', 'AREAWATER', 'BASENAME', 'CENTLAT', 'CENTLON', 'COUNTY_1',
       'FUNCSTAT', 'GEOID_y', 'INTPTLAT', 'INTPTLON', 'LSADC', 'MTFCC', 'NAME',
       'OBJECTID', 'OID', 'STATE_1', 'TRACT_1', 'renter_pct', 'geometry'],
      dtype='object')
```


</div>
</div>
</div>



While the ACS comes with a large number of attributes we can use for clustering
and regionalization, we are not limited to the original variables at hand; we
can construct additional variables. This is particularly useful when
we want to compare areas that are not very similar in some structural
characteristic, such as area or population. For example, a quick look into the
variable names shows most variables are counts. For tracts of different sizes,
these variables will mainly reflect their overall population, rather than provide direct information
about the variables itself. To get around this, we will cast many of these count variables to rates,
and use them in addition to a subset of the original variables. 
Together, this set of constructed and received variables will to
will be used for our clustering and regionalization.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Pull out total house units
total_units = db['Total Ho_1']
# Calculate percentage of renter occupied units
pct_rental = db['Renter Occ'] / (total_units + (total_units==0)*1)

# Pull out total number of households
total_hh = db['Total Hous']
# Calculate percentage of female households
pct_female_hh = db['Female hou'] / (total_hh + (total_hh==0)*1)

# Calculate percentage of population with a bachelor degree
pct_bachelor = db["Bachelor's"] / (db['Total Popu'] + (db['Total Popu']==0)*1)
# Assign newly created variables to main table `db`
db['pct_rental'] = pct_rental
db['pct_female_hh'] = pct_female_hh
db['pct_bachelor'] = pct_bachelor
# Calculate percentage of population white
db['pct_white'] = db["White"] / (db['Total Popu'] + (db['Total Popu']==0) * 1)

```
</div>

</div>



To make things easier later on, let us collect the variables we will use to
characterize Census tracts. These variables capture different aspects of the socio-
economic reality of each area and, taken together, they provide a comprehensive
characterization of San Diego as a whole:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
cluster_variables =  ['Median Val',   # Median house value
                      'pct_white',    # Percent of tract population that is white
                      'pct_rental',   # Percent of households that are rented
                      'pct_female_hh',# Percent of female-led households 
                      'pct_bachelor', # Percent of tract population with a Bachelors degree
                      'Median Num',   # Median number of rooms in the tract's households
                      'Gini index',   # Gini index measuring tract wealth inequality
                      'Med Age',      # Median age of tract population
                      'Travel tim'    # ???
                      ]

```
</div>

</div>



### Exploring the data

Now let's start building up our understanding of this
dataset through both visual and summary statistical measures.

We will start by
looking at the spatial distribution of each variable alone.
This will help us draw a picture of the multi-faceted view of the tracts we
want to capture with our clustering. Let's use choropleth maps for the
nine attributes and compare these choropleth maps side-by-side:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
f, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
# Make the axes accessible with single indexing
axs = axs.flatten()
# Start a loop over all the variables of interest
for i, col in enumerate(cluster_variables):
    # select the axis where the map will go
    ax = axs[i]
    # Plot the map
    #db.plot(column=col, ax=ax, scheme='Quantiles', 
    #        linewidth=0, cmap='RdPu')
    choropleth(db, col, cmap='RdPu', ax=ax)
    # Remove axis clutter
    ax.set_axis_off()
    # Set the axis title to the name of variable being plotted
    ax.set_title(col)
# Display the figure
plt.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_9_0.png)

</div>
</div>
</div>



Many visual patterns jump out from the maps, revealing both commonalities as
well as differences across the spatial distributions of the individual variables.
Several variables tend to increase in value from the east to the west
(`pct_rental`, `Median Val`, `Median Num`, and `Travel tim`) while others
have a spatial trend in the opposite direction (`pct_white`, `pct_female_hh`,
`pct_bachelor`, `Med Age`). This is actually desirable; when variables have
different spatial distributions, each variable to contributes distinct 
information to the profiles of each cluster. However, if all variables display very similar 
spatial patterns, the amount of useful information across the maps is 
actually smaller than it appears, so cluster profiles may be much less useful as well.
It is also important to consider whether the variables display any
spatial autocorrelation, as this will affect the spatial structure of the
resulting clusters. 

Recall from chapter XXX that Moran's I is a commonly used
measure for global spatial autocorrelation. 
Let us get a quick sense to what
extent this is present in our dataset.
First, we need to build a spatial weights matrix that encodes the spatial
relationships in our San Diego data. We will start with queen contiguity:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
w_queen = Queen.from_dataframe(db)

```
</div>

</div>



As the warning tells us, observation `103` is an *island*, a disconnected observation
with no queen contiguity neighbors. To make sure that every observation
has at least one neighbor, we can combine the queen contiguity matrix with a
nearest neighbor matrix. This would ensure that every observation is neighbor 
of at least the observation it is closest to, plus all the areas with which 
it shares any border. Let's first create the `KNN-1 W`:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
w_k1 = KNN.from_dataframe(db, k=1)

```
</div>

</div>



Now we can combine the queen and nearest neighbor matrices into a single representation
with no disconnected observations. This full-connected connectivity matrix is the 
one we will use for analysis:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
w = Wsets.w_union(w_queen, w_k1)

```
</div>

</div>



As we ensured (thanks to the nearest neighbor connections),  `w` does not contain
any islands:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
w.islands

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
[]
```


</div>
</div>
</div>



Now let's calculate Moran's I for the variables being used. This will measure
the extent to which each variable contains spatial structure:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Set seed for reproducibility
numpy.random.seed(123456)
# Calculate Moran's I for each variable
mi_results = [Moran(db[variable], w) for variable in cluster_variables]
# Display on table
table = pandas.DataFrame([(variable, res.I, res.p_sim) for variable,res 
                      in zip(cluster_variables, mi_results)],
                     columns=['Variable', "Moran's I", 'P-value'])\
          .set_index('Variable')
table

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
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
      <th>Moran's I</th>
      <th>P-value</th>
    </tr>
    <tr>
      <th>Variable</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Median Val</th>
      <td>0.664557</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>pct_white</th>
      <td>0.691054</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>pct_rental</th>
      <td>0.465938</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>pct_female_hh</th>
      <td>0.297972</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>pct_bachelor</th>
      <td>0.421799</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>Median Num</th>
      <td>0.554416</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>Gini index</th>
      <td>0.293869</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>Med Age</th>
      <td>0.399902</td>
      <td>0.001</td>
    </tr>
    <tr>
      <th>Travel tim</th>
      <td>0.096992</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



Each of the variables displays significant positive spatial autocorrelation,
suggesting that Tobler's law is alive and well in the socioeconomic geography of San
Diego County. This means we also should expect the clusters we find will have
a non random spatial distribution. In particular, we would expect clusters to have
a modest amount of spatial coherence in addition to the coherence in their profiles,
since there is strong positive autocorrelation in all of the input variables.

Spatial autocorrelation only describes relationships between a single observation at a time.
So, the fact that all of the clustering variables are positively autocorrelated does not tell us 
about the way the attributes covary over space. For that, we need to consider the
spatial correlation between variables. Here, we will measure this using the
bivariate correlation in the maps of covariates themselves.

Given the 9 maps, there are 36 pairs of maps that must be compared. This is too 
many maps to process visually, so we can turn to an alternative tool to
explicitly focus on the bivariate relations between each pair of attributes.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = seaborn.pairplot(db[cluster_variables], kind='reg', diag_kind='kde')

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_21_0.png)

</div>
</div>
</div>



Two different types of plots are contained in the scatterplot matrix. On the
diagonal are the density functions for the nine attributes. These allow for an
inspection of the overall morphology of the attribute's value distribution.
Examining these we see that our selection of variables includes those that are
negatively skewed (`pct_white` and `pct_female_hh`) as well as positively skewed
(`while median_val`, `pct_bachelor`, and `travel_tim`).

The second type of visualization lies in the off-diagonal cells of the matrix; 
these are bi-variate scatterplots. Each cell shows the association between one
pair of variables. Several of these cells indicate positive linear
associations (`med_age` Vs. `median_value`, `median_value` Vs. `Median Num`)
while other cells display negative correlation (`Median Val` Vs. `pct_rental`,
`Median Num` Vs. `pct_rental`, and `Med Age` Vs. `pct_rental`). The one variable
that tends to have consistenty weak association with the other variables is
`Travel tim`, and in part this appears to reflect its rather concentrated 
distribution as seen on the lower right diagonal corner cell.

## Geodemographic Clusters in San Diego Census Tracts

We now will move
beyond the implicitly bi-variate focus to consider the full multidimensional
nature of this data set. Geodemographic analysis is a form of multivariate
clustering where the observations represent geographical areas. The output
of these clusterings is nearly always mapped. Altogether, these methods use
multivariate clustering algorithms to construct a known number of
clusters ($k$), where the number of clusters is typically much smaller than the 
number of observations to be clustered. Each cluster is given a unique label,
and these labels are mapped. Using the clusters' profile and label, the map of 
labels can be interpreted to get a sense of the spatial distribution of 
sociodemographic traits. The power of (geodemographic) clustering comes
from taking statistical variation across several dimensions and compressing it
into a single categorical one that we can visualize through a map. To
demonstrate the variety of approaches in clustering, we will show two
distinct but very popular clustering algorithms: k-means and Ward's hierarchical method.

### K-means

K-means is probably the most widely used approach to
cluster a dataset. The algorithm groups observations into a
prespecified number of clusters so that that each observation is
closer to the mean of its own cluster than it is to the mean of any other cluster.
The k-means problem is solved by iterating between an assignment step and an update step. 
First, all observations are randomly assigned one of the $k$ labels. Next, the 
multivariate mean over all covariates is calculated for each of the clusters.
Then, each observation is reassigned to the cluster with the closest mean. 
If the observation is already assigned to the cluster whose mean it is closest to,
the observation remains in that cluster. This assignment-update process continues
until no further reassignments are necessary.

The nature of this algorithm requires us to select the number of clusters we 
want to create. The right number of clusters is unknown in practice. For
illustration, we will use $k=5$ in the `KMeans` implementation from
`scikit-learn`. To proceed, we first create a `KMeans` clusterer:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Initialise KMeans instance
kmeans = KMeans(n_clusters=5)

```
</div>

</div>



Next, we call the `fit` method to actually apply the k-means algorithm to our data:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Set the seed for reproducibility
numpy.random.seed(1234)
# Run K-Means algorithm
k5cls = kmeans.fit(db[cluster_variables])

```
</div>

</div>



Now that the clusters have been assigned, we can examine the label vector, which 
records the cluster to which each observation is assigned:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
k5cls.labels_

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
array([3, 4, 4, 0, 0, 4, 0, 0, 0, 2, 0, 2, 2, 2, 4, 0, 2, 2, 2, 4, 4, 4,
       0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 0,
       0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0,
       4, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2,
       2, 2, 4, 4, 4, 0, 0, 0, 0, 2, 2, 0, 2, 4, 4, 0, 2, 0, 0, 0, 4, 0,
       0, 0, 0, 0, 2, 0, 4, 4, 3, 3, 4, 4, 0, 0, 4, 4, 4, 4, 4, 2, 4, 0,
       4, 4, 4, 4, 4, 3, 1, 3, 4, 1, 0, 4, 4, 3, 1, 1, 3, 3, 4, 3, 0, 4,
       4, 3, 4, 4, 4, 0, 4, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
       2, 0, 2, 2, 2, 0, 2, 4, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 2, 0, 0,
       2, 2, 0, 0, 2, 2, 0, 4, 0, 4, 4, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3,
       4, 0, 4, 4, 0, 0, 0, 2, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 2,
       3, 3, 1, 3, 3, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0,
       0, 2, 0, 0, 0, 0, 0, 4, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0,
       0, 4, 0, 0, 0, 0, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0,
       0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 4, 2, 0, 4, 0, 0, 4, 2, 0, 2, 0, 2,
       2, 2, 2, 2, 0, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0,
       2, 0, 0, 0, 0, 2, 2, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 0, 4, 0, 0, 0,
       0, 2, 4, 4, 4, 0, 3, 4, 0, 4, 4, 0, 4, 0, 4, 0, 0, 0, 4, 4, 4, 4,
       4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 3, 1, 3, 4, 3, 3,
       3, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 0, 4, 0,
       4, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 0, 2, 2,
       0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 4, 0, 2,
       0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0,
       4, 4, 0, 4, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 4, 2, 2, 0, 2, 2, 0, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 4, 2, 0, 4, 2, 2,
       2, 0, 4, 2, 0, 0, 4, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 2, 2, 2, 4, 0,
       0, 0, 4, 4, 4, 3, 3, 1, 2, 2, 4], dtype=int32)
```


</div>
</div>
</div>



In this case, the second and third observations are assigned to cluster 4, while
the fourth and fifth observations have been placed in cluster 0. It is important
to note that the integer labels should be viewed as denoting membership only &mdash;
the numerical differences between the values for the labels are meaningless.
The profiles of the various clusters must be further explored by looking
at the values of each dimension. 

But, before we do that, let's make a map.

### Spatial Distribution of Clusters

Having obtained the cluster labels, we can display the spatial
distribution of the clusters by using the labels as the categories in a
choropleth map. This allows us to quickly grasp any sort of spatial pattern the 
clusters might have. Since clusters represent areas with similar
characteristics, mapping their labels allows to see to what extent similar areas tend
to have similar locations.
Thus, this gives us one map that incorporates the information of from all nine covariates.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Assign labels into a column
db['k5cls'] = k5cls.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
db.plot(column='k5cls', categorical=True, legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
plt.axis('equal')
# Add title
plt.title(r'Geodemographic Clusters (k-means, $k=5$)')
# Display the map
plt.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_29_0.png)

</div>
</div>
</div>



The map provides a useful view of the clustering results; it allows for
a visual inspection of the extent to which Tobler's first law of geography is
reflected in the multivariate clusters. Recall that the law implies that nearby
tracts should be more similar to one another than tracts that are geographically
more distant from each other. We can see evidence of this in
our cluster map, since clumps of tracts with the same color emerge. However, this
visual inspection is obscured by the complexity of the underlying spatial
units. Our eyes are drawn to the larger polygons in the eastern part of the
county, giving the impression that cluster 2 is the dominant cluster. While this
seems to be true in terms of land area (and we will verify this below), there is
more to the cluster pattern than this. Because the tract polygons are all 
different sizes and shapes, we cannot solely rely on our eyes to interpret 
the spatial distribution of clusters.

### Statistical Analysis of the Cluster Map

To complement the geovisualization of the clusters, we can explore the
statistical properties of the cluster map. This process allows us to delve
into what observations are part of each cluster and what their
characteristics are.
This gives us the profile of each cluster so we can interpret the meaning of the
labels we've obtained. We can start, for example, by
considering cardinality, or the count of observations in each cluster:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Group data table by cluster label and count observations
k5sizes = db.groupby('k5cls').size()
k5sizes

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
k5cls
0    232
1      8
2    247
3     24
4    116
dtype: int64
```


</div>
</div>
</div>



And we can get a visual representation of cardinality as well:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
_ = k5sizes.plot.bar()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_33_0.png)

</div>
</div>
</div>



There are substantial differences in the sizes of the five clusters, with two very
large clusters (0, 2), one medium sized cluster (4), and two small clusters (1,
3). Cluster 2 is the largest when measured by the number of assigned tracts.
This confirms our intuition from the map above, where we got the visual impression
that tracts in cluster 2 seemed to have the largest area. Let's see if this is 
the case. To do so we can use the `dissolve` operation in `geopandas`, which 
combines all tracts belonging to each cluster into a single
polygon object. After we have dissolved all the members of the clusters,
we report the total land area of the cluster:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Dissolve areas by Cluster, aggregate by summing, and keep column for area
areas = db.dissolve(by='k5cls', aggfunc='sum')['AREALAND']
areas

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
k5cls
0    3445601816
1      53860347
2    5816736150
3     220120882
4     752511344
Name: AREALAND, dtype: int64
```


</div>
</div>
</div>



And, to show this visually:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
areas.plot.bar()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
<matplotlib.axes._subplots.AxesSubplot at 0x1a28a8c3c8>
```


</div>
</div>
<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_37_1.png)

</div>
</div>
</div>



Our visual impression is confirmed: cluster 2 contains tracts that
together comprise 5,816,736,150 square meters (approximately 2,245 square miles),
which accounts for over half of the total land area in the county:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
areas[2]/areas.sum()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
0.5653447326157774
```


</div>
</div>
</div>



Let's move on to build the profiles for each cluster. Again, the profiles is what
provides the conceptual shorthand, moving from the arbitrary label to a meaningful
collection of observations with similar attributes. To build a basic profile, we can
compute the means of each of the attributes in every cluster:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Group table by cluster label, keep the variables used 
# for clustering, and obtain their mean
k5means = db.groupby('k5cls')[cluster_variables].mean()
k5means.T

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
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
      <th>k5cls</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Median Val</th>
      <td>430287.707171</td>
      <td>1.700850e+06</td>
      <td>272717.408907</td>
      <td>1.073158e+06</td>
      <td>658511.206897</td>
    </tr>
    <tr>
      <th>pct_white</th>
      <td>0.725640</td>
      <td>9.253820e-01</td>
      <td>0.651341</td>
      <td>8.485452e-01</td>
      <td>0.803962</td>
    </tr>
    <tr>
      <th>pct_rental</th>
      <td>0.405961</td>
      <td>2.344535e-01</td>
      <td>0.520519</td>
      <td>2.839565e-01</td>
      <td>0.332000</td>
    </tr>
    <tr>
      <th>pct_female_hh</th>
      <td>0.097588</td>
      <td>1.180614e-01</td>
      <td>0.105221</td>
      <td>1.012079e-01</td>
      <td>0.098318</td>
    </tr>
    <tr>
      <th>pct_bachelor</th>
      <td>0.009961</td>
      <td>1.874663e-03</td>
      <td>0.019729</td>
      <td>2.455057e-03</td>
      <td>0.004084</td>
    </tr>
    <tr>
      <th>Median Num</th>
      <td>5.349009</td>
      <td>6.437500e+00</td>
      <td>4.692308</td>
      <td>6.079167e+00</td>
      <td>5.688793</td>
    </tr>
    <tr>
      <th>Gini index</th>
      <td>0.400495</td>
      <td>5.257750e-01</td>
      <td>0.402543</td>
      <td>4.676458e-01</td>
      <td>0.423886</td>
    </tr>
    <tr>
      <th>Med Age</th>
      <td>37.031940</td>
      <td>5.057500e+01</td>
      <td>33.447368</td>
      <td>4.497083e+01</td>
      <td>41.429310</td>
    </tr>
    <tr>
      <th>Travel tim</th>
      <td>2438.086207</td>
      <td>1.215125e+03</td>
      <td>2072.421053</td>
      <td>2.056792e+03</td>
      <td>2289.508621</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



We see that cluster 3, for example, is composed of tracts that have
the highest average `Median_val`, while cluster 2 has the highest level of inequality
(`Gini index`), and cluster 1 contains an older population (`Med Age`)
who tend to live in housing units with more rooms (`Median Num`).
Average values, however, can hide a great deal of detail and, in some cases,
give wrong impressions about the type of data distribution they represent. To
obtain more detailed profiles, we can use the `describe` command in `pandas`, 
after grouping our observations by their clusters:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Group table by cluster label, keep the variables used 
# for clustering, and obtain their descriptive summary
k5desc = db.groupby('k5cls')[cluster_variables].describe()
# Loop over each cluster and print a table with descriptives
for cluster in k5desc.T:
    print('\n\t---------\n\tCluster %i'%cluster)
    print(k5desc.T[cluster].unstack())

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```

	---------
	Cluster 0
               count           mean           std          min            25%  \
Median Val     232.0  430287.707171  47636.060405  354400.0000  385050.000000   
pct_white      232.0       0.725640      0.143014       0.0000       0.665323   
pct_rental     232.0       0.405961      0.219609       0.0000       0.231308   
pct_female_hh  232.0       0.097588      0.029461       0.0000       0.086273   
pct_bachelor   232.0       0.009961      0.010269       0.0000       0.002747   
Median Num     232.0       5.349009      0.976408       2.7000       4.800000   
Gini index     232.0       0.400495      0.060185       0.0128       0.365100   
Med Age        232.0      37.031940      7.360078      14.9000      33.400000   
Travel tim     232.0    2438.086207   1794.321201       0.0000    1617.000000   

                         50%            75%            max  
Median Val     427600.000000  463825.000000  543900.000000  
pct_white           0.745380       0.815653       0.965002  
pct_rental          0.381552       0.551725       1.000000  
pct_female_hh       0.102241       0.116436       0.166731  
pct_bachelor        0.007200       0.013292       0.058414  
Median Num          5.400000       6.100000       7.800000  
Gini index          0.404450       0.437100       0.574000  
Med Age            36.500000      40.800000      71.500000  
Travel tim       2213.500000    2954.750000   21780.000000  

	---------
	Cluster 1
               count          mean            std           min           25%  \
Median Val       8.0  1.700850e+06  202293.816833  1.424300e+06  1.599575e+06   
pct_white        8.0  9.253820e-01       0.039039  8.565188e-01  9.067396e-01   
pct_rental       8.0  2.344535e-01       0.090575  1.058824e-01  1.461760e-01   
pct_female_hh    8.0  1.180614e-01       0.024708  8.609795e-02  9.646732e-02   
pct_bachelor     8.0  1.874663e-03       0.003391  0.000000e+00  0.000000e+00   
Median Num       8.0  6.437500e+00       0.919530  4.800000e+00  5.975000e+00   
Gini index       8.0  5.257750e-01       0.036813  4.547000e-01  5.069750e-01   
Med Age          8.0  5.057500e+01       3.143133  4.540000e+01  4.952500e+01   
Travel tim       8.0  1.215125e+03     375.770993  7.300000e+02  9.027500e+02   

                        50%           75%           max  
Median Val     1.704000e+06  1.777100e+06  2.000001e+06  
pct_white      9.347479e-01  9.496601e-01  9.718606e-01  
pct_rental     2.606813e-01  3.052264e-01  3.425729e-01  
pct_female_hh  1.201224e-01  1.363923e-01  1.556420e-01  
pct_bachelor   0.000000e+00  1.965965e-03  9.239815e-03  
Median Num     6.500000e+00  6.900000e+00  7.800000e+00  
Gini index     5.354000e-01  5.466250e-01  5.743000e-01  
Med Age        5.035000e+01  5.117500e+01  5.670000e+01  
Travel tim     1.281000e+03  1.398000e+03  1.854000e+03  

	---------
	Cluster 2
               count           mean           std           min  \
Median Val     247.0  272717.408907  68847.844348  16600.000000   
pct_white      247.0       0.651341      0.190667      0.101900   
pct_rental     247.0       0.520519      0.211114      0.109715   
pct_female_hh  247.0       0.105221      0.020948      0.038137   
pct_bachelor   247.0       0.019729      0.013652      0.000000   
Median Num     247.0       4.692308      0.852723      2.700000   
Gini index     247.0       0.402543      0.046071      0.262600   
Med Age        247.0      33.447368      5.470630     24.300000   
Travel tim     247.0    2072.421053    677.155331    550.000000   

                         25%            50%            75%            max  
Median Val     245050.000000  293000.000000  319800.000000  351200.000000  
pct_white           0.583502       0.699765       0.788069       0.927177  
pct_rental          0.336651       0.539113       0.685844       0.927035  
pct_female_hh       0.092346       0.103783       0.119947       0.170026  
pct_bachelor        0.010110       0.017334       0.028896       0.068300  
Median Num          4.000000       4.600000       5.400000       6.500000  
Gini index          0.373000       0.400300       0.429800       0.585300  
Med Age            29.400000      32.500000      36.450000      65.400000  
Travel tim       1609.500000    2003.000000    2460.500000    4659.000000  

	---------
	Cluster 3
               count          mean            std            min  \
Median Val      24.0  1.073158e+06  120340.198628  881700.000000   
pct_white       24.0  8.485452e-01       0.105947       0.586509   
pct_rental      24.0  2.839565e-01       0.161366       0.079941   
pct_female_hh   24.0  1.012079e-01       0.018171       0.063891   
pct_bachelor    24.0  2.455057e-03       0.002963       0.000000   
Median Num      24.0  6.079167e+00       1.375557       3.900000   
Gini index      24.0  4.676458e-01       0.042361       0.385800   
Med Age         24.0  4.497083e+01       8.540974      27.100000   
Travel tim      24.0  2.056792e+03    1237.905278     804.000000   

                         25%           50%           75%           max  
Median Val     964325.000000  1.098200e+06  1.131350e+06  1.354700e+06  
pct_white           0.842347  8.878381e-01  9.179736e-01  9.385125e-01  
pct_rental          0.145002  2.216039e-01  4.490072e-01  5.808486e-01  
pct_female_hh       0.092296  1.021899e-01  1.144775e-01  1.359801e-01  
pct_bachelor        0.000000  9.721194e-04  4.536051e-03  1.022677e-02  
Median Num          5.000000  6.400000e+00  7.250000e+00  8.100000e+00  
Gini index          0.445450  4.670000e-01  4.835250e-01  5.766000e-01  
Med Age            38.825000  4.555000e+01  5.355000e+01  5.630000e+01  
Travel tim       1152.750000  1.645500e+03  2.939250e+03  5.887000e+03  

	---------
	Cluster 4
               count           mean           std            min  \
Median Val     116.0  658511.206897  82853.851183  545600.000000   
pct_white      116.0       0.803962      0.122426       0.426606   
pct_rental     116.0       0.332000      0.199404       0.038530   
pct_female_hh  116.0       0.098318      0.024188       0.020283   
pct_bachelor   116.0       0.004084      0.005218       0.000000   
Median Num     116.0       5.688793      1.264001       2.300000   
Gini index     116.0       0.423886      0.056062       0.295500   
Med Age        116.0      41.429310      6.776468      21.700000   
Travel tim     116.0    2289.508621   1190.678772     692.000000   

                         25%            50%            75%            max  
Median Val     601750.000000  638900.000000  707225.000000  863700.000000  
pct_white           0.739922       0.844445       0.889507       0.960493  
pct_rental          0.166894       0.290281       0.453746       0.781067  
pct_female_hh       0.085454       0.097574       0.114602       0.166557  
pct_bachelor        0.000000       0.002867       0.005968       0.031444  
Median Num          4.575000       5.900000       6.800000       8.300000  
Gini index          0.385800       0.424550       0.465100       0.586100  
Med Age            36.850000      41.000000      47.125000      59.300000  
Travel tim       1484.500000    2077.500000    2794.250000    8339.000000  
```
</div>
</div>
</div>



However, this approach quickly gets out of hand: more detailed profiles can simply
return to an unwieldy mess of numbers. A better approach to constructing
cluster profiles is be to draw the distributions of cluster members' data.
To do this we need to "tidy up" the dataset. A tidy dataset ([Wickham,
2014](https://www.jstatsoft.org/article/view/v059i10)) is one where every row is
an observation, and every column is a variable. Thus, a few steps are required 
to tidy up our labelled data:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Index db on cluster ID
tidy_db = db.set_index('k5cls')
# Keep only variables used for clustering
tidy_db = tidy_db[cluster_variables]
# Stack column names into a column, obtaining 
# a "long" version of the dataset
tidy_db = tidy_db.stack()
# Take indices into proper columns
tidy_db = tidy_db.reset_index()
# Rename column names
tidy_db = tidy_db.rename(columns={
                        'level_1': 'Attribute', 
                        0: 'Values'})
# Check out result
tidy_db.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
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
      <th>k5cls</th>
      <th>Attribute</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>Median Val</td>
      <td>1.038200e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>pct_white</td>
      <td>9.385125e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>pct_rental</td>
      <td>7.994078e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>pct_female_hh</td>
      <td>8.726068e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>pct_bachelor</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



Now we are ready to plot. Below, we'll show the distribution of each cluster's values
for each variable. This gives us the full distributional profile of each cluster:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Setup the facets
facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue='k5cls', \
                  sharey=False, sharex=False, aspect=2, col_wrap=3)
# Build the plot from `sns.kdeplot`
_ = facets.map(seaborn.kdeplot, 'Values', shade=True).add_legend()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_47_0.png)

</div>
</div>
</div>



This allows us to see that, while some attributes such as the percentage of
female households (`pct_female_hh`) display largely the same distribution for
each cluster, others paint a much more divided picture (e.g. `Median Val`).
Taken altogether, these graphs allow us to start delving into the multidimensional 
complexity of each cluster and the types of areas behind them.

## Hierarchical Clustering

As mentioned above, k-means is only one clustering algorithm. There are
plenty more. In this section, we will take a similar look at the San Diego
dataset using another staple of the clustering toolkit: agglomerative
hierarchical clustering (AHC). Agglomerative clustering works by building a hierarchy of
clustering solutions that starts with all singletons (each observation is a single
cluster in itself) and ends with all observations assigned to the same cluster.
These extremes are not very useful in themselves. But, in between, the hierarchy
contains many distinct clustering solutions with varying levels of detail. 
The intuition behind the algorithm is also rather straightforward: 

1) begin with everyone as part of its own cluster; 
2) find the two closest observations based on a distance metric (e.g. euclidean); 
3) join them into a new cluster; 
4) repeat steps 2) and 3) until reaching the degree of aggregation desired. 

The algorithm is thus called "agglomerative"
because it starts with individual clusters and "agglomerates" them into fewer
and fewer clusters containing more and more observations each. Also, like with 
k-means, AHC does require the user to specify a number of clusters in advance.
This is because, following from the mechanism the method has to build clusters, 
AHC can provide a solution with as many clusters as observations ($k=n$),
or with a only one ($k=1$).

Enough of theory, let's get coding! In Python, AHC can be run
with `scikit-learn` in very much the same way we did for k-means in the previous
section. In this case, we use the `AgglomerativeClustering` class and again 
use the `fit` method to actually apply the clustering algorithm to our data:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Set seed for reproducibility
numpy.random.seed(0)
# Iniciate the algorithm
model = AgglomerativeClustering(linkage='ward', n_clusters=5)
# Run clustering
model.fit(db[cluster_variables])
# Assign labels to main data table
db['ward5'] =model.labels_

```
</div>

</div>



As above, we can check the number of observations that fall within each cluster:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
ward5sizes = db.groupby('ward5').size()
ward5sizes

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
ward5
0    134
1    326
2     16
3    145
4      6
dtype: int64
```


</div>
</div>
</div>



Further, we can check the simple average profiles of our clusters:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
ward5means = db.groupby('ward5')[cluster_variables].mean()
ward5means.T

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
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
      <th>ward5</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Median Val</th>
      <td>673423.880597</td>
      <td>298094.171779</td>
      <td>1.193119e+06</td>
      <td>453176.883198</td>
      <td>1.786634e+06</td>
    </tr>
    <tr>
      <th>pct_white</th>
      <td>0.804847</td>
      <td>0.673469</td>
      <td>8.664667e-01</td>
      <td>0.713027</td>
      <td>9.341647e-01</td>
    </tr>
    <tr>
      <th>pct_rental</th>
      <td>0.319893</td>
      <td>0.492282</td>
      <td>2.994236e-01</td>
      <td>0.417104</td>
      <td>2.222373e-01</td>
    </tr>
    <tr>
      <th>pct_female_hh</th>
      <td>0.099066</td>
      <td>0.104657</td>
      <td>1.077750e-01</td>
      <td>0.093584</td>
      <td>1.201785e-01</td>
    </tr>
    <tr>
      <th>pct_bachelor</th>
      <td>0.004003</td>
      <td>0.017606</td>
      <td>1.985041e-03</td>
      <td>0.009723</td>
      <td>2.324019e-03</td>
    </tr>
    <tr>
      <th>Median Num</th>
      <td>5.763433</td>
      <td>4.832515</td>
      <td>5.887500e+00</td>
      <td>5.350138</td>
      <td>6.600000e+00</td>
    </tr>
    <tr>
      <th>Gini index</th>
      <td>0.425901</td>
      <td>0.401975</td>
      <td>4.849938e-01</td>
      <td>0.399595</td>
      <td>5.212333e-01</td>
    </tr>
    <tr>
      <th>Med Age</th>
      <td>41.945522</td>
      <td>34.262883</td>
      <td>4.521875e+01</td>
      <td>36.689725</td>
      <td>5.145000e+01</td>
    </tr>
    <tr>
      <th>Travel tim</th>
      <td>2273.664179</td>
      <td>2124.248466</td>
      <td>1.877125e+03</td>
      <td>2538.565517</td>
      <td>1.148167e+03</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



And again, we can create a plot of the profiles' distributions (after properly 
tidying up):



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Index db on cluster ID
tidy_db = db.set_index('ward5')
# Keep only variables used for clustering
tidy_db = tidy_db[cluster_variables]
# Stack column names into a column, obtaining 
# a "long" version of the dataset
tidy_db = tidy_db.stack()
# Take indices into proper columns
tidy_db = tidy_db.reset_index()
# Rename column names
tidy_db = tidy_db.rename(columns={
                        'level_1': 'Attribute', 
                        0: 'Values'})
# Check out result
tidy_db.head()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
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
      <th>ward5</th>
      <th>Attribute</th>
      <th>Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Median Val</td>
      <td>1.038200e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>pct_white</td>
      <td>9.385125e-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>pct_rental</td>
      <td>7.994078e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>pct_female_hh</td>
      <td>8.726068e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>pct_bachelor</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
# Setup the facets
facets = seaborn.FacetGrid(data=tidy_db, col='Attribute', hue='ward5', \
                  sharey=False, sharex=False, aspect=2, col_wrap=3)
# Build the plot as a `sns.kdeplot`
_ = facets.map(seaborn.kdeplot, 'Values', shade=True).add_legend()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_56_0.png)

</div>
</div>
</div>



For the sake of brevity, we will not spend much time on the plots above.
However, the interpretation is analogous to that of the k-means example.

On the spatial side, we can explore the geographical dimension of the
clustering solution by making a map the clusters:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
db['ward5'] =model.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
db.plot(column='ward5', categorical=True, legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
plt.axis('equal')
# Add title
plt.title('Geodemographic Clusters (AHC, $k=5$)')
# Display the map
plt.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_58_0.png)

</div>
</div>
</div>



And, to make comparisons simpler, we can display both the k-means and the AHC
results side by side:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
db['ward5'] =model.labels_
# Setup figure and ax
f, axs = plt.subplots(1, 2, figsize=(12, 6))

ax = axs[0]
# Plot unique values choropleth including a legend and with no boundary lines
db.plot(column='ward5', categorical=True, cmap='Set2', 
        legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
ax.axis('equal')
# Add title
ax.set_title('K-Means solution ($k=5$)')

ax = axs[1]
# Plot unique values choropleth including a legend and with no boundary lines
db.plot(column='k5cls', categorical=True, cmap='Set3',
        legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
ax.axis('equal')
# Add title
ax.set_title('AHC solution ($k=5$)')

# Display the map
plt.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_60_0.png)

</div>
</div>
</div>



While we must remember our earlier caveat about how irregular polygons can 
baffle our visual intuition, a closer visual inspection of the cluster geography
suggests a clear pattern: although they are not identical, both clusterings capture
very similar overall spatial structure. Furthermore, both solutions slightly violate 
Tobler's law, since all of the clusters have disconnected components. The five
multivariate clusters in each case are actually composed of many disparate 
geographical areas, strewn around the map according only to the structure of the
data and not its geography. That is, in order to travel to
every tract belonging to a cluster, we would have to journey through
other clusters as well.

## Spatially Constrained Hierarchical Clustering

Fragmented clusters are not intrinsically invalid, particularly if we are
interested in exploring the overall structure and geography of multivariate
data. However, in some cases, the application we are interested in might
require that all the observations in a class be spatially connected. For
example, when detecting communities or neighborhoods (as is sometimes needed when
drawing electoral or census boundaries), they are nearly always distinct 
self-connected areas, unlike our clusters shown above. To ensure that clusters are
not spatially fragmented, we turn to regionalization.

Regionalization methods are clustering techniques that impose a spatial constraints
on clusters. In other words, the result of a regionalization algorithm contains clusters with
areas that are geographically coherent, in addition to having coherent data profiles. 
Effectively, this means that regionalization methods construct clusters that are 
all internally-connected; these are the *regions*. Thus, a regions' members must
be geographically *nested* within the region's boundaries.

This type of nesting relationship is easy to identify
in the real world. For example, counties nest within states, or, in the UK, 
local super output areas (LSOAs) nest within middle super output areas (MSOAs). 
The difference between these real-world nestings and the output of a regionalization
algorithm is that the real-world nestings are aggregated according to administrative principles, but regions' members are aggregated according to a statistical technique. In the same manner as the
clustering techniques explored above, these regionalization methods aggregate 
observations that are similar in their covariates; the profiles of regions are useful
in a similar manner as the profiles of clusters. But, in regionalization, the 
clustering is also spatially constrained, so the region profiles and members will
likely be different from the unconstrained solutions.

As in the non-spatial case, there are many different regionalization methods.
Each has a different way to measure (dis)similarity, how the similarity is used
to assign labels, how these labels are iteratively adjusted, and so on. However,
as with clustering algorithms, regionalization methods all share a few common traits.
In particular, they all take a set of input attributes and a representation of 
spatial connectivity in the form of a binary spatial weights matrix. Depending 
on the algorithm, they also require the desired number of output regions. For
illustration, we will take the AHC algorithm we have just used above, and apply 
an additional spatial constraint. In `scikit-learn`, this is done using
our spatial weights matrix as a `connectivity` option.
This will force the agglomerative algorithm to only allow observations to be grouped
in a cluster if they are also spatially connected:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
numpy.random.seed(123456)
model = AgglomerativeClustering(linkage='ward',
                                            connectivity=w.sparse,
                                            n_clusters=5)
model.fit(db[cluster_variables])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=<627x627 sparse matrix of type '<class 'numpy.float64'>'
	with 3961 stored elements in Compressed Sparse Row format>,
            linkage='ward', memory=None, n_clusters=5,
            pooling_func=<function mean at 0x10ab8af28>)
```


</div>
</div>
</div>



Let's inspect the output:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
db['ward5wq'] = model.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
db.plot(column='ward5wq', categorical=True, legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
plt.axis('equal')
# Add title
plt.title(r'Geodemographic Regions (Ward, $k=5$, Queen Contiguity)')
# Display the map
plt.show()


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_64_0.png)

</div>
</div>
</div>



Introducing the spatial constraint results in fully-connected clusters with much
more concentrated spatial distributions. From an initial visual impression, it might
appear that our spatial constraint has been violated: there are tracts for both cluster 0 and
cluster 1 that appear to be disconnected from the rest of their clusters.
However, closer inspection reveals that each of these tracts is indeed connected
to another tract in its own cluster by very narrow shared boundaries.

### Changing the spatial constraint

The spatial constraint in regionalization algorithms is structured by the
spatial weights matrix we use. An interesting
question is thus how the choice of weights influences the final region structure.
Fortunately, we can directly explore the impact that a change in the spatial weights matrix has on
regionalization. To do so, we use the same attribute data
but replace the Queen contiguity matrix with a spatial k-nearest neighbor matrix,
where each observation is connected to its four nearest observations, instead
of those it touches.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
w = KNN.from_shapefile(data.san_diego_tracts(), k=4)

```
</div>

</div>



With this matrix connecting each tract to the four closest tracts, we can run 
another AHC regionalization:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
numpy.random.seed(123456)
model = AgglomerativeClustering(linkage='ward',
                                            connectivity=w.sparse,
                                            n_clusters=5)
model.fit(db[cluster_variables])

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=<627x627 sparse matrix of type '<class 'numpy.float64'>'
	with 2508 stored elements in Compressed Sparse Row format>,
            linkage='ward', memory=None, n_clusters=5,
            pooling_func=<function mean at 0x10ab8af28>)
```


</div>
</div>
</div>



And plot the final regions:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
db['ward5wknn'] = model.labels_
# Setup figure and ax
f, ax = plt.subplots(1, figsize=(9, 9))
# Plot unique values choropleth including a legend and with no boundary lines
db.plot(column='ward5wknn', categorical=True, legend=True, linewidth=0, ax=ax)
# Remove axis
ax.set_axis_off()
# Keep axes proportionate
plt.axis('equal')
# Add title
plt.title('Geodemographic Regions (Ward, $k=5$, four nearest neighbors)')
# Display the map
plt.show()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">

{:.output_png}
![png](../images/notebooks/10_clustering_and_regionalization_70_0.png)

</div>
</div>
</div>



Even though we have specified a spatial constraint, the constraint applies to the
connectivity graph modeled by our weights matrix. Therefore, using k-nearest neighbors
to constrain the agglomerative clustering may not result in regions that are connected
according to a different connectivity rule, such as the queen contiguity rule used
in the previous section. However, the regionalization here is fortuitous; even though
we used the 4-nearest tracts to constrain connectivity, all but one of the clusters, 
cluster 4, is *also* connected according to our earlier queen contiguity rule. 

At first glance, this may seem counter-intuitive. We did specify the spatial
constraint, so our initial reaction is that the connectivity constraint is
violated. However, this is not the case, since the constraint applies to the
k-nearest neighbor graph, not the queen contiguity graph. Therefore, since tracts
in this solution are considered as connected to their four closest neighbors,
clusters can "leapfrog" over one another. Thus, it is important to recognize that
the apparent spatial structure of regionalizations will depend on how the 
connectivity of observations is modeled. 

## Conclusion

Overall, clustering and regionalization are two complementary tools to reduce the
complexity in multivariate data and build better understandings of the spatial structure 
of data. Often, there is simply too much data to examine every variables' map and its
relation to all other variable maps. 
Thus, clustering reduces this complexity into a single conceptual shorthand by which 
people can easily describe complex and multifaceted data. 
Clustering constructs groups of observations (called *clusters*)
with coherent *profiles*, or distinct and internally-consistent 
distributional/descriptive characteristics. 
These profiles are the conceptual shorthand, since members of each cluster should
be more similar to the cluster at large than they are to any other cluster. 
Many different clustering methods exist; they differ on how the "cluster at large" 
is defined, and how "similar" members must be to clusters, or how these clusters
are obtained.
Regionalization is a special kind of clustering with an additional geographic requirement. 
Observations should be grouped so that each spatial cluster,
or *region*, is spatially-coherent as well as data-coherent. 
Thus, regionalization is often concerned with connectivity in a contiguity 
graph for data collected in areas; this ensures that the regions that are identified
are fully internally-connected. 
However, since many regionalization methods are defined for an arbitrary connectivity structure,
these graphs can be constructed according to different rules as well, such as the k-nearest neighbor graph. 

In this chapter, we discussed the conceptual basis for clustering and regionalization, 
as well showing why clustering is done. 
Further, we have shown how to build clusters using spatial data science packages, 
and how to interrogate the meaning of these clusters as well.
More generally, clusters are often used in predictive and explanatory settings, 
in addition to being used for exploratory analysis in their own right.
Clustering and regionalization are intimately related to the analysis of spatial autocorrelation as well,
since the spatial structure and covariation in multivariate spatial data is what
determines the spatial structure and data profile of discovered clusters or regions.
Thus, clustering and regionalization are essential tools for the spatial data scientist.

## Questions

1. What disciplines employ regionalization? Cite concrete examples for each discipline you list.
2. Contrast and compare  the concepts of *clusters* and *regions*?
3. In evaluating the quality of the solution to a regionalization problem, how might traditional measures of cluster evaluation be used? In what ways might those measures be limited and need expansion to consider the geographical dimensions of the problem?
4. Discuss the implications for the processes of regionalization that follow from the number of *connected components* in the spatial weights matrix that would be used.
5. True or false: The average silhouette score for a spatially constrained solution will be no larger than the average silhouette score for an unconstrained solution. Why, or why not? (add reference and  or explain silhouette)
6. Consider two possible weights matrices for use in a spatially constrained clustering problem. Both form a single connected component for all the areal units. However, they differ in the sparsity of their adjacency graphs (think Rook versus queen). How might this sparsity affect the quality of the clustering solution?
7. What are the challenges and opportunities that spatial dependence pose for spatial cluster formation?
8. In other areas of spatial analysis, the concept of multilevel modeling (cites) exploits the hierarchical nesting of spatial units at different levels of aggregation. How might such nesting be exploited in the implementation of regionalization algorithms? What are some possible limitations/challenges that such nesting imposes/represents in obtaining a regionalization solution.
9. Using a spatial weights object obtained as `w = libpysal.weights.lat2W(20,20)`, what are the number of unique ways to partition the graph into 20 clusters of 20 units each, subject to each cluster being a connected component? What are the unique number of possibilities for `w = libpysal.weights.lat2W(20,20, rook=False)` ?

---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

