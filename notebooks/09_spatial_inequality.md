---
jupyter:
  jupytext:
    cell_metadata_json: true
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
# Spatial Inequality Analysis


## Introduction

At the time this is being written, the topic of inequality is at the top of
agenda of policy makers and is drawing considerable attention in academic
circles. This is due to historic levels of inequality across the globe
(references). Much of the focus has been on *interpersonal income inequality*,
yet there is a growing recognition that the question of *interregional income
inequality* requires further attention as the growing gaps between poor and rich
regions have been identified as key drivers of political polarization in
developing and developed countries (CITE ANDRES).

Indeed, while the two literatures, personal and regional inequality, are
related, they have developed in a largely parallel fashion with limited
cross-fertilization. In this notebook, we examine how a spatially explicit focus
can provide insights on inequality and its dynamics. We also show the lineage of
regional inequality analysis to make explicit the linkage between it and the
older literature on personal inequality analysis.

We begin with an introduction to classic methods for interpersonal income
inequality analysis and how they have been adopted to the question of regional
inequalities. These include a number of graphical tools along side familiar
indices of inequality. As we discuss more fully, the use of these classical
methods in spatially referenced data, while useful in providing insights on some
of the aspects of spatial inequality, fails to fully capture the nature of
geographical disparities and their dynamics. Thus, we next move to spatially
explicit measures for regional inequality analysis. The notebook closes with
some recent extensions of some classical measures to more fully examine the
spatial dimensions of regional inequality dynamics.


## Data: US State Per-Capita Income 1969-2017


We focus on the case of the United States over the
period  1969-2017.

<!-- #endregion -->

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
---
<!-- #endregion -->
```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
%matplotlib inline

import seaborn
import pandas
import geopandas
import pysal
import numpy
import mapclassify
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 10, 5
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}

gdf = geopandas.read_file('../data/us_county_income/usincome_final.shp')
gdf.head()

```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
Inspection of the head of the data frame reveals that the years appear as columns in the data set, together with information about the particular record.
This format is an example of a [*wide* longitudinal data set](https://www.theanalysisfactor.com/wide-and-long-data/).

<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
gdf[['LineCode', 'Descriptio']].head()

```
```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pci_df = gdf[gdf.LineCode == 3]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pci_df.head()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pci_df.shape
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pci_df[['1969', 'STATEFP', 'COUNTYFP', 'geometry']].head()
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
## Global Inequality
We begin our examination of inequality by focusing on several global measures. Here, global means the measure is concerned with the overall, or a-spatial, nature of inequality. Several classic measures of inequality are available for this purpose.

In general terms, measures of inequality focus on the dispersion present in an income distribution. In the case of regional, or spatial, inequality the distributions are defined on average or per-capita incomes for a spatial unit, such as a county, census tract, or region. For the US counties, we can visualize the distribution of per capita incomes for the first year in the sample as follows:


To get a sense for the value distribution for per capita income, we can first discretize the distribution:
<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
import seaborn  as sns
sns.set()
sns.distplot(pci_df['1969'])

```


The long right tail is a prominent feature of the distribution, and is common in the study of incomes. A key point to keep in mind here is that the unit of measurement in this data is a spatial aggregate - a county. By contrast, in the wider inequality literature the observational unit is typically a household or individual. In the latter distributions, the degree of skewness is often more pronounced. (Say something about averaging effect for regional distributions).


The density is a powerful summary device that captures the overall morphology of the *value* distribution. At the same time, the density is silent on the underlying *spatial distribution* of county incomes. We can look at this second view of the distribution using a choropleth map:

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
from pysal.viz import mapclassify
pci_1969 = pci_df['1969']
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
q5_1969 = mapclassify.Quantiles(pci_df['1969'])

```
```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
_= pci_df.plot(column='1969', scheme='Quantiles', legend=True,
                 edgecolor='none',
             legend_kwds={'loc': 'lower right',
                         'bbox_to_anchor':(1.5, 0.0)}, figsize=(12, 12))
```

SAY SOMETHING ABOUT CLUSTERING AND PATTERN
The choropleth and the kernel density provide different visual depictions of the distribution of county incomes. These are useful for developing an understanding of the overall  To gain more specific insights on the level of inequality in the distribution, we turn to a number of inequality indices.


### 20:20 Ratio

One commonly used measure of inequality in a distribution is the so called 20:20 ratio, which is defined as the ratio of the incomes at the 80th percentile over that at the 20th percentile: 
<!-- #endregion -->
```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
q5_1969.bins
```


```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
i_20_20 = q5_1969.bins[-2]/q5_1969.bins[0]
i_20_20
```
<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
In 1969 the richest 20% of the counties had an income that was 1.5 times the poorest 20% of the counties. The 20:20 ratio has the advantage of being robust to outliers at the top and the bottom of the distribution. 

We can examine the dynamics of this global inequality measure by creating a simple function to apply to all years in our time series:
<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
def ineq_2020(values):
    q5 = mapclassify.Quantiles(values)
    return q5.bins[-2]/q5.bins[0]
    
    
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
years = list(range(1969, 2018))
i_20_20_all = numpy.array([ ineq_2020(pci_df[str(year)]) for year in years])
_ = plt.plot(years, i_20_20_all)
_ = plt.ylabel("20:20 ratio")
_ = plt.xlabel("Year")


```
<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
The ratio has a U-shaped pattern over time, bottoming out around 1994 after a long secular decline in the early period. Post 1994, however, the trend is one of increasing inequality up until 2013 where there is a turn towards lower income inequality between the counties.

<!-- #endregion -->

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
For other classic measures of inequality, we will use the `inequality` package from `pysal`:
<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
from pysal.explore import inequality
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
### Gini Index

The Gini index can be derived in a number of ways. Visually, the Gini is defined as the ratio of the area between
the line of equality and the Lorenze curve, both of which are defined on a a two-dimensional space with population
percentiles on the x-axis and cumulative income on the vertical axis.

We can construct each of these as follows:

<!-- #endregion -->

```python
y = pci_df.sort_values(by=['1969'])['1969']
```

```python
Fy = (y / y.sum()).cumsum()
```

```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
```

```python
n = y.shape[0]
Fn = np.arange(1, n+1)/n
```

```python
f, ax = plt.subplots()
ax.plot(Fn,Fy)
ax.plot(Fn, Fn)
#plt.xlim(0, 1.0)
```
The blue line is the Lorenze curve, and the Gini is the area between it and the 45-degree line of equality, expressed as a ratio over the area under the line of equality.

To examine how inequality evolves over time, we can create a function for the Lorenze curve:

```python
def lorenz(y):
    y = np.sort(y)
    Fy = (y / y.sum()).cumsum()
    n = y.shape[0]
    Fn = np.arange(1, n+1)/n
    return Fy
```

```python
lorenz(y)
```
and then call this for each year in our sample:
```python
lorenz_curves = np.array([ lorenz(pci_df[str(year)]) for year in years])
```

```python
lorenz_curves
```

```python
lorenz_curves.shape
```

```python
f, ax = plt.subplots()
ax.plot(Fn,Fn)
for c in lorenz_curves:
    ax.plot(Fn, c)

```

The compression of the Lorenze curves makes it difficult to ascertain the temporal pattern in inequality. Focusing explicilty on the Gini coefficients may shed more light on this evolution:

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
g69 = inequality.gini.Gini(pci_df['1969'].values)

```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
g69.g
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
years = [str(y) for y in range(1969, 2018)]
ginis = numpy.array([inequality.gini.Gini(pci_df[year].values).g for year in years])
years = numpy.array([int(y) for y in years])
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
gini_df = pandas.DataFrame(data = numpy.hstack([[years, ginis]]).T, columns=['Year', 'Gini'])
gini_df['Year'] = gini_df['Year'].astype(int)
gini_df.head()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
gini_df.index = pandas.to_datetime(gini_df['Year'], format="%Y")
gini_df = gini_df.drop(columns=["Year"])
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
gini_df.plot(y=["Gini"])
```

MAYBE ADD 
One interesting note about the relationship between the Gini coefficient, the Loreznze curve, and the spatial distribution of incomes



### Theil's index

A third commonly used measure of inequality is Theil's $T$ given as:
$$T = \sum_{i=1}^m \left( \frac{y_i}{\sum_{i=1}^m y_i} \ln \left[ m \frac{y_i}{\sum_{i=1}^m y_i}\right] \right)$$
where $y_i$ is per-capita income in area $i$ among $m$ areas. In PySAL, we can calculate this index each year as:

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
gini_df['T'] = [inequality.theil.Theil(pci_df[str(y)]).T for y in years]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
#gini_df.plot(figsize=(15,4))
gini_df.plot(subplots=True, figsize=(15,6))
#gini_df.plot(y=['Gini', 'T'], figsize=(15,4))
```

The time paths of the Gini and the Theil coefficients show striking
similarities, and, at first glance, this might suggest that the indices are
substitutes for one another. As we shall see below, however, each index has
properties that lend themselves to particular spatial extensions that provide
important complementarities.

## Personal versus Regional Income
There is a subtle but important distinction between the study of personal and
regional income inequality. To see this, we first need to express the
relationships between the two types of inequality. Consider a country composed
of $N$ individuals who are distributed over $m$ regions. Let $Y_l$ denote the
income of individual $l$. Total personal income in region $i$ is given as $Y_i =
\sum_{l \in i} Y_l$. Per-capita income in region $i$ is $y_i = \frac{Y_i}{N_i}$,
where $N_i$ is the number of individuals in region $i$.

At the national level,  the coefficient of variation as an index of  interpersonal income inequality would be:

$$CV_{nat} = \sqrt{\frac{\sum_{l=1}^N (Y_l - \bar{y})^2}{N}}$$

where $\bar{y}$ is national per-capita income. The key component here is the sum
of squares term, and unpacking this sheds light on personal versus regional
inequality question:


$$TSS = \sum_{l=1}^N (Y_l - \bar{y})^2$$

Focusing on an individual deviation: $\delta_l = Y_l - \bar{y}$, this is the contribution to inequality associated with individual $l$. We can break this into two components:

$$\delta_l = (Y_l - y_i) +  (y_i - \bar{y})$$

The first term is the difference between the individual's income and per-capita income in the individual's region of residence, while the second term is the difference between the region's per capita income and average national per capita income.

In regional studies, the intraregional personal income distribution is typically
not available. As a result, the assumption is often made that intraregional
personal inequality is zero. In other words, all individuals in the same region
have identical incomes. With this assumption in hand, the first term vanishes:
$Y_l -y_i = 0$, leading to: **FOOTNOTE:** It should also be noted that even at
the national scale, the analysis of interpersonal income inequality also relies
on aggregate data grouping individuals into income cohorts. See, for example,
Piketty, T. and E. Saez (2003) "Income inequality in the United States,
1913-1998", The Quarterly Journal of Economics, 118: 1-41.

$$
\begin{aligned}
TSS &= \sum_{l=1}^N (Y_l - \bar{y})^2 \\
    &= \sum_{l=1}^N \delta_l^2 \\
    &= \sum_{l=1}^N ((Y_l - y_i) +  (y_i - \bar{y}))^2 \\
    &= \sum_{l=1}^N (0 +  (y_i - \bar{y}))^2 \\
    &= \sum_{i=1}^m\sum_{l \in i}  (y_i - \bar{y})^2 \\
    &= \sum_{i=1}^m  [N_i(y_i - \bar{y})]^2
\end{aligned}
$$
This means that each individual in a region has an equal contribution to the
overall level of national interpersonal inequality, given by $(y_i - \bar{y})$,
while the region in question contributes $N_i(y_i - \bar{y})$. While it may seem
that the assumption of zero intraregional interepersonal income inequality is
overly restrictive, it serves to isolate the nature of interregional income
inequality. That is, inequality between places, rather than inequality between
people within those places. In essence, this strategy shifts the question up one
level in the spatial hierarchy by aggregating micro-level individual data to
areal units.



<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
## Spatial Inequality
<!-- #endregion -->
The analysis of regional income inequality is distinguished from the analysis of
national interpersonal income inequality in its focus on spatial units. As
regional incomes are embedded in geographical space, it is important to consider
the special nature of spatial data. In the regional inequality literature this
has been approaches in a number of ways.

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
### Spatial Autocorrelation
To get some insights on the spatial properties of regional income data, we can
turn to global measures of spatial autocorrelation that we encountered in
chapter XX. We use a queen spatial weights matrix to calculate Moran's I for
each year in the sample.
<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
wq = pysal.lib.weights.Queen.from_dataframe(pci_df)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
wq.n
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
MI = [pysal.explore.esda.moran.Moran(pci_df[str(y)], wq) for y in years]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res = np.array([ (mi.I, mi.p_sim) for mi in MI])
```
```python
res.shape
```

```python
res
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res_df = gini_df
res_df['I'] = res[:,0]
res_df['I pvalue'] = res[:,1]
_ = res_df[["Gini", "T", "I", "I pvalue"]].plot(subplots=True, figsize=(15,6))
```
```python
res_df.columns
```

Several patterns emerge from the time series of Moran's I. First, the is a secular decline in the value of Moran's I. Second, despite this decline,  there is never a year in which the spatial autocorrelation is not statistically significant. In other words, there is a strong spatial structure in the distribution of regional incomes that needs to be accounted for when focusing on inequality questions.
<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->

### Decompositional Approaches

A popular approach to the analysis of inequality is to group the observations into mutually exclusive and exhaustive subsets in order to understand how much of the inequality is due to differences between members of the same subset versus between observations from different subsets. In the personal income literature, the groups have been defined in a number of ways: male vs. female, age cohorts, occupation types, race, etc. In regional applications, small areas are grouped into larger sets such that the resulting sets are spatially defined. **Add notions of connected components here and link to other chapters where appropriate.**


<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
### Regional Inequality Decompositions
One approach to incorporating spatial considerations into regional inequality extends this decompositional approach by using space to define the subgroups. This can be seen using   Theil's $T$, which we encountered previously and decomposing the index  into so called *between* and *within* regional inequality components.
<!-- #endregion -->

Applied to a collection of observations on  per capita incomes for $m$ regional economies: $y = \left( y_1, y_2, \ldots, y_m \right)$, which are 
are grouped into $\omega$ mutually exclusive regions such that $\sum_{g=1}^{\omega} m_g=m$, where $m_g$ is the number of areas
assigned to region $g$, Theil's index from above can be rewritten as: 


$$
\begin{align}
T &= \sum_{i=1}^m \left( \frac{y_i}{\sum_{i=1}^m y_i} \ln \left[ m \frac{y_i}{\sum_{i=1}^m y_i}\right] \right) \\
  &= \left[ \sum_{g=1}^{\omega} s_{g} log(\frac{m}{m_g} s_g)  \right] + \left[ \sum_{g=1}^{\omega} s_g \sum_{i \in g} s_{i,g} log(m_g s_{i,g}) \right] \\
  &= B + W \\
\end{align}
$$

where $s_g = \frac{\sum_{i \in g} y_i}{\sum_i y_i}$, and   $s_{i,g} = y_i / \sum_{i \in g} y_i$. 

The first term is the between regions inequality component, and the second is the within regions inequality component.
The within region term is a weighted average of inequality between economies belonging to the same region.
Similar to what is done above for the case of interpersonal inequality, the
estimate of the between region (group) component of the decomposition is based on
setting the incomes of all economies (individuals )belonging to a region (group) equal to that of the
regional (group) average of these per capita incomes. Now, however, intraregional
inequality between economies within the same region is explicitly considered in
the second  component.


<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
**Note:** The regional decomposition does not involve weighting the regions by their respective population. See  [Gluschenko (2018)](https://webvpn.ucr.edu/+CSCO+0075676763663A2F2F6A6A6A2E676E6171736261797661722E70627A++/doi/full/10.1080/17421772.2017.1343491) for further details. 

<!-- #endregion -->

































```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pci_df.columns
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pandas.unique(pci_df['Region'])
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pci_df.plot(column='Region', categorical=True, linewidth=0.1)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
region_df = pci_df.dissolve(by='STATEFP')
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pandas.unique(region_df.Region)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
region_df.plot(column='Region', categorical=True)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
for region in range(1, 9):
    pci_df[pci_df.Region==region].plot()

```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
region_names = ["New England",
               'Mideast', 'Great Lakes', 'Plains',
               'Southeast', 'Southwest', 'Rocky Mountain',
                'Far West']
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
pci_df.groupby('Region').mean()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
regimes = pci_df['Region']
ys = [str(y) for y in years]
```

## Time series plot of the regional means (WIP)

```python
rdf = pci_df.groupby('Region').mean().transpose()
```

```python
rdf.shape
```

```python
rdf.columns
```

```python

```

```python

```

```python
theil_dr = pysal.explore.inequality.theil.TheilD(pci_df[ys].values, regimes)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
theil_dr.bg
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res_df['bgr'] = theil_dr.bg
res_df['wgr'] = theil_dr.wg
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res_df.plot(subplots=True, figsize=(15,6))
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}

numpy.random.seed(12345)
theil_drs = pysal.explore.inequality.theil.TheilDSim(pci_df[ys].values, regimes, 999)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res_df['bgr_pvalue'] = theil_drs.bg_pvalue
res_df.plot(subplots=True, figsize=(15,6))
```

```python
res_df['bgr_share'] = res_df['bgr'] / res_df['T']
res_df['bgr_share'].plot()
```

## Decomposition Using States

```python
theil_ds = pysal.explore.inequality.theil.TheilD(pci_df[ys].values, pci_df['STATEFP'])
```

```python
theil_ds.T
```

```python
theil_ds.bg
```

```python
res_df['bgs_share'] = theil_ds.bg / theil_ds.T
res_df['bgs'] = theil_ds.bg
```

```python
res_df[['bgr', 'bgs', 'T']].plot()
```

```python
res_df[['bgr_share', 'bgs_share']].plot()
```

```python
res_df[['bgr_share', 'bgs_share','T']].corr()
```

- inequality between the states is larger than inequality between regions
- inequality within states is smaller than inequality within regions
- time series of the between component are similar for regional and state partitions of the counties
- correlation of between share and overall inequality is higher at the state level than the region level

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
### Intraregional inequality trends

Theil applied to each of the regions.

Plot the eight series
<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
region_df.plot(column='Region', categorical=True)
```

```python
fig = plt.figure(figsize=(15,12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 9):
    ax = fig.add_subplot(2, 4, i)
    rdf = pci_df[pci_df.Region==i]
    rdf.plot(ax=ax, linewidth=0.1)
    
```

```python
region_names = ["New England",
               'Mideast', 'Great Lakes', 'Plains',
               'Southeast', 'Southwest', 'Rocky Mountain',
                'Far West']
results = []
table = []
for region in range(1, 9):
    rdf = pci_df[pci_df.Region==region]
    #rdf.plot()
    print(region, len(pandas.unique(rdf.STATEFP)), rdf.shape[0])
    table.append([region, region_names[region-1], len(pandas.unique(rdf.STATEFP)), rdf.shape[0]])
    #results.append(pysal.explore.inequality.theil.TheilDSim(rdf[ys].values, rdf.STATEFP,999))
    
```

```python
summary = pandas.DataFrame(table)
```

```python
summary
```

```python
summary.columns = [ "Region", "Name", "States", "Counties"]
```

```python
summary
```

### Intra regional inequality
We can take a closer look at the within region inequality component by dissagregating the total value from XX into that occuring within each of the 8 regions. This can be done by calculating the global Theil index on the counties belonging to a given region.

```python
region_names = ["New England",
               'Mideast', 'Great Lakes', 'Plains',
               'Southeast', 'Southwest', 'Rocky Mountain',
                'Far West']
results = []
table = []
for region in range(1, 9):
    rdf = pci_df[pci_df.Region==region]
    #rdf.plot()
    print(region, len(pandas.unique(rdf.STATEFP)), rdf.shape[0])
    table.append([region, region_names[region-1], len(pandas.unique(rdf.STATEFP)), rdf.shape[0]])
    results.append(pysal.explore.inequality.theil.TheilDSim(rdf[ys].values, rdf.STATEFP,999))
    
```

```python
r1 = results[0]
```

```python
r1.bg_pvalue
```

```python
r1.bg
```

```python
Tr = pandas.DataFrame([result.T for result in results]).transpose()
```

```python
Tr.head()
```

```python
region_names = ["New England",
               'Mideast', 'Great Lakes', 'Plains',
               'Southeast', 'Southwest', 'Rocky Mountain',
                'Far West']
```

```python
Tr = Tr.rename(columns=dict([(i,name) for i,name in enumerate(region_names)]))
```

```python
pinned0 = Tr.divide(Tr.ix[0])
```

```python
import pandas as pd

df = pinned0
markers = [ "+", "8", "^", "p", ".", ">", "1", '2']
ax = df.plot(kind='line', figsize=(15, 12))
for i, line in enumerate(ax.get_lines()):
    line.set_marker(markers[i])

# for adding legend
ax.legend(ax.get_lines(), df.columns, loc='best')
```
Unpacking the intraregional inequality term reveals that the original decomposition of inequality into within and between regions actually masks a great deal of heterogeneity in the internal inequality dynamics across the eight regions. Put another way, the overall trend in the aggregate within region component above is an average of the trends exhibited in each of the eight regions. There are two distinct groups of regions in this regard. The first consists of regions where the inequality between counties within each region has been increasing over the sample period. This group is composed of the New England,  Mideast, Far West, and Rocky Mountains regions. The second group are those regions where intraregional inequality has remained stable, or even decreased, over time. The Great Lakes, Southeast, and Southwest regions compose this group. The one outlier region is the Plains which does not fall neatly into either of these two groups.

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
theil_ds = pysal.explore.inequality.theil.TheilDSim(rdf[ys].values, rdf.STATEFP, 999)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
len(theil_ds.bg[0])
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
theil_ds.bg[0]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
theil_ds.bg_pvalue
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
theil_ds.T
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
theil_ds.bg/theil_ds.T
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
wgp_ds = theil_ds.wg[0]/theil_ds.T
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
wgp_dr = theil_ds.wg[0]/theil_ds.T
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
wgp_dr.shape
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res_df.shape
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
wgp_ds.shape = (49, 1)
res_df['wgp_ds'] = wgp_ds
wgp_dr.shape = (49, 1)

res_df['wgp_dr'] = wgp_dr

res_df.plot(subplots=True, figsize=(15,6))
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res_df.plot(y=['wgp_ds','wgp_dr'])
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
res_df.head()
```

```python
theil_ds.T
```

```python
theil_ds.T
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->


### Spatializing Classic Measures
<!-- #endregion -->

We now turn to two newer spatial analytics that extend a selction of the classic (a-spatial) inequality measures above to introduce a spatially explicit focus.

#### Spatial Gini

The first spatial extension was introduced by Rey and Smith (2013) and is designed to consider the role of adjacency in a decomposition of the Gini index of inequality. More specifically, The Gini in mean xxx is rewritten.

- point out theil decomposition ignores spatial interactions between counties, both within a region and between region
- spatial gini as a complement

```python
from pysal.explore.inequality.gini import Gini_Spatial
```

```python
wq.transform = 'B'
```

```python
wq.pct_nonzero
```

```python
wq.s0
```

```python
wq.n * wq.n
```

```python
wq.n
```

```python
W = wq.full()[0]
```

```python
W.sum()
```

```python
9467929 * wq.pct_nonzero / 100
```

```python
18358/2
```

```python
wq.histogram
```

```python
hist = np.array(wq.histogram)
```

```python
(hist[:,0]*hist[:,1]).sum()
```

```python
gs69 = Gini_Spatial(pci_df['1969'], wq)
```

```python
gs69.g
```

```python
gs69.p_sim
```

```python
gs = [Gini_Spatial(pci_df[y], wq) for y in ys]
```

```python
gs_array = np.array([(gsi.e_wcg, gsi.wcg, gsi.z_wcg, gsi.p_sim) for gsi in gs])
```

```python
gs_array
```

```python

```

```python
gs69.wcg
```

```python
res_df['z_wcg'] = gs_array[:,2]
res_df.plot(y=['z_wcg'])
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
#### Spatial 20:20
<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
ranks = pci_df.rank()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
for year in years:
    pci_df["{}_rank".format(year)] = pci_df[str(year)].rank(method='first')
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
ridx_20 = int(.2 * 3077)
ridx_80 = int(.8 * 3077)
ridx_20, ridx_80
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
df = pci_df
df['1969_rank']
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
df[df['1969_rank']==615]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
df.index[df['1969_rank']==615].tolist()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
idx_20 = []
idx_80 = []
for year in years:
    column = "{}_rank".format(year)
    idx_20_i = df.index[df[column]==ridx_20]
    idx_20.extend(idx_20_i)
    idx_80_i = df.index[df[column]==ridx_80]
    idx_80.extend(idx_80_i)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
len(idx_20)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
len(years)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
idx_20[0], idx_80[0]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
df.loc[[8999, 2561], :].plot()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
df.loc[[idx_20[-1], idx_80[-1]], :].plot()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
import geopandas as gpd
from shapely.geometry import Point,Polygon
geom=[Point(xy) for xy in zip([117.454361,117.459880],[38.8459879,38.846255])]
ldf=gpd.GeoDataFrame(geometry=geom,crs={'init':'epsg:4326'})
ldf.to_crs(epsg=3310,inplace=True)
l=gdf.distance(ldf.shift())
print(l)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
ldf = df.to_crs({'init':'epsg:4326'})
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
ldf.loc[[idx_20[-1], idx_80[-1]], :].plot()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
ldf.to_crs(epsg=3310,inplace=True)

```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
ldf.loc[[idx_20[-1], idx_80[-1]], :].plot()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
small = ldf.loc[[idx_20[-1], idx_80[-1]], :]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
small
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
small.distance(small.shift()).values[-1]
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
small.geometry.centroid
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
idxs = zip(idx_20, idx_80)
distances = []
for idx in idxs:
    o,d = idx
    #print(o,d, idx)
    pair = df.loc[idx, :]
    d = pair.distance(pair.shift()).values[-1]
    distances.append(d)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
distances
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
idxs = zip(idx_20, idx_80)

len(distances), len(years), len(list(idxs))
gini_df['s_dist'] = numpy.array(distances)
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
gini_df.plot(y=["s_dist"])
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
idxs = numpy.array(list(zip(idx_20, idx_80)))
idxs
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
df.loc[idxs[:,0],:].centroid.plot()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
df.loc[idxs[:,1],:].centroid.plot()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
df.loc[idxs[:,0],:].centroid.plot(ax=ax, color='r')
df.loc[idxs[:,1],:].centroid.plot(ax=ax, color='b')
ax.set_axis_off()


```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
df.loc[idxs[:,0],:].centroid.plot(ax=ax, color='r')
df.loc[idxs[:,1],:].centroid.plot(ax=ax, color='b')
gdf.plot(ax=ax,edgecolor='gray', alpha=0.2)
ax.set_axis_off()


```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
## Rank paths
<!-- #endregion -->

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
from shapely.geometry import LineString
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
ls20 = geopandas.GeoSeries(LineString(df.loc[idxs[:,0],:].centroid.tolist()))
ls80 = geopandas.GeoSeries(LineString(df.loc[idxs[:,1],:].centroid.tolist()))
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
ls20.plot(ax=ax, color='r', label='20p')
ls80.plot(ax=ax, color='b')
#gdf.plot(ax=ax,edgecolor='gray', alpha=0.2)
plt.title('Rank Paths (80pct Blue, 20pct Red)')
ax.set_axis_off()


```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
columns = 4
rows = 12
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(15,15))
year=0
for i,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        axes.set_title('{},{}'.format(i,j))
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        if year < 43:
            ls20 = geopandas.GeoSeries(LineString(df.loc[idxs[:,0],:].centroid.tolist()[year:year+5]))
            ls20.plot(ax=axes, color='r', label='20p')
            ls80 = geopandas.GeoSeries(LineString(df.loc[idxs[:,1],:].centroid.tolist()[year:year+5]))
            ls80.plot(ax=axes, color='b', label='20p')
    
        year += 1
#         axes.plot(you_data_goes_here,'r-')
        #axes.set_aspect('equal')
plt.show()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
columns = 4
rows = 12
fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=(15,15))
year=2
for i,ax_row in enumerate(ax_array):
    for j,axes in enumerate(ax_row):
        axes.set_title('{},{}'.format(i,j))
        axes.set_yticklabels([])
        axes.set_xticklabels([])
        if year < 48:
            ls20 = geopandas.GeoSeries(LineString(df.loc[idxs[:,0],:].centroid.tolist()[0:year]))
            ls20.plot(ax=axes, color='r', label='20p')
            ls80 = geopandas.GeoSeries(LineString(df.loc[idxs[:,1],:].centroid.tolist()[0:year]))
            ls80.plot(ax=axes, color='b', label='20p')
    
        year += 1
#         axes.plot(you_data_goes_here,'r-')
        #axes.set_aspect('equal')
plt.show()
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}
for i in range(2,48):
    #print(0,i)
    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
    ls20 = geopandas.GeoSeries(LineString(df.loc[idxs[:,0],:].centroid.tolist()[0:i]))
    ls20.plot(ax=ax, color='r', label='20p')
    ls80 = geopandas.GeoSeries(LineString(df.loc[idxs[:,1],:].centroid.tolist()[0:i]))
    ls80.plot(ax=ax, color='b', label='20p')
    
    #ls80.plot(ax=ax, color='b')
    #gdf.plot(ax=ax,edgecolor='gray', alpha=0.2)
    title = 'Percentile Paths 1969-{} (80p Blue, 20p Red)'.format(1969+i-1)
    plt.title(title)
    ax.set_axis_off()

    
```

```python ein.tags="worksheet-0" ein.hycell=false slideshow={"slide_type": "-"} jupyter={"outputs_hidden": false}

```

---

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
<!-- #endregion -->
