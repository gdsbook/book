---
jupyter:
  jupytext:
    cell_metadata_json: true
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python [conda env:analysis]
    language: python
    name: conda-env-analysis-py
---

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
# Spatial Inequality


## Introduction

Social and economic inequality is often at the top of policy makers' agendas.
It has always drawn considerable attention in academic circles. Currently, 
the world is extremely unequal, stratified, and segregated; no matter the 
society or economic system, inequality is at a high point. Much of the focus
has been on *interpersonal income inequality*, yet there is a growing recognition
that the question of *interregional income inequality* requires further 
attention as the growing gaps between poor and rich regions have been identified
as key drivers of political polarization in developing and developed countries
{cite}`Rodriguez_Pose_2018`.

Indeed, while the two literatures (personal and regional inequality) are
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


For this chapter, we focus on the case of the United States from 1969 to 2017. Specifically, we will analyze median incomes at the county level, examining the trends both in terms of how individual counties, states, or regions get richer or poorer, as well as how the overall distribution of income moves, skews, or spreads out. 

<!-- #endregion -->

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
---
<!-- #endregion -->
```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
%matplotlib inline

import seaborn
import pandas
import geopandas
import pysal
import numpy
import mapclassify
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 5
```

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
pci_df = geopandas.read_file('../data/us_county_income/uscountypcincome.gpkg')
pci_df.head()
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
Inspection of the head of the data frame reveals that the years appear as columns in the data set, together with information about the particular record.
This format is an example of a [*wide* longitudinal data set](https://en.wikipedia.org/wiki/Wide_and_narrow_data).
In wide-format data, each column represents a different time period, meaning that each row represents a set of measurements made about the same "entity" over time time (as well as any unique identifying information about that entity.)
This contrasts with a *narrow* or *long* format, where each row describes an entity at a specific point in time. 
Long data results in significant duplication for records and is generally worse for data storage. However, long form data is sometimes a more useful format when manipulating and analyzing data, as {cite}`Wickham_2014` discusses. Nonetheless, when analyzing *trajectories*, that is, the paths that entities take over time, wide data is more useful, and we will use that here. 

In this data, we have 3076 counties across 49 years, as well as 28 extra columns that describe each county. 
<!-- 
ljw: we (1) don't discuss wide vs. long anywhere else in the book and (2) don't ever show a wide-to-long pivot using something like pandas.wide_to_long... We also don't need the LineCode filtering, since the final dataframe is already filtered by LineCode so that all LineCode==3.
-->

<!-- #endregion -->

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
pci_df.shape
```

As an example, we can see the first few years for Jackson Counti, Mississippi below:

```python
pci_df.query('NAME == "Jackson" & STATEFP == "28"')
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
## Global Inequality

We begin our examination of inequality by focusing on several global measures of income inequlity. Here, "global" means that the measure is concerned with the overall nature of inequality within the income distribution. That is, these measures focus on the direct disparity between rich and poor, considering nothing about where the rich and poor live. Several classic measures of inequality are available for this purpose. 

In general terms, measures of inequality focus on the dispersion present in an income distribution. In the case of regional or spatial inequality, the distributions describe the average or per-capita incomes for spatial units, such as for counties, census tracts, or regions. For our US counties data, we can visualize the distribution of per capita incomes for the first year in the sample as follows:
<!-- #endregion -->

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
seaborn.set_theme(style='whitegrid')
seaborn.histplot(x=pci_df['1969'], kde=True)
```


Looking at this distribution, notice that the right side of the distribution is much longer than the left side. This long right tail is a prominent feature of the distribution, and is common in the study of incomes, as it reflects the fact that within a single income distribution, the super-rich are generally much more wealthy than the super-poor are deprived. 

A key point to keep in mind here is that the unit of measurement in this data is a spatial aggregate of individual incomes. Here, we are using the per-capita incomes for each county. By contrast, in the wider inequality literature, the observational unit is typically a household or individual. In the latter distributions, the degree of skewness is often more pronounced. This difference arises from the smoothing that is intrinsic to aggregation: the regional distributions are based on averages obtained from the individual distributions, and so the extremely high-income individuals are averaged with the rest of their county. 


The kernel density estimate (or histogram) is a powerful visualization device that captures the overall morphology of the *feature* distribution for this measure of income. At the same time, the density is silent on the underlying *geographic distribution* of county incomes. We can look at this second view of the distribution using a choropleth map. To construct this, we can use the standard `geopandas` plotting tools. First, though, we will clean up the data for mapping. We do this by first setting the coordinate reference system (as it is currently missing) and then by re-projecting the data into a coordinate reference system suitable for mapping, the Albers Equal Area projection for North America:  

```python
pci_df = (pci_df.set_crs(epsg=4326) # US Census default projection
                .to_crs(epsg=5070)) # Albers Equal Area North America
```

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
pci_df.plot(column='1969', scheme='Quantiles', 
            legend=True, edgecolor='none',
            legend_kwds={'loc': 'lower left'}, 
            figsize=(12, 12))
plt.title('Per Capita Income by County, 1969')
plt.show()
```

The choropleth and the kernel density provide different visual depictions of the distribution of county incomes. The kernel density estimate is a *feature*-based representation, and the map is a *geographic* representation. Both are useful for developing an understanding of the overall. To gain more specific insights on the level of inequality in the distribution, we'll discuss a few inequality indices common in econometrics. 


### 20:20 Ratio

One commonly used measure of inequality in a distribution is the so called 20:20 ratio, which is defined as the ratio of the incomes at the 80th percentile over that at the 20th percentile: 
<!-- #endregion -->
```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
top20, bottom20 = pci_df['1969'].quantile([.8, .2])
```


```python
top20/bottom20
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
In 1969 the richest 20% of the counties had an income that was 1.5 times the poorest 20% of the counties. The 20:20 ratio has the advantage of being robust to outliers at the top and the bottom of the distribution. 

We can examine the dynamics of this global inequality measure by creating a simple function to apply to all years in our time series:
<!-- #endregion -->

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
def ineq_2020(values):
    top20, bottom20 = values.quantile([.8, .2])
    return top20/bottom20
```

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
years = numpy.arange(1969, 2018).astype(str)
ratio_2020 = pci_df[years].apply(ineq_2020, axis=0)
ax = plt.plot(years, ratio_2020)
figure = plt.gcf()
plt.xticks(years[::2])
plt.ylabel("20:20 ratio")
plt.xlabel("Year")
figure.autofmt_xdate(rotation=90)
plt.show()
```
<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
The ratio has a U-shaped pattern over time, bottoming out around 1994 after a long decline. Post 1994, however, the 20:20 ratio indicates there is increasing inequality up until 2013, where there is a turn towards lower income inequality between the counties.
<!-- #endregion -->

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
For other classic measures of inequality, we will use the `inequality` package from `pysal`:
<!-- #endregion -->

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
from pysal.explore import inequality
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
### Gini Index

The Gini index is a longstanding measure of inequality. It can be derived in a number of ways. One of the simplest ways to define the Gini Curve is *via* the cumulative wealth distribution, called the *Lorenz* curve. To plot a Lorenz curve, the cumulative share of wealth is plotted against the share of the population that owns that wealth. For example, in an extremely unequal society where few people own nearly all the wealth, the Lorenz curve increases very slowly at first, then skyrockets the wealthiest people are included. 

In contrast, a "perfectly equal" society would look like a straight line connecting $(0,0)$ and $(1,1)$. This is called the *line of perfect equality*, and represents the case where $p$% of the population owns exactly $p$% of the wealth. For example, this might mean that 50% of the population earns exactly 50% of the income, or 90% of the population owns 90% of the wealth. The main idea is that the share of wealth or income is exactly proportional to the share of population that owns that wealth or earns that income, which occurs only when everyone has the same income or owns the same amount of wealth. 

With this, we can define the Gini index as the ratio of the area between the line of perfect equality and the Lorenz curve for a given income or wealth distribution, standardized by the area under the line of perfect equality (which is always $\frac{1}{2}$). Thus, the Gini index is a measure of the gap between a perfectly equal society and the observed society over every level of wealth/income. 

We can construct one of the Lorenz curves for 1969 by first computing the share of population below each observation:
<!-- #endregion -->

```python
N = len(pci_df)
share_of_population = numpy.arange(1, N+1)/N
```

Then, we need to find out how many incomes are *smaller than* the values for each observation. Empirically, this can be computed by first sorting the incomes:

```python
incomes = pci_df['1969'].sort_values()
```

Then, we need to find the overall percentage of income at (or below) each value. To do this, we need to first compute what percentage of the total income each income represents:

```python
shares = incomes / incomes.sum()
```

Then, we construct the *cumulative sum* of these shares, which reflects the sum of all of the shares of income up to the current one:
$$ \texttt{cumsum(v, k)} = \sum_{i=1}^k v_i$$

This starts at $0$ and reaches $1$ once the last share is included. 

```python
cumulative_share = shares.cumsum()
```

With this, we can plot both the Lorenz curve and the line of perfect equality:

```python
f, ax = plt.subplots()
ax.plot(share_of_population, cumulative_share, label='Lorenz')
ax.plot((0,1), (0,1), color='r', label='Equality')
ax.set_xlabel('Share of population')
ax.set_ylabel('Share of income')
ax.legend()
plt.show()
```
The blue line is the Lorenze curve for county incomes in 1969. The Gini index is the area between it and the 45-degree line of equality shown in red, all standardized by the area underneath the line of equality.

To examine how inequality evolves over time, we will plot the Lorenz curves for each year. To do this, we can create a function that will compute the Lorenz curve for arbitrary inputs. 

```python
def lorenz(y):
    y = numpy.asarray(y)
    incomes = numpy.sort(y)
    income_shares = (incomes / incomes.sum()).cumsum()
    N = y.shape[0]
    pop_shares = numpy.arange(1, N+1)/N
    return pop_shares, income_shares
```

Then, we use the same method as before to compute lorenz curves (as a set of population shares that correspond to income shares) in each time:
```python
lorenz_curves = pci_df[years].apply(lorenz, axis=0)
```

Practically, this becomes a dataframe with columns for each year. Rows contain the population shares (or income shares) as lists.

```python
lorenz_curves.head()
```

By iterating over the columns of this dataframe, we can make a plot of the lorenz curves for each year:

```python
f, ax = plt.subplots()
ax.plot((0,1),(0,1), color='r')
for year in lorenz_curves.columns:
    pop_shares, inc_shares = lorenz_curves[year].values
    ax.plot(pop_shares, inc_shares, color='k', alpha=.05)
```

The compression of the Lorenze curves makes it difficult to ascertain the temporal pattern in inequality. Focusing explicilty on the Gini coefficients may shed more light on the evolution of inequality over time. 

To express this, we first show that the Gini coefficient is computed by the `Gini` class in `inequality`:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
g69 = inequality.gini.Gini(pci_df['1969'].values)
```

To actually get the coefficient, we extract the `g` property. Here, the Gini coefficient in 1969 was .13:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
g69.g
```

To do this for all years, we can use a similar pattern as we have before. First, define a function to compute the quantity of interest. Then, apply the function across the dataframe:

```python
def gini(values):
    values = numpy.asarray(values)
    return inequality.gini.Gini(values).g
```

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
inequalities = pci_df[years].apply(gini, axis=0).to_frame('gini')
```

Then, we have the results as follows:

```python
inequalities.head()
```

<!-- #region {"ein.hycell": false, "ein.tags": "worksheet-0", "jupyter": {"outputs_hidden": false}, "slideshow": {"slide_type": "-"}} -->
And, to make a plot, we use the standard `pandas` methods, which reveals a similar pattern to the 20:20 ratio above:
<!-- #endregion -->

```python
inequalities.plot()
```

### Theil's index

A third commonly used measure of inequality is Theil's $T$ given as:
$$T = \sum_{i=1}^m \left( \frac{y_i}{\sum_{i=1}^m y_i} \ln \left[ m \frac{y_i}{\sum_{i=1}^m y_i}\right] \right)$$
where $y_i$ is per-capita income in area $i$ among $m$ areas. Conceptually, this metric is related to the entropy of the income distribution, measuring how evenly-distributed incomes are across the population. However, elucidating the deeper links between these concepts is beyond the scope of this chapter. 

We can calculate the Theil index using the same methods as above:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
def theil(y):
    y = numpy.asarray(y)
    return inequality.theil.Theil(y).T
```

```python
inequalities['theil'] = pci_df[years].apply(theil, axis=0)
```

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
inequalities.plot(subplots=True, figsize=(15,6))
```

The time paths of the Gini and the Theil coefficients show striking
similarities. At first glance, this might suggest that the indices are
substitutes for one another. However, they are not perfectly correlated: 

```python
seaborn.regplot(x='theil', y='gini', data=inequalities)
```

Indeed, as we shall see below, each index has
properties that lend themselves to particular spatial extensions that work in complementary ways. We need both (and more) for a complete picture. 

## Personal versus Regional Income
There is a subtle but important distinction between the study of personal and
regional income inequality. To see this, we first need to express the
relationships between the two types of inequality. Consider a country composed
of $N$ individuals who are distributed over $m$ regions. Let $Y_l$ denote the
income of individual $l$. Total personal income in region $i$ is given as $Y_i =
\sum_{l \in i} Y_l$. Per-capita income in region $i$ is $y_i = \frac{Y_i}{N_i}$,
where $N_i$ is the number of individuals in region $i$.

At the national level,  the coefficient of variation in incomes could be used as an index of interpersonal income inequality. This would be:

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
$Y_l -y_i = 0$, leading to:[^reg] 

[^reg]: It should also be noted that even at the national scale, the analysis of interpersonal income inequality also relies on aggregate data grouping individuals into income cohorts. See, for example, {cite}`Piketty_2003`.

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
turn to global measures of spatial autocorrelation that we encountered earlier in the book. We use a queen spatial weights matrix to calculate Moran's I for
each year in the sample.
<!-- #endregion -->

```python
from pysal.explore import esda
from pysal.lib import weights
```

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
wq = weights.Queen.from_dataframe(pci_df)
```

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
wq.n
```

Then, as before, we create a function that returns the results we need from each statistic. Here, we will also keep the $p$-value for the Moran statistic, which indicates whether it is statistically significant under the null hypothesis that incomes are randomly distributed geographically. 

```python
def moran(y, w=wq):
    mo = esda.Moran(y, w=w)
    return mo.I, mo.p_sim
```

Using this function, we compute each of these statistics.

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
moran_stats = pci_df[years].apply(moran, axis=0)
```

Further, we then re-arrange them from being one long list of ($I$, $p$-value) into two lists that solely contain $I$ and $p$-values:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
moran_coefs, p_values  = moran_stats.values
```
Finally, we will store these as columns in the `inequalities` dataframe:

```python
inequalities['moran'] = moran_coefs
inequalities['moran_pvalue'] = p_values
```

To show the overall inter-relationships between these statistics, we make another plot below:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
inequalities.plot(subplots=True, figsize=(15,12))
plt.show()
```
Several patterns emerge from the time series of Moran's I. First, the is a long-term decline in the value of Moran's I. This suggests a gradual decline in the geographic structure of inequality with two implications: (a) per capita incomes are now less similar between nearby counties and (b), this has been consistently declining, regardless of whether inequality is high or low. 

Second, despite this decline, there is never a year in which the spatial autocorrelation is not statistically significant. In other words, there is a strong geographic structure in the distribution of regional incomes that needs to be accounted for when focusing on inequality questions.
<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->

### Decompositional Approaches

One common objection to the analysis of inequality in aggregate is about confounding. Inequality can be driven by differences between groups and not by discrepancies in income between similar individuals. That is, there is always the possibility that observed inequality can be "explained" by a confounding variate, such as age, sex, or education. For example, income differences between older and younger people can "explain" a large part of the societal inequality in wealth: older people have much longer to acquire experience, and thus are generally paid more for that experience. Younger people do not have as much experience, so young people (on average) have lower incomes than older people. 

To combat this issue, it is often useful to *decompose* inequality indices into constituent groups. This allows us to understand how much of inequality is driven by aggregate group differences and how much is driven by observation-level inequality. This also allows us to characterize how unequal each group is separately. In geographic applications, these groups are usually spatially defined, in that *regions* are contiguous geographic groups of observations. Thus, we will dicuss regional inequality decompositions in the following sections as a way to introduce geography into the study of inequality. 

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
### Regional Inequality Decompositions

One way to introduce geography into the analysis of inequality is to use geography to define groups for decompositions. Theil's $T$, which we encountered previously, can be decomposed using regions into so called *between* and *within* regional inequality components.
<!-- #endregion -->

To define this regional income decomposition, we first re-define our observations of per capita incomes for $m$ regional economies as $y = \left( y_1, y_2, \ldots, y_m \right)$. These are grouped into $\omega$ mutually exclusive regions. Formally, this means that when $m_g$ represents the number of areas assigned to region $g$, the total number of areas must be equal to the count of all the areas in each region: $\sum_{g=1}^{\omega} m_g=m$.[^mut-ex] With this notation, Theil's index from above can be rewritten to emphasize its between and within components:

[^mut-ex]: This would be violated, for example, if one area were in two regions. This area would get "double counted" in this total. 


$$
\begin{align}
T &= \sum_{i=1}^m \left( \frac{y_i}{\sum_{i=1}^m y_i} \ln \left[ m \frac{y_i}{\sum_{i=1}^m y_i}\right] \right) \\
  &= \left[ \sum_{g=1}^{\omega} s_{g} log(\frac{m}{m_g} s_g)  \right] + \left[ \sum_{g=1}^{\omega} s_g \sum_{i \in g} s_{i,g} log(m_g s_{i,g}) \right] \\
  &= B + W \\
\end{align}
$$

where $s_g = \frac{\sum_{i \in g} y_i}{\sum_i y_i}$, and   $s_{i,g} = y_i / \sum_{i \in g} y_i$. 

The first term is the between regions inequality component, and the second is
the within regions inequality component. The within regions term is a weighted
average of inequality between economies belonging to the same region. Similar
to what is done above for the case of interpersonal inequality, the estimate of
the between region (group) component of the decomposition is based on setting
the incomes of all economies (individuals) belonging to a region (group) equal
to that of the regional (group) average of these per capita incomes. Now,
however, intraregional inequality between economies within the same region is
explicitly considered in the second component.[^weight]

[^weight]: The regional decomposition does not involve weighting the regions by their respective population. See  {cite}`Gluschenko_2018` for further details. 

In our data, we record the United States Census Bureau regions, stored in the `Region` variable. These divide the country into eight regions:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
pci_df.plot(column='Region', categorical=True, linewidth=0.1, legend=True)
```

These regions' names are:


- New England
- Mideast
- Great Lakes
- Plains
- Southeast
- Southwest
- Rocky Mountain
- Far West


We can map these region names to a new variable in our data and make a plot:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
region_names = ["New England",
                "Mideast", 
                "Great Lakes", 
                "Plains",
                "Southeast", 
                "Southwest", 
                "Rocky Mountain",
                "Far West"]
```

```python
pci_df['Region_Name'] = pci_df.Region.apply(lambda region: region_names[region - 1])
```

```python
pci_df.plot('Region_Name', linewidth=0, legend=True, 
            legend_kwds=dict(bbox_to_anchor=(1.2,.5)))
```

To see the income changes for each region separately, we can use standard `pandas` methods. First, we can group the data by region and compute the mean of each column. For good measure, we will also select only the year columns of interest:

```python
rmeans = pci_df.groupby(by='Region_Name').mean()[years]
```

```python
rmeans
```

Once we transpose this dataframe, we can plot the region sequences as distinct variables:

```python
rmeans.T.plot.line()
```

## Regional Decomposition of Inequality


Now, we can compute the regional Theil decomposition using the `inequality` module of pysal:

```python
theil_dr = inequality.theil.TheilD(pci_df[years].values, pci_df.Region)
```

The `theil_dr` object has the between and within components stored in the `bg` and `wg` attributes. The `bg` stands for the "between" component, and "wg" for the within component. For example the "between" component for each year is computed is:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
theil_dr.bg
```

storing these components in our result dataframe as before:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
inequalities['theil_between'] = theil_dr.bg
inequalities['theil_within'] = theil_dr.wg
inequalities.drop('moran_pvalue', axis=1, inplace=True)
```

We can visualize them alongside our earlier results:

```python ein.hycell=false ein.tags="worksheet-0" jupyter={"outputs_hidden": false} slideshow={"slide_type": "-"}
inequalities.plot(subplots=True, figsize=(15,15))
plt.show()
```

Inference on these decompositions can be done using the `inequality.theil.TheilDSim` class, but we omit that here for brevity and report that, like the Moran's $I$, all of the Theil decompositions are statistically significant. Further, since the within and between components are interpreted as shares of the overall Theil index, we can compute the share of the Theil index due to the between-region inequality, and note that it also generally shares the same pattern, but does not see minima in the same places. The between-region share of inequality is at its lowest in the mid-2000s, not in the mid-1990s. This suggests that regional differences were very important in the 1970s and 80s, but this importance has been waning, relative to the inequality *within* US Census Regions. 

```python
inequalities['theil_between_share'] = inequalities['theil_between'] / inequalities['theil']
inequalities['theil_between_share'].plot()
plt.show()
```

<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->
<!-- #region {"ein.tags": "worksheet-0", "slideshow": {"slide_type": "-"}} -->


### Spatializing Classic Measures
<!-- #endregion -->

While regional decompositions are useful, they do not tell the whole story. Indeed, a "region" is just a special kind of geographical group; its "geography" is only made manifest through group membership: is the county "in" the region or not? This kind of "place-based" thinking, while geographic, is not necessarily *spatial*. It does not incorporate the notions of distance or proximity into the study of inequality; the geographical locations of the regions could be re-arranged without impact, so long as the group membership structure is maintained. While, arguably, shuffling regions around means they are no longer "regions," the statistical methods do not differentiate.

So, we will now turn to two newer methods for analyzing inequality that include more spatial elements. These "spatial" measures of inequality care directly about the 

#### Spatial Gini

The first spatial extension was introduced by {cite}`Rey_2012` and is designed to consider
the role of adjacency in a decomposition of the Gini index of inequality. One formulation for the Gini coefficient we discussed above focuses on the set of pairwise absolute differences in incomes:
$$G = \frac{\sum_i \sum_j \left | y_i - y_j \right|}{2 n^2 \bar{x}} $$

Focusing on the set of pairwise absolute differences in income, we can de-compose this into the set of differences between "nearby" observations and the set of differences among "distant" observations. This is the main conceptual point of the "Spatial Gini" coefficient. This decomposition works similarly to the regional decomposition of the Theil index:
$$
\sum_i \sum_j \left |y_i - y_j \right | =\sum_i \sum_j \underset{\text{near differences}}{\left( w_{ij} \left |y_i - y_j \right | \right )} + \underset{\text{far differences}}{\left( (1-w_{ij})  \left |y_i - y_j \right | \right )}
$$
In this decomposition, $w_{ij}$ is a binary variable that is $1$ when $i$ and $j$ are neighbors, and is zero otherwise. Recalling the spatial weights matrices from Chapter 4, this can be used directly from a spatial weights matrix.^[However, non-binary spatial weights matrices require a correction factor, and are not discussed here.] Thus, with this decomposition, the Spatial Gini can be stated as
$$G = \frac{\sum_i \sum_j w_{i,j}\left | x_i - x_j \right|}{2 n^2 \bar{x}} +   \frac{\sum_i \sum_j \left (1-w_{i,j} )| x_i - x_j \right|}{2 n^2 \bar{x}}$$
with the first term being the component among neighbors and the second term being the component among non-neighbors. The "spatial Gini", then, is the first component that describes the differences between nearby observations. 

The spatial Gini allows for a consideration of the spatial dependence in inequality. If spatial depenedence is very strong and positive, incomes are very similar among nearby observations, so the inequality of "near" differences will be small. Most of the inequality in the society will be driven by disparities in income between distant places. In contrast, when dependence is very weak (or even negative), then the two components may equalize. Inference on the spatial Gini can be based on random spatial permutations of the income values, as we have seen elsewhere in this book. This tests whether the distribution of the compoents are different from that obtained when incomes are randomly distributed across the map. 


The spatial Gini also provides a useful complement to the regional decomposition used in the Theil statistic. The latter does not consider pairwise relationships between observations, while the spatial Gini does. By considering the pairwise relationships between observations, the Gini coefficient is more sensitive, and can also be more strongly affected by small groups of significanatly wealthy observations.  

We can estimate spatial Gini coefficients using the `Gini_Spatial` class:

```python
from inequality.gini import Gini_Spatial
```

First, since the spatial Gini requires binary spatial weights, we will ensure this is so before proceeding:

```python
wq.transform = 'B'
```

Then, the spatial Gini can be computed from an income vector and the spatial weights describing adjacency among the observations. 

```python
gs69 = Gini_Spatial(pci_df['1969'], wq)
```

The aspatial Gini is stored in the `g` attribute, just like for the aspatial class:

```python
gs69.g
```

The share of the overall gini coefficient that is due to the "far" differences is stored in the `wcg` share:

```python
gs69.wcg_share
```

The $p$-value for this tests whether the component measuring inequality among neighbors is larger (or smaller) than that would have occurred if incomes were shuffled randomly around the map:

```python
gs69.p_sim
```

The value is statistically significant for 1969, indicating that inequality between neighboring pairs of counties is different from the inequality between county paris that are not geographically proximate.

We can apply the same statistic over each year in the sample using the same approach as before:

```python
def gini_spatial(incomes, weights=wq):
    gs = Gini_Spatial(incomes, weights)
    denom = 2*incomes.mean()*weights.n**2
    return gs.g, gs.wg/denom, gs.wcg/denom, gs.p_sim
```

Inference on this estimator is computationally demanding, since the pairwise differences have to be re-computed every permutation. 

```python
spatial_gini_results = pci_df[years].apply(gini_spatial, axis=0).T
spatial_gini_results.columns = ['gini', 'near_diffs', 'far_diffs', 'p_sim']
```

```python
spatial_gini_results.head()
```

The $p$-values are always small, suggesting that the contribution of the local ties is always smaller than that that would be expected if incomes were distributed randomly in the map.^[While it is possible that the "near differences" component could be *larger* than expected, that would imply negative spatial dependence, which is generally rare in empirical work.] We can compute the percent of times the $p$-value is smaller than a threshold using the mean:

```python
(spatial_gini_results.p_sim < 0.05).mean()
```

While it may appear that the component due to "near differences" is quite small, this has two reasons. First, the number of "nearby" pairs are less than 1% of all pairs of observations:

```python
wq.pct_nonzero
```

Second, when spatial dependence is high, nearby observations will be similar. So, each "near difference" will also be small. Adding together a small number of small observations will generally be small, relative to the large differences between distant observations. Thus, small values of the "near" distances are indicative of spatial dependence.  

Indeed, you can see that as the spatial dependence weakens, the `near_diffs` get larger:

```python
inequalities['near_diffs'] = spatial_gini_results.near_diffs
```

```python
inequalities[['near_diffs', 'moran']].plot.line(subplots=True, figsize=(15,6))
```

# Conclusion


Inequality is an important social phenomenon, and its geography is a serious, important concern for social scientists. This chapter discusses methods to assess inequality, as well as examine its spatial and regional structure. Through the Gini coefficient and Theil index, you can summarize the overall levels of inequality, as well as divide the components of inequality to those due to geographical region or proximate pairs of observations. Together, this gives us a good sense of how inequality manifests geographically, and how it is (possibly) distinct from other kinds of spatial measures, such as the measures of autocorrelation discussed in Chapter 7. 


# Questions

```python

```
