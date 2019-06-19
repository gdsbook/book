---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# Choropleth mapping



```

```python
%matplotlib inline

import seaborn
import pandas
import geopandas
import pysal
import numpy
from booktools import choropleth
from pysal.viz.mapclassify import classifiers as mapclassify
import matplotlib.pyplot as plt
```

<!-- #region -->
## Principles


Choropleth maps play a prominent role in geographic data science as they allow us to display non-geographic attributes or variables on a geographic map. The word
choropleth stems from the root "choro", meaning "region". As such choropleth maps
represent data at the region level, and
are appropriate for areal unit data where each observation combines a value of
an attribute and a geometric figure, usually a polygon. Choropleth maps derive from an earlier era where
cartographers faced technological constraints that precluded the use of
unclassed maps where each unique attribute value could be represented by a
distinct symbol or color. Instead, attribute values were grouped into a smaller number of
classes, usually not more than 12. Each class was associated with a unique symbol that was in turn
applied to all observations with attribute values falling in the class.

Although today these technological constraints are no longer binding, and
unclassed mapping is feasible, there are still good reasons for adopting a
classed approach. Chief among these is to reduce the cognitive load involved in
parsing the complexity of an unclassed map. A choropleth map reduces this
complexity by drawing upon statistical and visualization theory to provide an
effective representation of the spatial distribution of the attribute values
across the areal units. 

The effectiveness of a choropleth map will be a
function of the choice of classification scheme together with the color or
symbolization strategy adopted. In broad terms, the classification scheme
defines the number of classes as well as the rules for assignment, while the
symbolization should convey information about the value differentiation across
the classes.

In this chapter we first discuss the approaches used to classify
attribute values. This is followed by an overview of color theory and the
implications of different color schemes for effective map design. We  combine
theory and practice by exploring how these concepts are implemented in different Python packages, including `geopandas`, and `PySAL`.



<!-- #endregion -->

## Quantitative data classification 

Data classification considers the problem of 
partitioning the attribute values into mutually exclusive and exhaustive
groups. The precise manner in which this is done will be a function of the
measurement scale of the attribute in question. For quantitative attributes
(ordinal, interval, ratio scales) the classes will have an explicit ordering.
More formally, the classification problem is to define class boundaries such
that
$$
c_j < y_i \le  c_{j+1} \ \forall y_i \in C_{j+1}
$$
where $y_i$ is the
value of the attribute for spatial location $i$, $j$ is a class index, and $c_j$
represents the lower bound of interval $j$.

Different classification schemes obtain from their definition of the class
boundaries. The choice of the classification scheme should take into
consideration the statistical distribution of the attribute values.

To illustrate these considerations, we will examine regional income data for the
32 Mexican states. The variable we focus on is per capita gross domestic product
for 1940 (PCGDP1940):

```python
mx = geopandas.read_file("../data/mexicojoin.shp")
mx[['NAME', 'PCGDP1940']].head()
```

Which displays the following statistical distribution:

```python
h = seaborn.distplot(mx['PCGDP1940'], bins=5, rug=True);
```

As we can see, the distribution is positively skewed as in common in regional income studies. In other words,
the mean exceeds the median (`50%`, in the table below), leading the to fat right tail in the figure. As
we shall see, this skewness will have implications for the choice of choropleth
classification scheme.

```python
mx['PCGDP1940'].describe()
```

### Classification schemes

For quantitative attributes we first sort the data by their value,
such that $x_0 \le x_2 \ldots \le x_{n-1}$. For a prespecified number of classes
$k$, the classification problem boils down to selection of $k-1$ break points
along the sorted values that separate the values into mutually exclusive and
exhaustive groups.

In fact, the determination of the histogram above can
be viewed as one approach to this selection.
The method `seaborn.distplot` uses the matplotlib `hist`
function under the hood to determine the class boundaries and the counts of
observations in each class. In the figure, we have five classes which can be
extracted with an explicit call to the `hist` function:

```python
counts, bins, patches = h.hist(mx['PCGDP1940'], bins=5)
```

The `counts` object captures how many observations each category in the classification has:

```python
counts
```

The `bin` object stores these break points we are interested in when considering classification schemes (the `patches` object can be ignored in this context, as it stores the geometries of the histogram plot):

```python
bins
```

This yields 5 bins, with the first having a lower bound of 1892 and an upper
bound of 5985.8 which contains 17 observations. 
The determination of the
interval width ($w$) and the number of bins in `seaborn` is based on the Freedman-Diaconis rule:

$$w = 2 * IQR * n^{-1/3}$$

where $IQR$ is the inter quartile
range of the attribute values. Given $w$ the number of bins ($k$) is:

$$k=(max-
min)/w.$$

Below we present several approaches to create these break points that follow criteria that can be of interest in different contexts, as they focus on different priorities.
 
#### Equal Intervals

The Freedman-Diaconis approach provides a rule to determine
the width and, in turn, the number of bins for the classification. This is a
special case of a more general classifier known as "equal intervals", where each
of the bins has the same width in the value space. 
For a given value of $k$, equal intervals
classification splits the range of the attribute space into $k$ equal length
intervals, with each interval having a width
$w = \frac{x_0 - x_{n-1}}{k}$.
Thus the maximum class is $(x_{n-1}-w, x_{n-1}]$ and the first class is
$(-\infty, x_{n-1} - (k-1)w]$.

Equal intervals have the dual advantages of
simplicity and ease of interpretation. However, this rule only considers the extreme
values of the distribution and, in some cases, this can result in one or more
classes being sparse. This is clearly the case in our income dataset, as the majority of
the values are placed into the first two classes leaving the last three classes
rather sparse:

```python
ei5 = mapclassify.Equal_Interval(mx['PCGDP1940'], k=5)
ei5
```

  Note that each of the intervals, however, has equal width of
$w=4093.8$. This value of $k=5$ also coincides with the default classification
in the Seaborn histogram in Figure 1.


#### Quantiles
To avoid the potential problem of sparse classes, the quantiles of
the distribution can be used to identify the class boundaries. Indeed, each
class will have approximately $\mid\frac{n}{k}\mid$ observations using the quantile
classifier. If $k=5$ the sample quintiles are used to define the upper limits of
each class resulting in the following classification:

```python
q5 = mapclassify.Quantiles(mx.PCGDP1940, k=5)
q5
```

Note that while the numbers of values in each class are roughly equal, the
widths of the first four intervals are rather different:

```python
q5.bins[1:]-q5.bins[:-1]
```

While quantiles does avoid the pitfall of sparse classes, this classification is
not problem free. The varying widths of the intervals can be markedly different
which can lead to problems of interpretation. A second challenge facing quantiles
arises when there are a large number of duplicate values in the distribution
such that the limits for one or more classes become ambiguous. For example, if one had a variable with $n=20$ but 10 of the observations took on the same value which was the minimum observed, then for values of $k>2$, the class boundaries become ill-defined since a simple rule of splitting at the $n/k$ ranked observed value would depend upon how ties are tried when ranking.

#### Mean-standard deviation

Our third classifer uses the sample mean $\bar{x} =
\frac{1}{n} \sum_{i=1}^n x_i$ and sample standard deviation $s = \sqrt{
\frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})  }$ to define class boundaries as
some distance from the sample mean, with the distance being a multiple of the
standard deviation. For example, a common definition for $k=5$ is to set the
upper limit of the first class to two standard deviations ($c_{0}^u = \bar{x} - 2 s$), and the intermediate
classes to have upper limits within one standard deviation ($c_{1}^u = \bar{x}-s,\ c_{2}^u = \bar{x}+s, \ c_{3}^u
= \bar{x}+2s$). Any values greater (smaller) than two standard deviations above (below) the mean
are placed into the top (bottom) class.

```python
msd = mapclassify.Std_Mean(mx['PCGDP1940'])
msd
```

This classifier is best used when data is normally distributed or, at least, when the sample mean is a meaningful measure to anchor the classification around. Clearly this is
not the case for our income data as the positive skew results in a loss of
information when we use the standard deviation. The lack of symmetry leads to
an inadmissible upper bound for the first  class as well as a concentration of the
vast majority of values in the middle class.

#### Maximum Breaks

The maximum breaks classifier decides where to set the break points between
classes by considering the difference between sorted values. That is, rather
than considering a value of the dataset in itself, it looks at how appart each
value is from the next one in the sorted sequence. The classifier then places
the the $k-1$ break points in between the pairs of values most stretched apart from
each other in the entire sequence, proceeding in descending order relative to
the size of the breaks:

```python
mb5 = mapclassify.Maximum_Breaks(mx['PCGDP1940'], k=5)
mb5
```

Maximum breaks is an appropriate approach when we are interested in making sure
observations in each class are separated from those in neighboring classes. As
such, it works well in cases where the distribution of values is not unimodal.
In addition, the algorithm is relatively fast to compute. However, its
simplicitly can sometimes cause unexpected results. To the extent in only
considers the top $k-1$ differences between consecutive values, other more nuanced
within-group differences and dissimilarities can be ignored.

#### Box-Plot

The box-plot classification is a blend of the quantile and
standard deviation classifiers. Here $k$ is predefined to six, with the upper limit of class 0 set
to $q_{0.25}-h \, IQR$. $IQR = q_{0.75}-q_{0.25}$ is the
inter-quartile range; $h$ corresponds to the hinge, or the multiplier of the $IQR$ to obtain the bounds of the whiskers. The lower limit of the sixth class is set to $q_{0.75}+h \,
IQR$. Intermediate classes have their upper limits set to the 0.25, 0.50 and
0.75 percentiles of the attribute values.

```python
bp = mapclassify.Box_Plot(mx['PCGDP1940'])
bp
```

Any values falling into either of the extreme classes are defined as outlers.
Note that because the income values are non-negative by definition, the lower
outlier class has an inadmissible upper bound meaning that lower outliers would
not be possible for this sample.

The default value for the hinge is $h=1.5$ in
PySAL. However, this can be specified by the user for an alternative classification:

```python
bp1 = mapclassify.Box_Plot(mx['PCGDP1940'], hinge=1)
bp1
```

Doing so will affect the definition of the outlier classes, as well as the
neighboring internal classes.

#### Head Tail Breaks

The head tail algorithm, introduced by Jiang (2013), is based on a recursive partioning of the data using splits around
iterative means. The splitting process continues until the distributions within each of
the classes no longer display a heavy-tailed distribution in the sense that
there is a balance between the number of smaller and larger values assigned to
each class.

```python
ht = mapclassify.HeadTail_Breaks(mx['PCGDP1940'])
ht
```

For data with a heavy-tailed distribution, such as power law and log normal
distributions, the head tail breaks classifier (Jiang 2015) can be particularly
effective.

#### Jenks Caspall

This approach, as well as the following two, tackles the calssification challenge from a heuristic perspective, rather than from deterministic one. Originally proposed by Jenks & Caspall (1971), this algorithm aims to minimize the sum of absolute deviations around
class means. The approach begins with a prespecified number of classes and an
arbitrary initial set of class breaks - for example using quintiles. The
algorithm attempts to improve the objective function by considering the movement
of observations between adjacent classes. For example, the largest value in the
lowest quintile would be considered for movement into the second quintile, while
the lowest value in the second quintile would be considered for a possible move
into the first quintile. The candidate move resulting in the largest reduction
in the objective function would be made, and the process continues until no
other improving moves are possible.

```python
numpy.random.seed(12345)
jc5 = mapclassify.Jenks_Caspall(mx['PCGDP1940'], k=5)
jc5
```

#### Fisher Jenks

The second optimal algorithm adopts a dynamic programming approach to minimize
the sum of the absolute deviations around class medians. In contrast to the
Jenks-Caspall approach, Fisher-Jenks is guaranteed to produce an optimal
classification for a prespecified number of classes:

```python
numpy.random.seed(12345)
fj5 = mapclassify.Fisher_Jenks(mx['PCGDP1940'], k=5)
fj5
```

#### Max-p

Finally, the max-p classifiers adopts the algorithm underlying the max-p region
building method (Duque, Anselin and Rey, 2012) to the case of map classification. It is similar in spirit to
Jenks Caspall in that it considers greedy swapping between adjacent classes to
improve the objective function. It is a heuristic, however, so unlike
Fisher-Jenks, there is no optimial solution guaranteed:

```python
mp5 = mapclassify.Max_P_Classifier(mx['PCGDP1940'], k=5)
mp5
```

### Comparing Classification schemes

As a special case of clustering, the definition of
the number of classes and the class boundaries pose a problem to the map
designer. Recall that the Freedman-Diaconis rule was said to be optimal,
however, the optimality necessitates the specification of an objective function.
In the case of Freedman-Diaconis, the objective function is to minimize the
difference between the area under estimated kernel density based on the sample
and the area under the theoretical population distribution that generated the
sample.

This notion of statistical fit is an important one. However, it is not the
only consideration when evaluating classifiers for the purpose of choropleth
mapping. Also relevant is the spatial distribution of the attribute values and
the ability of the classifier to convey a sense of that spatial distribution. As
we shall see, this is not necessarily directly related to the statistical
distribution of the attribute values. We will return to a joint consideration of both
the statistical and spatial distribution of the attribute values in comparison
of classifiers below.

For map classification, one optimiality criterion that
can be used is a measure of fit. In PySAL the "absolute deviation around class
medians" (ADCM) is calculated and provides a measure of fit that allows for
comparison of alternative classifiers for the same value of $k$.

To see this, we can compare different classifiers for $k=5$ on the Mexico data:

```python
class5 = q5, ei5, ht, mb5, msd, fj5, jc5
fits = numpy.array([ c.adcm for c in class5])
data = pandas.DataFrame(fits)
data['classifier'] = [c.name for c in class5]
data.columns = ['ADCM', 'Classifier']
ax = seaborn.barplot(y='Classifier', x='ADCM', data=data)
```

As is to be expected, the Fisher-Jenks classifier dominates all other k=5
classifiers with an ACDM of 23,729. Interestingly, the equal interval classifier
performs well despite the problems associated with being sensitive to the
extreme values in the distribution. The mean-standard deviation classifier has a
very poor fit due to the skewed nature of the data and the concentrated
assignment of the majority of the observations to the central class.

The ADCM provides a global measure of fit which can be used to compare the
alternative classifiers. As a complement to this global perspective, it can be
revealing to consider how each of the spatial observations was classified across
the alternative approaches. To do this we can add the class bin attribute (`yb`)
generated by the PySAL classifiers as additional columns in the data frame and
present these jointly in a table:

```python
mx['q540'] = q5.yb
mx['ei540'] = ei5.yb
mx['ht40'] = ht.yb
mx['mb540'] = mb5.yb
mx['msd40'] = msd.yb
mx['fj540'] = fj5.yb
mx['jc540'] = jc5.yb
```

```python
mxs = mx.sort_values('PCGDP1940')
```

```python
def highlight_values(val):
    if val==0:
        return 'background-color: %s' % '#ffffff'
    elif val==1:
        return 'background-color: %s' % '#e0ffff'
    elif val==2:
        return 'background-color: %s' % '#b3ffff'
    elif val==3:
        return 'background-color: %s' % '#87ffff'
    elif val==4:
        return 'background-color: %s' % '#62e4ff'
    else:
        return ''
```

```python
t = mxs[['NAME', 'PCGDP1940', 'q540', 'ei540', 'ht40', 'mb540', 'msd40', 'fj540', 'jc540']]
t.style.applymap(highlight_values)
```

Inspection of this table reveals a number of interesting results. First, the
only Mexican state that is treated consistantly across the k=5 classifiers is
Baja California Norte which is placed in the highest class by all classifiers.
Second, the mean-standard deviation classifier has an empty first class due to
the inadmissible upper bound and the overconcentration of values in the central
class (2).

Finally, we can consider a meso-level view of the clasification
results by comparing the number of values assigned to each class across the
different classifiers:

```python
pandas.DataFrame({c.name: c.counts for c in class5},
                 index=['Class-{}'.format(i) for i in range(5)])
```

Doing so highlights the similarities between Fisher Jenks and equal intervals as
the distribution counts are very similar as the two approaches agree on all 17
states assigned to the first class. Indeed, the only observation that
distinguishes the two classifiers is the treatment of Baja California Sur which
is kept in class 1 in equal intervals, but assigned to class 2 by Fisher Jenks.

### Color

Having considered the evaluation of the statisitcal distribution of
the attribute values and the alternative classification approaches, the next
step is to select the symbolization and color scheme. Together with the choice of classifier, these will determine the overall
effectiveness of the choropleth map in representing the spatial
distribution of the attribute values.

Let us start by refreshing the `mx` object and exploring the base polygons for the Mexican states:

```python
mx = geopandas.read_file("../data/mexicojoin.shp")
mx.plot(color='blue', edgecolor='y');
```

Prior to examining the attribute values it is important to note that the
spatial units for these states are far from homogenous in their shapes and
sizes. This can have major impacts on our brain's pattern recognition capabilities
as we tend to be drawn to the larger polygons. Yet, when we considered the
statistical distribution above, each observation was given equal weight. Thus
the spatial distribution becomes more complicated to evaluate from a visual and
statistical perspective.

With this qualification in mind, we will explore the construction of choropleth
maps using PySAL and the helper method `choropleth`:

```python
choropleth(mx, 'PCGDP1940', cmap='BuGn')
```

The default uses a quantile classification with k=5, together with a green color
pallete where darker hues indicate higher class assignment for the attribute
values associated with each polygon.

These defaults can be overriden in a
number of ways, for example changing the colormap and number of quantiles:

```python
choropleth(mx, 'PCGDP1940', cmap='Blues', k=4)
```

We will continue to use the `Blues`
colormap in what follows in order to examine the spatial distribution revealed by
each of the k=5 classifiers:

- Equal Interval
- Quantiles
- Mean-Standard deviation
- Maximum breaks
- Box-Plot
- Head Tail
- Jenks Caspall
- Fisher Jenks

**DA-B NOTE**: it would *really* nice to have a `mapclassify.plot` method to go from created classifications to choropleths in this context.

```python
choropleth(mx, 'PCGDP1940', cmap='Blues',
           scheme='equal_interval', k=5)
```

```python
#choropleth(mx, 'PCGDP1940', cmap='Blues',
#           scheme='Std_Mean', k=5)
```

```python
choropleth(mx, 'PCGDP1940', cmap='Blues',
           scheme='maximum_breaks', k=5)
```

```python
#choropleth(mx, 'PCGDP1940', cmap='Blues',
#           scheme='Box_Plot')
```

```python
#choropleth(mx, 'PCGDP1940', cmap='Blues', 
#           scheme='headtail_breaks')
```

```python
numpy.random.seed(12345)
#choropleth(mx, 'PCGDP1940', cmap='Blues',
#           scheme='jenks_caspall', k=5)
```

```python
numpy.random.seed(12345)
choropleth(mx, 'PCGDP1940', cmap='Blues', 
           scheme='fisher_jenks', k=5)
```

We also note that geopandas can be used to do the plotting here, and we add some customization to the legend to report the upper bounds of each class:

```python
### Fill in with example once `geopandas` is back in sync with PySAL 2.0
```

### Diverging Attributes

A slightly different type of attribute is the so-called "diverging" values attribute. This is
useful when one wishes to place equal emphasis on mid-range critical values as
well as extremes at both ends of the distribution. Light colors are used to
emphasize the mid-range class while dark colors with contrasting hues are used
to distinguish the low and high extremes.

To illustrate this for the Mexican
income data we can derive a new variable which measures the change in a state's
rank in the income distribution between 1940 to 2000:

```python
rnk = mx.rank(ascending=False) # ascending ranks 1=high, n=lowest
rnk['NAME']=mx['NAME']
delta_rnk = rnk.PCGDP1940 - rnk.PCGDP2000
delta_rnk
cls = numpy.digitize(delta_rnk, [-5, 0, 5, 20])
cls
```

Here we have created four classes for the rank changes: [-inf, -5), [-5, 0), [0,
5), [5, 20]. Note that these are descending ranks, so the wealthiest state in
any period has a rank of 1 and therefore when considering the change in ranks, a
negative change reflects moving down the income distribution.

```python
choropleth(mx.assign(rkd=cls), 'rkd', cmap='RdYlBu',
           scheme='equal_interval', k=4)
```

Here the red (blue) hues are states that have moved downwards (upwards) in the
income distribution, with the darker hue representing a larger movement.


## Qualitative Attributes

The Mexico data set also has several variables that
are on a nominal measurement scale. One of these is a region definition variable
that groups individual states in contiguous clusters of similar characteristics:

```python
mx['HANSON98'].head()
```

This regionalization scheme partions Mexico into 5 regions. A naive (and
incorrect) way to display this would be to treat the region variable as
sequential and use equal_inteval $k=5$ to display the regions:

```python
choropleth(mx, 'HANSON98', cmap='Blues', scheme='equal_interval')
```

This is not correct because the region variable is not on an interval scale, so
the differences between the values have no quantitative significance but rather
the values simply indicate region membership. However, the choropleth above gives
a clear visual cue that regions in the south are more (of whichever is being plot)
than those in the north, as the color map implies an intensity gradient.

A more appropriate visualization
is to use a "qualitative" color palette:

```python
choropleth(mx, 'HANSON98', cmap='Pastel1',
           scheme='equal_interval')
```

## Conclusion

In this chapter we have considered the construction of choropleth maps for
spatial data visualization. The key issues of the choice of classification
scheme, variable measurement scale, spatial configuration and color palettes
were illustrated using PySAL's map classification module together with other
related packages in the PyData stack.

Choropleth maps are a central tool in the geographic data science arsenal as
they provide powerful visualizations of the spatial distribution of attribute
values. We have only touched on the basic concepts in this chapter, as there is
much more that can be said about cartographic theory and the design of effective
choropleth maps. Readers interested in pursuing this literature are encouraged
to see the references cited.

At the same time, given the philosophy underlying PySAL the methods we cover
here are sufficient for exploratory data analysis where the rapid and flexible
generation of views is critical to the work flow. Once the analysis is complete,
and the final presentation quality maps are to be generated, there are excellent
packages in the data stack that the user can turn to.

## Questions

1. A variable such as population density measured for census tracts in a metropolitan area can display a high degree of skewness. What is an appropriate choice for a choropleth classification for such a variable?
2. Provide two solutions to the problem of ties when applying quantile classification to the following series: $y=[2,2,2,2,2,2,4,7,8,9,20,21]$ and $k=4$. Discuss the merits of each approach.
3. Which classifiers are appropriate for data that displays a high degree of multi-modality in its statistical distribution?
4. Contrast and compare classed choropleth maps with class-less choropleth maps? What are the strengths and limitations of each type of visualization for spatial data?
5. In what ways do choropleth classifiers treat intra-class and inter-class heterogeneity differently? What are the implications of these choices?
6. To what extent do most commonly employed choropleth classification methods take the geographical distribution of the variable into consideration? Can you think of ways to incorporate the spatial features of a variable into a classification?
7. Discuss the similarities between  the choice of the number of classes in choropleth mapping, on the one hand, and the determination of the number of clusters in a data set on the other. What aspects of choropleth mapping differentiate the former from the latter?
8. The Fisher-Jenks classifier will always dominate other k-classifiers for a given data set, with respect to statistical fit. Given this, why might one decide on choosing a different k-classifier for a particular data set?


## References

Duque, J.C., L. Anselin, and S.J. Rey. (2012) "The max-p regions problem." *Journal of Regional Science*, 52:397-419.

Jenks, G. F., & Caspall, F. C. (1971). Error on choroplethic maps: definition, measurement, reduction. Annals of the Association of American Geographers, 61(2), 217-244.

Jian, B. (2013) "Head/Tail Breaks: A New Classification Scheme for Data with a Heavy-Tailed Distribution." *The Professional Geographer*, 65(3): 482-494.

Jiang, Bin. (2015) "Head/tail breaks for visualization of city
structure and dynamics." *Cities*, 43: 69-77.

Rey, S.J. and M.L. Guitierez. (2010)
"Interregional inequality dynamics in Mexico." *Spatial Economic Analysis*, 5:
277-298.
