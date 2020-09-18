# San Diego Dataset



## Airbnb Listings

- points
- 255 observations

Source: [Inside Airbnb](http://data.insideairbnb.com/united-states/ca/san-diego/2016-07-07/data/listings.csv.gz)

Downloaded: 2017-03-01, Complied 2016-07-02


## ACS Data


- polygons (tracts
- 628 observations
- 2011-2015 (Centered on 2013)

Source: [Cenpy notebook](https://github.com/sjsrey/gds/blob/sandiego/chapters/data/sandiego/san_diego_acs.ipynb)

Compiled: 2017-03-09


## Elevation Data

- DEM from USGS  https://earthexplorer.usgs.gov/
- Spatial join for point elevation [Soure notebook](https://github.com/sjsrey/gds/blob/sandiego/chapters/data/sandiego/sampling_elevation.ipynb)


Compiled: 2017-03-09

## Coastline

- Shapefile for US Coastline from https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2015&layergroup=Coastline
- In QGIS san diego coastline extracted via a intersection with buffer for dissolved sd county census tracts
-

Downloaded: 2017-03-15


