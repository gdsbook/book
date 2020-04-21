# US County Income 1969-2017

This documents the creation of the data set: `uscountypcincome.gpkg` that is used in the following chapter(s):

- 09_inequality


## Data Sources

The original files, and the final geopackage,  have been archived at [gdsbook/data\_archive/us\_county_income](https://github.com/gdsbook/data_archive)

- `CANINC1.zip` has attribute data for per capita income for US counties 1969-2017
- `tl_2019_us_county.zip` has the geometries for the counties

The data archive has urls back to the original sources for the downloads.

## Notebooks

To produce the dataset, run these notebooks in sequence:

- `010_csv.ipynb`: Reads `CAINC1.zip` and creates csv files for each state
- `015_csv_state_shp.ipynb`: Creates state specific shapefiles
- `020_virginia.ipynb`: Handles independent cities and counties in Virginia
- `030_wisconsin.ipynb`: Handles county merges  in Wisconsin
- `040_other.ipynb`: Handles county histories in CO, NM, AZ
- `050_shrink.ipynb`: Creates final geopackage with pcincome by county

## Notes

- The geopackage can either be created locally by running the notebooks in sequence, or by downloading from the gdsbook data archive listed above.
- Population and Total County income are not in the final geopackage but are available in the file `uscountyincome.csv`
