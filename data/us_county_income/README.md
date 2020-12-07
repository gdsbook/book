# US County Income 1969-2017

This documents the creation of the data set: `uscountypcincome.gpkg` that is used in the following chapter(s):

- 09_inequality

**Note that because of the large size of the file, the consuming notebook reads a remote version of the file. We do not store the file in this repository. The file can be reconstructed locally using the notebooks below.**

## Data Sources

The original files, and the final geopackage,  have been archived at [gdsbook/data\_archive/us\_county_income](https://github.com/gdsbook/data_archive)

- `CANINC1.zip` has attribute data for per capita income for US counties 1969-2017
- `tl_2019_us_county.zip` has the geometries for the counties

The data archive has urls back to the original sources for the downloads.


## File Generated

- `uscountypcincome.gpk` Final geopackage with US county per capita incomes 1969-2017

## Notebooks

To produce the dataset, run the following notebook.

- `us_county_income_clean.ipynb`

## Notes

- The geopackage can either be created locally by running the notebooks in sequence, or by downloading from the gdsbook data archive listed above.
- Population and Total County income are not in the final geopackage but are available in the file `uscountyincome.csv`
