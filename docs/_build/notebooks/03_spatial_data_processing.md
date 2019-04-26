---
redirect_from:
  - "/notebooks/03-spatial-data-processing"
interact_link: content/notebooks/03_spatial_data_processing.ipynb
kernel_name: python3
has_widgets: false
title: 'Spatial Data Processing'
prev_page:
  url: /notebooks/02_spatial_data
  title: 'Spatial Data'
next_page:
  url: /notebooks/04_spatial_weights
  title: 'Spatial Weights'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---

# Spatial Data Processing

Intro paragraph
* deterministic spatial analysis (SG)

* Explain what we mean by dsa
* outline what we will cover below


 airports.csv

## Vignette: Airports

- Querying based on attributes (volume, lon/lat, etc.)

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
import pysal as ps
import pandas as pd
df = pd.read_csv("data/airports/world-airports.csv")
```
</div>

</div>

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
df.head()
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ident</th>
      <th>type</th>
      <th>name</th>
      <th>latitude_deg</th>
      <th>longitude_deg</th>
      <th>elevation_ft</th>
      <th>continent</th>
      <th>iso_country</th>
      <th>iso_region</th>
      <th>municipality</th>
      <th>scheduled_service</th>
      <th>gps_code</th>
      <th>iata_code</th>
      <th>local_code</th>
      <th>home_link</th>
      <th>wikipedia_link</th>
      <th>keywords</th>
      <th>score</th>
      <th>last_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2434</td>
      <td>EGLL</td>
      <td>large_airport</td>
      <td>London Heathrow Airport</td>
      <td>51.470600</td>
      <td>-0.461941</td>
      <td>83.0</td>
      <td>EU</td>
      <td>GB</td>
      <td>GB-ENG</td>
      <td>London</td>
      <td>1</td>
      <td>EGLL</td>
      <td>LHR</td>
      <td>NaN</td>
      <td>http://www.heathrowairport.com/</td>
      <td>http://en.wikipedia.org/wiki/London_Heathrow_A...</td>
      <td>LON, Londres</td>
      <td>1251675</td>
      <td>2016-02-09T10:54:31+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3632</td>
      <td>KLAX</td>
      <td>large_airport</td>
      <td>Los Angeles International Airport</td>
      <td>33.942501</td>
      <td>-118.407997</td>
      <td>125.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CA</td>
      <td>Los Angeles</td>
      <td>1</td>
      <td>KLAX</td>
      <td>LAX</td>
      <td>LAX</td>
      <td>http://www.iflylax.com/</td>
      <td>http://en.wikipedia.org/wiki/Los_Angeles_Inter...</td>
      <td>NaN</td>
      <td>1335475</td>
      <td>2010-01-23T11:27:55+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3754</td>
      <td>KORD</td>
      <td>large_airport</td>
      <td>Chicago O'Hare International Airport</td>
      <td>41.978600</td>
      <td>-87.904800</td>
      <td>672.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-IL</td>
      <td>Chicago</td>
      <td>1</td>
      <td>KORD</td>
      <td>ORD</td>
      <td>ORD</td>
      <td>http://www.flychicago.com/ohare/en/home/Pages/...</td>
      <td>http://en.wikipedia.org/wiki/O'Hare_Internatio...</td>
      <td>CHI, Orchard Place</td>
      <td>1503175</td>
      <td>2014-08-19T18:55:27+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3622</td>
      <td>KJFK</td>
      <td>large_airport</td>
      <td>John F Kennedy International Airport</td>
      <td>40.639801</td>
      <td>-73.778900</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NY</td>
      <td>New York</td>
      <td>1</td>
      <td>KJFK</td>
      <td>JFK</td>
      <td>JFK</td>
      <td>http://www.panynj.gov/CommutingTravel/airports...</td>
      <td>http://en.wikipedia.org/wiki/John_F._Kennedy_I...</td>
      <td>Manhattan, New York City, NYC, Idlewild</td>
      <td>1052075</td>
      <td>2010-01-23T11:27:54+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3384</td>
      <td>KATL</td>
      <td>large_airport</td>
      <td>Hartsfield Jackson Atlanta International Airport</td>
      <td>33.636700</td>
      <td>-84.428101</td>
      <td>1026.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-GA</td>
      <td>Atlanta</td>
      <td>1</td>
      <td>KATL</td>
      <td>ATL</td>
      <td>ATL</td>
      <td>http://www.atlanta-airport.com/</td>
      <td>http://en.wikipedia.org/wiki/Hartsfield-Jackso...</td>
      <td>NaN</td>
      <td>2002475</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
  </tbody>
</table>
</div>
</div>


</div>
</div>
</div>

Let's use pandas to query for the airports within the `large_airport` class:

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
df[df.type == 'large_airport']
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ident</th>
      <th>type</th>
      <th>name</th>
      <th>latitude_deg</th>
      <th>longitude_deg</th>
      <th>elevation_ft</th>
      <th>continent</th>
      <th>iso_country</th>
      <th>iso_region</th>
      <th>municipality</th>
      <th>scheduled_service</th>
      <th>gps_code</th>
      <th>iata_code</th>
      <th>local_code</th>
      <th>home_link</th>
      <th>wikipedia_link</th>
      <th>keywords</th>
      <th>score</th>
      <th>last_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2434</td>
      <td>EGLL</td>
      <td>large_airport</td>
      <td>London Heathrow Airport</td>
      <td>51.470600</td>
      <td>-0.461941</td>
      <td>83.0</td>
      <td>EU</td>
      <td>GB</td>
      <td>GB-ENG</td>
      <td>London</td>
      <td>1</td>
      <td>EGLL</td>
      <td>LHR</td>
      <td>NaN</td>
      <td>http://www.heathrowairport.com/</td>
      <td>http://en.wikipedia.org/wiki/London_Heathrow_A...</td>
      <td>LON, Londres</td>
      <td>1251675</td>
      <td>2016-02-09T10:54:31+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3632</td>
      <td>KLAX</td>
      <td>large_airport</td>
      <td>Los Angeles International Airport</td>
      <td>33.942501</td>
      <td>-118.407997</td>
      <td>125.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CA</td>
      <td>Los Angeles</td>
      <td>1</td>
      <td>KLAX</td>
      <td>LAX</td>
      <td>LAX</td>
      <td>http://www.iflylax.com/</td>
      <td>http://en.wikipedia.org/wiki/Los_Angeles_Inter...</td>
      <td>NaN</td>
      <td>1335475</td>
      <td>2010-01-23T11:27:55+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3754</td>
      <td>KORD</td>
      <td>large_airport</td>
      <td>Chicago O'Hare International Airport</td>
      <td>41.978600</td>
      <td>-87.904800</td>
      <td>672.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-IL</td>
      <td>Chicago</td>
      <td>1</td>
      <td>KORD</td>
      <td>ORD</td>
      <td>ORD</td>
      <td>http://www.flychicago.com/ohare/en/home/Pages/...</td>
      <td>http://en.wikipedia.org/wiki/O'Hare_Internatio...</td>
      <td>CHI, Orchard Place</td>
      <td>1503175</td>
      <td>2014-08-19T18:55:27+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3622</td>
      <td>KJFK</td>
      <td>large_airport</td>
      <td>John F Kennedy International Airport</td>
      <td>40.639801</td>
      <td>-73.778900</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NY</td>
      <td>New York</td>
      <td>1</td>
      <td>KJFK</td>
      <td>JFK</td>
      <td>JFK</td>
      <td>http://www.panynj.gov/CommutingTravel/airports...</td>
      <td>http://en.wikipedia.org/wiki/John_F._Kennedy_I...</td>
      <td>Manhattan, New York City, NYC, Idlewild</td>
      <td>1052075</td>
      <td>2010-01-23T11:27:54+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3384</td>
      <td>KATL</td>
      <td>large_airport</td>
      <td>Hartsfield Jackson Atlanta International Airport</td>
      <td>33.636700</td>
      <td>-84.428101</td>
      <td>1026.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-GA</td>
      <td>Atlanta</td>
      <td>1</td>
      <td>KATL</td>
      <td>ATL</td>
      <td>ATL</td>
      <td>http://www.atlanta-airport.com/</td>
      <td>http://en.wikipedia.org/wiki/Hartsfield-Jackso...</td>
      <td>NaN</td>
      <td>2002475</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4185</td>
      <td>LFPG</td>
      <td>large_airport</td>
      <td>Charles de Gaulle International Airport</td>
      <td>49.012798</td>
      <td>2.550000</td>
      <td>392.0</td>
      <td>EU</td>
      <td>FR</td>
      <td>FR-J</td>
      <td>Paris</td>
      <td>1</td>
      <td>LFPG</td>
      <td>CDG</td>
      <td>NaN</td>
      <td>http://www.aeroportsdeparis.fr/</td>
      <td>http://en.wikipedia.org/wiki/Charles_de_Gaulle...</td>
      <td>PAR, Aéroport Roissy-Charles de Gaulle, Roissy...</td>
      <td>1127475</td>
      <td>2010-01-05T18:17:43+00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3878</td>
      <td>KSFO</td>
      <td>large_airport</td>
      <td>San Francisco International Airport</td>
      <td>37.618999</td>
      <td>-122.375000</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CA</td>
      <td>San Francisco</td>
      <td>1</td>
      <td>KSFO</td>
      <td>SFO</td>
      <td>SFO</td>
      <td>http://www.flysfo.com/</td>
      <td>http://en.wikipedia.org/wiki/San_Francisco_Int...</td>
      <td>QSF, QBA</td>
      <td>1112475</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2513</td>
      <td>EHAM</td>
      <td>large_airport</td>
      <td>Amsterdam Airport Schiphol</td>
      <td>52.308601</td>
      <td>4.763890</td>
      <td>-11.0</td>
      <td>EU</td>
      <td>NL</td>
      <td>NL-NH</td>
      <td>Amsterdam</td>
      <td>1</td>
      <td>EHAM</td>
      <td>AMS</td>
      <td>NaN</td>
      <td>http://www.schiphol.nl/</td>
      <td>http://en.wikipedia.org/wiki/Amsterdam_Schipho...</td>
      <td>NaN</td>
      <td>1093275</td>
      <td>2011-10-05T23:26:15+00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2212</td>
      <td>EDDF</td>
      <td>large_airport</td>
      <td>Frankfurt am Main International Airport</td>
      <td>50.033333</td>
      <td>8.570556</td>
      <td>364.0</td>
      <td>EU</td>
      <td>DE</td>
      <td>DE-HE</td>
      <td>Frankfurt-am-Main</td>
      <td>1</td>
      <td>EDDF</td>
      <td>FRA</td>
      <td>NaN</td>
      <td>http://www.frankfurt-airport.de/</td>
      <td>http://en.wikipedia.org/wiki/Frankfurt_Airport</td>
      <td>EDAF, Rhein-Main Air Base</td>
      <td>1144675</td>
      <td>2016-09-15T03:34:21+00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3521</td>
      <td>KEWR</td>
      <td>large_airport</td>
      <td>Newark Liberty International Airport</td>
      <td>40.692501</td>
      <td>-74.168701</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NJ</td>
      <td>Newark</td>
      <td>1</td>
      <td>KEWR</td>
      <td>EWR</td>
      <td>EWR</td>
      <td>http://www.panynj.gov/CommutingTravel/airports...</td>
      <td>http://en.wikipedia.org/wiki/Newark_Liberty_In...</td>
      <td>Manhattan, New York City, NYC</td>
      <td>1064475</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3631</td>
      <td>KLAS</td>
      <td>large_airport</td>
      <td>McCarran International Airport</td>
      <td>36.080101</td>
      <td>-115.152000</td>
      <td>2181.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NV</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>KLAS</td>
      <td>LAS</td>
      <td>LAS</td>
      <td>http://www.mccarran.com/</td>
      <td>http://en.wikipedia.org/wiki/McCarran_Internat...</td>
      <td>NaN</td>
      <td>1068475</td>
      <td>2010-07-05T16:48:39+00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3488</td>
      <td>KDFW</td>
      <td>large_airport</td>
      <td>Dallas Fort Worth International Airport</td>
      <td>32.896801</td>
      <td>-97.038002</td>
      <td>607.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-TX</td>
      <td>Dallas-Fort Worth</td>
      <td>1</td>
      <td>KDFW</td>
      <td>DFW</td>
      <td>DFW</td>
      <td>http://www.dfwairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Dallas-Fort_Worth...</td>
      <td>QDF</td>
      <td>1203175</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3670</td>
      <td>KMCO</td>
      <td>large_airport</td>
      <td>Orlando International Airport</td>
      <td>28.429399</td>
      <td>-81.308998</td>
      <td>96.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-FL</td>
      <td>Orlando</td>
      <td>1</td>
      <td>KMCO</td>
      <td>MCO</td>
      <td>MCO</td>
      <td>http://www.orlandoairports.net/</td>
      <td>http://en.wikipedia.org/wiki/Orlando_Internati...</td>
      <td>Disney World,Epcot Center</td>
      <td>1044075</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3486</td>
      <td>KDEN</td>
      <td>large_airport</td>
      <td>Denver International Airport</td>
      <td>39.861698</td>
      <td>-104.672997</td>
      <td>5431.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CO</td>
      <td>Denver</td>
      <td>1</td>
      <td>KDEN</td>
      <td>DEN</td>
      <td>DEN</td>
      <td>http://www.flydenver.com/</td>
      <td>http://en.wikipedia.org/wiki/Denver_Internatio...</td>
      <td>DVX, KVDX</td>
      <td>1103275</td>
      <td>2015-11-13T09:28:42+00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3685</td>
      <td>KMIA</td>
      <td>large_airport</td>
      <td>Miami International Airport</td>
      <td>25.793200</td>
      <td>-80.290604</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-FL</td>
      <td>Miami</td>
      <td>1</td>
      <td>KMIA</td>
      <td>MIA</td>
      <td>MIA</td>
      <td>http://www.miami-airport.com/</td>
      <td>http://en.wikipedia.org/wiki/Miami_Internation...</td>
      <td>MFW, South Florida</td>
      <td>1073075</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3602</td>
      <td>KIAD</td>
      <td>large_airport</td>
      <td>Washington Dulles International Airport</td>
      <td>38.944500</td>
      <td>-77.455803</td>
      <td>312.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-DC</td>
      <td>Washington</td>
      <td>1</td>
      <td>KIAD</td>
      <td>IAD</td>
      <td>IAD</td>
      <td>http://www.mwaa.com/dulles/</td>
      <td>http://en.wikipedia.org/wiki/Washington_Dulles...</td>
      <td>WAS</td>
      <td>1023875</td>
      <td>2010-01-23T11:27:53+00:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3643</td>
      <td>KLGA</td>
      <td>large_airport</td>
      <td>La Guardia Airport</td>
      <td>40.777199</td>
      <td>-73.872597</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NY</td>
      <td>New York</td>
      <td>1</td>
      <td>KLGA</td>
      <td>LGA</td>
      <td>LGA</td>
      <td>http://www.panynj.gov/CommutingTravel/airports...</td>
      <td>http://en.wikipedia.org/wiki/LaGuardia_Airport</td>
      <td>Manhattan, New York City, NYC, Glenn H. Curtis...</td>
      <td>1030575</td>
      <td>2010-01-23T11:27:55+00:00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3772</td>
      <td>KPHX</td>
      <td>large_airport</td>
      <td>Phoenix Sky Harbor International Airport</td>
      <td>33.434299</td>
      <td>-112.012001</td>
      <td>1135.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-AZ</td>
      <td>Phoenix</td>
      <td>1</td>
      <td>KPHX</td>
      <td>PHX</td>
      <td>PHX</td>
      <td>http://phoenix.gov/skyharborairport/</td>
      <td>http://en.wikipedia.org/wiki/Phoenix_Sky_Harbo...</td>
      <td>NaN</td>
      <td>1060875</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2429</td>
      <td>EGKK</td>
      <td>large_airport</td>
      <td>London Gatwick Airport</td>
      <td>51.148102</td>
      <td>-0.190278</td>
      <td>202.0</td>
      <td>EU</td>
      <td>GB</td>
      <td>GB-ENG</td>
      <td>London</td>
      <td>1</td>
      <td>EGKK</td>
      <td>LGW</td>
      <td>NaN</td>
      <td>http://www.gatwickairport.com/</td>
      <td>http://en.wikipedia.org/wiki/London_Gatwick_Ai...</td>
      <td>LON, Crawley, Charlwood</td>
      <td>1049275</td>
      <td>2009-01-30T17:53:54+00:00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3875</td>
      <td>KSEA</td>
      <td>large_airport</td>
      <td>Seattle Tacoma International Airport</td>
      <td>47.449001</td>
      <td>-122.308998</td>
      <td>433.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-WA</td>
      <td>Seattle</td>
      <td>1</td>
      <td>KSEA</td>
      <td>SEA</td>
      <td>SEA</td>
      <td>http://www.portseattle.org/seatac/</td>
      <td>http://en.wikipedia.org/wiki/Seattle-Tacoma_In...</td>
      <td>NaN</td>
      <td>1038675</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3422</td>
      <td>KBOS</td>
      <td>large_airport</td>
      <td>General Edward Lawrence Logan International Ai...</td>
      <td>42.364300</td>
      <td>-71.005203</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-MA</td>
      <td>Boston</td>
      <td>1</td>
      <td>KBOS</td>
      <td>BOS</td>
      <td>BOS</td>
      <td>http://www.massport.com/logan/</td>
      <td>http://en.wikipedia.org/wiki/Logan_Internation...</td>
      <td>General Edward Lawrence Logan International Ai...</td>
      <td>1035875</td>
      <td>2010-01-23T11:27:48+00:00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3771</td>
      <td>KPHL</td>
      <td>large_airport</td>
      <td>Philadelphia International Airport</td>
      <td>39.871899</td>
      <td>-75.241096</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-PA</td>
      <td>Philadelphia</td>
      <td>1</td>
      <td>KPHL</td>
      <td>PHL</td>
      <td>PHL</td>
      <td>http://www.phl.org/</td>
      <td>http://en.wikipedia.org/wiki/Philadelphia_Inte...</td>
      <td>NaN</td>
      <td>1031075</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1990</td>
      <td>CYYZ</td>
      <td>large_airport</td>
      <td>Lester B. Pearson International Airport</td>
      <td>43.677200</td>
      <td>-79.630600</td>
      <td>569.0</td>
      <td>NaN</td>
      <td>CA</td>
      <td>CA-ON</td>
      <td>Toronto</td>
      <td>1</td>
      <td>CYYZ</td>
      <td>YYZ</td>
      <td>YYZ</td>
      <td>http://www.gtaa.com/</td>
      <td>http://en.wikipedia.org/wiki/Toronto_Pearson_I...</td>
      <td>YTO, Toronto International Airport, Malton</td>
      <td>1040275</td>
      <td>2009-10-15T13:44:39+00:00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3483</td>
      <td>KDCA</td>
      <td>large_airport</td>
      <td>Ronald Reagan Washington National Airport</td>
      <td>38.852100</td>
      <td>-77.037697</td>
      <td>15.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-DC</td>
      <td>Washington</td>
      <td>1</td>
      <td>KDCA</td>
      <td>DCA</td>
      <td>DCA</td>
      <td>http://www.flyreagan.com/dca/reagan-national-a...</td>
      <td>http://en.wikipedia.org/wiki/Ronald_Reagan_Was...</td>
      <td>WAS</td>
      <td>1016175</td>
      <td>2016-12-13T09:13:00+00:00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3604</td>
      <td>KIAH</td>
      <td>large_airport</td>
      <td>George Bush Intercontinental Houston Airport</td>
      <td>29.984400</td>
      <td>-95.341400</td>
      <td>97.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-TX</td>
      <td>Houston</td>
      <td>1</td>
      <td>KIAH</td>
      <td>IAH</td>
      <td>IAH</td>
      <td>http://www.fly2houston.com/iah</td>
      <td>http://en.wikipedia.org/wiki/George_Bush_Inter...</td>
      <td>QHO</td>
      <td>1055675</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3457</td>
      <td>KCLT</td>
      <td>large_airport</td>
      <td>Charlotte Douglas International Airport</td>
      <td>35.214001</td>
      <td>-80.943100</td>
      <td>748.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NC</td>
      <td>Charlotte</td>
      <td>1</td>
      <td>KCLT</td>
      <td>CLT</td>
      <td>CLT</td>
      <td>http://www.charlotteairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Charlotte/Douglas...</td>
      <td>NaN</td>
      <td>1028075</td>
      <td>2009-01-30T14:52:52+00:00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3497</td>
      <td>KDTW</td>
      <td>large_airport</td>
      <td>Detroit Metropolitan Wayne County Airport</td>
      <td>42.212399</td>
      <td>-83.353401</td>
      <td>645.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-MI</td>
      <td>Detroit</td>
      <td>1</td>
      <td>KDTW</td>
      <td>DTW</td>
      <td>DTW</td>
      <td>http://www.metroairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Detroit_Metropoli...</td>
      <td>DTT, Detroit Metro Airport</td>
      <td>1079375</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2218</td>
      <td>EDDM</td>
      <td>large_airport</td>
      <td>Munich International Airport</td>
      <td>48.353802</td>
      <td>11.786100</td>
      <td>1487.0</td>
      <td>EU</td>
      <td>DE</td>
      <td>DE-BY</td>
      <td>Munich</td>
      <td>1</td>
      <td>EDDM</td>
      <td>MUC</td>
      <td>NaN</td>
      <td>http://www.munich-airport.com/</td>
      <td>http://en.wikipedia.org/wiki/Munich_Internatio...</td>
      <td>Franz Josef Strauss International Airport, Flu...</td>
      <td>1026675</td>
      <td>2015-10-17T03:56:24+00:00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3709</td>
      <td>KMSP</td>
      <td>large_airport</td>
      <td>Minneapolis-St Paul International/Wold-Chamber...</td>
      <td>44.882000</td>
      <td>-93.221802</td>
      <td>841.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-MN</td>
      <td>Minneapolis</td>
      <td>1</td>
      <td>KMSP</td>
      <td>MSP</td>
      <td>MSP</td>
      <td>http://www.mspairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Minneapolis-Saint...</td>
      <td>NaN</td>
      <td>1085275</td>
      <td>2009-12-17T08:53:11+00:00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4372</td>
      <td>LIRF</td>
      <td>large_airport</td>
      <td>Leonardo da Vinci–Fiumicino Airport</td>
      <td>41.800278</td>
      <td>12.238889</td>
      <td>13.0</td>
      <td>EU</td>
      <td>IT</td>
      <td>IT-62</td>
      <td>Rome</td>
      <td>1</td>
      <td>LIRF</td>
      <td>FCO</td>
      <td>RM11</td>
      <td>http://www.adr.it/portal/portal/adr/Fiumicino/...</td>
      <td>http://en.wikipedia.org/wiki/Leonardo_da_Vinci...</td>
      <td>ROM, Rome Fiumicino Airport, Fiumicino Airport</td>
      <td>1031875</td>
      <td>2016-10-08T00:37:02+00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6695</th>
      <td>3567</td>
      <td>KGSB</td>
      <td>large_airport</td>
      <td>Seymour Johnson Air Force Base</td>
      <td>35.339401</td>
      <td>-77.960602</td>
      <td>109.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NC</td>
      <td>Goldsboro</td>
      <td>0</td>
      <td>KGSB</td>
      <td>GSB</td>
      <td>GSB</td>
      <td>http://www.seymourjohnson.af.mil/</td>
      <td>http://en.wikipedia.org/wiki/Seymour_Johnson_A...</td>
      <td>NaN</td>
      <td>850</td>
      <td>2014-10-04T01:51:25+00:00</td>
    </tr>
    <tr>
      <th>6696</th>
      <td>3914</td>
      <td>KSZL</td>
      <td>large_airport</td>
      <td>Whiteman Air Force Base</td>
      <td>38.730301</td>
      <td>-93.547897</td>
      <td>870.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-MO</td>
      <td>Knob Noster</td>
      <td>0</td>
      <td>KSZL</td>
      <td>SZL</td>
      <td>SZL</td>
      <td>http://www.whiteman.af.mil/</td>
      <td>http://en.wikipedia.org/wiki/Whiteman_Air_Forc...</td>
      <td>Sedalia Glider Base</td>
      <td>850</td>
      <td>2015-05-31T03:54:50+00:00</td>
    </tr>
    <tr>
      <th>6697</th>
      <td>3391</td>
      <td>KBAB</td>
      <td>large_airport</td>
      <td>Beale Air Force Base</td>
      <td>39.136101</td>
      <td>-121.436996</td>
      <td>113.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CA</td>
      <td>Marysville</td>
      <td>0</td>
      <td>KBAB</td>
      <td>BAB</td>
      <td>BAB</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Beale_Air_Force_Base</td>
      <td>NaN</td>
      <td>850</td>
      <td>2012-06-21T01:43:23+00:00</td>
    </tr>
    <tr>
      <th>6698</th>
      <td>3715</td>
      <td>KMUO</td>
      <td>large_airport</td>
      <td>Mountain Home Air Force Base</td>
      <td>43.043598</td>
      <td>-115.872002</td>
      <td>2996.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-ID</td>
      <td>Mountain Home</td>
      <td>0</td>
      <td>KMUO</td>
      <td>MUO</td>
      <td>MUO</td>
      <td>http://www.mountainhome.af.mil/</td>
      <td>https://en.wikipedia.org/wiki/Mountain_Home_Ai...</td>
      <td>NaN</td>
      <td>850</td>
      <td>2016-05-31T09:36:32+00:00</td>
    </tr>
    <tr>
      <th>6699</th>
      <td>3573</td>
      <td>KGUS</td>
      <td>large_airport</td>
      <td>Grissom Air Reserve Base</td>
      <td>40.648102</td>
      <td>-86.152100</td>
      <td>812.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-IN</td>
      <td>Peru</td>
      <td>0</td>
      <td>KGUS</td>
      <td>GUS</td>
      <td>GUS</td>
      <td>http://www.grissom.afrc.af.mil/</td>
      <td>http://en.wikipedia.org/wiki/Grissom_Air_Reser...</td>
      <td>NaN</td>
      <td>850</td>
      <td>2014-10-03T23:32:05+00:00</td>
    </tr>
    <tr>
      <th>6700</th>
      <td>5510</td>
      <td>RCBS</td>
      <td>large_airport</td>
      <td>Kinmen Airport</td>
      <td>24.427900</td>
      <td>118.359001</td>
      <td>93.0</td>
      <td>AS</td>
      <td>TW</td>
      <td>TW-X-KM</td>
      <td>Shang-I</td>
      <td>1</td>
      <td>RCBS</td>
      <td>KNH</td>
      <td>NaN</td>
      <td>http://www.kma.gov.tw/index-e.htm</td>
      <td>http://en.wikipedia.org/wiki/Kinmen_Airport</td>
      <td>Shang Yi Airport, 金門尚義機場, 金门尚义机场, Jīnmén Shàng...</td>
      <td>850</td>
      <td>2009-07-11T09:13:57+00:00</td>
    </tr>
    <tr>
      <th>7035</th>
      <td>6061</td>
      <td>SELT</td>
      <td>large_airport</td>
      <td>Cotopaxi International Airport</td>
      <td>-0.906833</td>
      <td>-78.615799</td>
      <td>9205.0</td>
      <td>SA</td>
      <td>EC</td>
      <td>EC-X</td>
      <td>Latacunga</td>
      <td>1</td>
      <td>SELT</td>
      <td>LTX</td>
      <td>NaN</td>
      <td>http://www.dgac.gob.ec/latacunga/</td>
      <td>http://es.wikipedia.org/wiki/Aeropuerto_Intern...</td>
      <td>Latacunga, Vulcano, Ecuador, Cargo, Long runwa...</td>
      <td>400</td>
      <td>2013-02-25T10:27:48+00:00</td>
    </tr>
    <tr>
      <th>10197</th>
      <td>42924</td>
      <td>RU-0035</td>
      <td>large_airport</td>
      <td>Grozny North Airport</td>
      <td>43.388302</td>
      <td>45.698601</td>
      <td>548.0</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-CE</td>
      <td>Grozny</td>
      <td>1</td>
      <td>URMG</td>
      <td>GRV</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://en.wikipedia.org/wiki/Grozny_Airport</td>
      <td>Grozny Severny, Grozny, Аэропорт Грозный Север...</td>
      <td>0</td>
      <td>2016-05-31T10:29:00+00:00</td>
    </tr>
    <tr>
      <th>10260</th>
      <td>299238</td>
      <td>rjns</td>
      <td>large_airport</td>
      <td>Mt. Fuji Shizuoka Airport</td>
      <td>34.796043</td>
      <td>138.187752</td>
      <td>433.0</td>
      <td>AS</td>
      <td>JP</td>
      <td>JP-22</td>
      <td>Makinohara / Shimada</td>
      <td>1</td>
      <td>RJNS</td>
      <td>FSZ</td>
      <td>NaN</td>
      <td>http://www.mtfuji-shizuokaairport.jp/english/i...</td>
      <td>http://en.wikipedia.org/wiki/Shizuoka_Airport</td>
      <td>NaN</td>
      <td>0</td>
      <td>2013-09-29T13:51:58+00:00</td>
    </tr>
    <tr>
      <th>10316</th>
      <td>320984</td>
      <td>GB-0458</td>
      <td>large_airport</td>
      <td>[Duplicate] Pic</td>
      <td>55.505694</td>
      <td>-4.586792</td>
      <td>NaN</td>
      <td>EU</td>
      <td>GB</td>
      <td>GB-SCT</td>
      <td>Prestwick</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-11-12T08:02:29+00:00</td>
    </tr>
    <tr>
      <th>10381</th>
      <td>309668</td>
      <td>DNAI</td>
      <td>large_airport</td>
      <td>Akwa Ibom International Airport</td>
      <td>4.872500</td>
      <td>8.093000</td>
      <td>170.0</td>
      <td>AF</td>
      <td>NG</td>
      <td>NG-AK</td>
      <td>Uyo</td>
      <td>1</td>
      <td>DNAI</td>
      <td>QUO</td>
      <td>NaN</td>
      <td>http://www.ibomairport.com.ng/</td>
      <td>http://en.wikipedia.org/wiki/Akwa_Ibom_Airport</td>
      <td>NaN</td>
      <td>0</td>
      <td>2013-09-28T12:53:18+00:00</td>
    </tr>
    <tr>
      <th>10407</th>
      <td>27197</td>
      <td>ZGNN</td>
      <td>large_airport</td>
      <td>Nanning Wuxu Airport</td>
      <td>22.608299</td>
      <td>108.171997</td>
      <td>421.0</td>
      <td>AS</td>
      <td>CN</td>
      <td>CN-45</td>
      <td>Nanning</td>
      <td>1</td>
      <td>ZGNN</td>
      <td>NNG</td>
      <td>NaN</td>
      <td>http://www.nnairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Nanning_Wuxu_Inte...</td>
      <td>南宁吴圩机场, Nanning Wuxu Air Base</td>
      <td>1200</td>
      <td>2009-04-16T20:31:13+00:00</td>
    </tr>
    <tr>
      <th>10418</th>
      <td>27195</td>
      <td>ZGHA</td>
      <td>large_airport</td>
      <td>Changsha Huanghua International Airport</td>
      <td>28.189199</td>
      <td>113.220001</td>
      <td>217.0</td>
      <td>AS</td>
      <td>CN</td>
      <td>CN-43</td>
      <td>Changsha</td>
      <td>1</td>
      <td>ZGHA</td>
      <td>CSX</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Changsha_Huanghua...</td>
      <td>长沙黄花国际机场</td>
      <td>1050</td>
      <td>2013-05-17T01:38:00+00:00</td>
    </tr>
    <tr>
      <th>10500</th>
      <td>4168</td>
      <td>LFOA</td>
      <td>large_airport</td>
      <td>Avord (BA 702) Air Base</td>
      <td>47.053299</td>
      <td>2.632500</td>
      <td>580.0</td>
      <td>EU</td>
      <td>FR</td>
      <td>FR-F</td>
      <td>Avord</td>
      <td>0</td>
      <td>LFOA</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://en.wikipedia.org/wiki/Avord_Air_Base</td>
      <td>NaN</td>
      <td>850</td>
      <td>2016-05-31T09:37:29+00:00</td>
    </tr>
    <tr>
      <th>10501</th>
      <td>35102</td>
      <td>RU-2244</td>
      <td>large_airport</td>
      <td>Ukrainka Air Base</td>
      <td>51.169997</td>
      <td>128.445007</td>
      <td>771.0</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-AMU</td>
      <td>Belogorsk</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XHBU</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Ukrainka_(air_base)</td>
      <td>ЬХБУ, Украинка</td>
      <td>850</td>
      <td>2017-01-20T18:03:03+00:00</td>
    </tr>
    <tr>
      <th>10502</th>
      <td>30601</td>
      <td>PK-0001</td>
      <td>large_airport</td>
      <td>(Under construction) New Islamabad Internation...</td>
      <td>33.560714</td>
      <td>72.851614</td>
      <td>646.0</td>
      <td>AS</td>
      <td>PK</td>
      <td>PK-PB</td>
      <td>Islamabad</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://www.niia.com.pk/</td>
      <td>http://en.wikipedia.org/wiki/New_Islamabad_Int...</td>
      <td>Fateh Jang</td>
      <td>850</td>
      <td>2010-12-28T15:14:35+00:00</td>
    </tr>
    <tr>
      <th>10503</th>
      <td>41136</td>
      <td>RU-0016</td>
      <td>large_airport</td>
      <td>Lipetsk Air Base</td>
      <td>52.634998</td>
      <td>39.445000</td>
      <td>636.0</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-LIP</td>
      <td>Lipetsk</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Lipetsk_Air_Base</td>
      <td>NaN</td>
      <td>850</td>
      <td>2008-08-11T13:10:57+00:00</td>
    </tr>
    <tr>
      <th>10504</th>
      <td>35046</td>
      <td>RU-4464</td>
      <td>large_airport</td>
      <td>Olenya Air Base</td>
      <td>68.151802</td>
      <td>33.463902</td>
      <td>702.0</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-MUR</td>
      <td>Olenegorsk</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>XLMO</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Olenya_(air_base)</td>
      <td>ЬЛМО, Оленья, Olenegorsk</td>
      <td>850</td>
      <td>2015-09-25T10:50:57+00:00</td>
    </tr>
    <tr>
      <th>10506</th>
      <td>42967</td>
      <td>UUBW</td>
      <td>large_airport</td>
      <td>Zhukovsky International Airport</td>
      <td>55.553299</td>
      <td>38.150002</td>
      <td>377.0</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-MOS</td>
      <td>Zhukovsky</td>
      <td>1</td>
      <td>UUBW</td>
      <td>ZIA</td>
      <td>NaN</td>
      <td>http://zia.aero/</td>
      <td>https://en.wikipedia.org/wiki/Zhukovsky_Intern...</td>
      <td>Аэропорт Раменское, Zhukovsky Air Base, Gromov...</td>
      <td>850</td>
      <td>2016-10-05T13:48:10+00:00</td>
    </tr>
    <tr>
      <th>11631</th>
      <td>34954</td>
      <td>UWKG</td>
      <td>large_airport</td>
      <td>Borisoglebskoye Airport</td>
      <td>55.866699</td>
      <td>49.133301</td>
      <td>213.0</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-TA</td>
      <td>Kazan</td>
      <td>0</td>
      <td>UWKG</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Borisoglebskoye_A...</td>
      <td>Аэропорт Борисоглебское</td>
      <td>250</td>
      <td>2015-08-21T19:49:50+00:00</td>
    </tr>
    <tr>
      <th>41518</th>
      <td>309579</td>
      <td>VCRI</td>
      <td>large_airport</td>
      <td>Mattala Rajapaksa International Airport</td>
      <td>6.284467</td>
      <td>81.124128</td>
      <td>157.0</td>
      <td>AS</td>
      <td>LK</td>
      <td>LK-3</td>
      <td>NaN</td>
      <td>1</td>
      <td>VCRI</td>
      <td>HRI</td>
      <td>VCRI</td>
      <td>http://www.mria.lk/</td>
      <td>http://en.wikipedia.org/wiki/Mattala_Rajapaksa...</td>
      <td>Hambantota International Airport</td>
      <td>0</td>
      <td>2015-10-06T05:20:29+00:00</td>
    </tr>
    <tr>
      <th>42016</th>
      <td>317314</td>
      <td>US-0303</td>
      <td>large_airport</td>
      <td>atl</td>
      <td>33.137551</td>
      <td>-84.375000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-U-A</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-03-19T09:20:40+00:00</td>
    </tr>
    <tr>
      <th>42399</th>
      <td>320983</td>
      <td>CA-0671</td>
      <td>large_airport</td>
      <td>Cyll</td>
      <td>53.310365</td>
      <td>-110.072021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA</td>
      <td>CA-U-A</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-10-31T22:20:17+00:00</td>
    </tr>
    <tr>
      <th>42571</th>
      <td>322136</td>
      <td>YYZ</td>
      <td>large_airport</td>
      <td>Toronto Pearson International Airport</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA</td>
      <td>CA-ON</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>YYZ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2017-02-12T14:34:35+00:00</td>
    </tr>
    <tr>
      <th>42689</th>
      <td>299747</td>
      <td>OM-0004</td>
      <td>large_airport</td>
      <td>Rustaq Airport</td>
      <td>23.640556</td>
      <td>57.487500</td>
      <td>349.0</td>
      <td>AS</td>
      <td>OM</td>
      <td>OM-BA</td>
      <td>Al Masna'ah</td>
      <td>0</td>
      <td>OORQ</td>
      <td>MNH</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://en.wikipedia.org/wiki/Rustaq_Airport</td>
      <td>Musanaa, Wadi al Maawil, Al Muladdah</td>
      <td>0</td>
      <td>2016-03-16T00:34:25+00:00</td>
    </tr>
    <tr>
      <th>43270</th>
      <td>44093</td>
      <td>ZBNY</td>
      <td>large_airport</td>
      <td>Beijing Nanyuan Airport</td>
      <td>39.782799</td>
      <td>116.388000</td>
      <td>NaN</td>
      <td>AS</td>
      <td>CN</td>
      <td>CN-11</td>
      <td>Beijing</td>
      <td>1</td>
      <td>ZBNY</td>
      <td>NAY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Nanyuan_Airport</td>
      <td>NaN</td>
      <td>0</td>
      <td>2009-08-30T19:53:52+00:00</td>
    </tr>
    <tr>
      <th>44663</th>
      <td>299738</td>
      <td>OM-0003</td>
      <td>large_airport</td>
      <td>Duqm Jaaluni Airport</td>
      <td>19.500000</td>
      <td>57.642800</td>
      <td>380.0</td>
      <td>AS</td>
      <td>OM</td>
      <td>OM-WU</td>
      <td>Duqm</td>
      <td>1</td>
      <td>OODQ</td>
      <td>JNJ</td>
      <td>NaN</td>
      <td>http://www.omanairports.com/</td>
      <td>https://en.wikipedia.org/wiki/Duqm_Jaaluni_Air...</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-05-31T09:45:53+00:00</td>
    </tr>
    <tr>
      <th>47391</th>
      <td>302303</td>
      <td>RKJB</td>
      <td>large_airport</td>
      <td>Muan International Airport</td>
      <td>34.991406</td>
      <td>126.382814</td>
      <td>35.0</td>
      <td>AS</td>
      <td>KR</td>
      <td>KR-46</td>
      <td>Muan</td>
      <td>1</td>
      <td>RKJB</td>
      <td>MWX</td>
      <td>NaN</td>
      <td>http://muan.airport.co.kr/doc/muan_eng/</td>
      <td>http://en.wikipedia.org/wiki/Muan_Internationa...</td>
      <td>Gwangju, Mokpo</td>
      <td>0</td>
      <td>2015-08-19T02:40:29+00:00</td>
    </tr>
    <tr>
      <th>48343</th>
      <td>45229</td>
      <td>AMC</td>
      <td>large_airport</td>
      <td>Mar de Cortés International Airport</td>
      <td>31.351621</td>
      <td>-113.305864</td>
      <td>71.0</td>
      <td>NaN</td>
      <td>MX</td>
      <td>MX-SON</td>
      <td>Puerto Peñasco</td>
      <td>1</td>
      <td>MMPE</td>
      <td>PPE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Mar_de_Cort%C3%A9...</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-05-31T10:21:05+00:00</td>
    </tr>
    <tr>
      <th>48768</th>
      <td>309298</td>
      <td>ID-0017</td>
      <td>large_airport</td>
      <td>Sungai Siring International Airport</td>
      <td>-0.373611</td>
      <td>117.255556</td>
      <td>82.0</td>
      <td>AS</td>
      <td>ID</td>
      <td>ID-KI</td>
      <td>Samarinda</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://www.proyek.dishubprovkaltim.info/?pro=1</td>
      <td>http://en.wikipedia.org/wiki/Samarinda_Interna...</td>
      <td>Sungai Siring Airport, فرودگاه سیرینگ</td>
      <td>0</td>
      <td>2015-02-15T02:52:02+00:00</td>
    </tr>
  </tbody>
</table>
<p>574 rows × 20 columns</p>
</div>
</div>


</div>
</div>
</div>

Since both latitude and longitude are columns in the dataframe we can use pandas to carry out a limited number of geospatial queries. For example, extract all the airports in the northern hemisphere:

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
df[df.latitude_deg > 0.0]
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ident</th>
      <th>type</th>
      <th>name</th>
      <th>latitude_deg</th>
      <th>longitude_deg</th>
      <th>elevation_ft</th>
      <th>continent</th>
      <th>iso_country</th>
      <th>iso_region</th>
      <th>municipality</th>
      <th>scheduled_service</th>
      <th>gps_code</th>
      <th>iata_code</th>
      <th>local_code</th>
      <th>home_link</th>
      <th>wikipedia_link</th>
      <th>keywords</th>
      <th>score</th>
      <th>last_updated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2434</td>
      <td>EGLL</td>
      <td>large_airport</td>
      <td>London Heathrow Airport</td>
      <td>51.470600</td>
      <td>-0.461941</td>
      <td>83.0</td>
      <td>EU</td>
      <td>GB</td>
      <td>GB-ENG</td>
      <td>London</td>
      <td>1</td>
      <td>EGLL</td>
      <td>LHR</td>
      <td>NaN</td>
      <td>http://www.heathrowairport.com/</td>
      <td>http://en.wikipedia.org/wiki/London_Heathrow_A...</td>
      <td>LON, Londres</td>
      <td>1251675</td>
      <td>2016-02-09T10:54:31+00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3632</td>
      <td>KLAX</td>
      <td>large_airport</td>
      <td>Los Angeles International Airport</td>
      <td>33.942501</td>
      <td>-118.407997</td>
      <td>125.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CA</td>
      <td>Los Angeles</td>
      <td>1</td>
      <td>KLAX</td>
      <td>LAX</td>
      <td>LAX</td>
      <td>http://www.iflylax.com/</td>
      <td>http://en.wikipedia.org/wiki/Los_Angeles_Inter...</td>
      <td>NaN</td>
      <td>1335475</td>
      <td>2010-01-23T11:27:55+00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3754</td>
      <td>KORD</td>
      <td>large_airport</td>
      <td>Chicago O'Hare International Airport</td>
      <td>41.978600</td>
      <td>-87.904800</td>
      <td>672.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-IL</td>
      <td>Chicago</td>
      <td>1</td>
      <td>KORD</td>
      <td>ORD</td>
      <td>ORD</td>
      <td>http://www.flychicago.com/ohare/en/home/Pages/...</td>
      <td>http://en.wikipedia.org/wiki/O'Hare_Internatio...</td>
      <td>CHI, Orchard Place</td>
      <td>1503175</td>
      <td>2014-08-19T18:55:27+00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3622</td>
      <td>KJFK</td>
      <td>large_airport</td>
      <td>John F Kennedy International Airport</td>
      <td>40.639801</td>
      <td>-73.778900</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NY</td>
      <td>New York</td>
      <td>1</td>
      <td>KJFK</td>
      <td>JFK</td>
      <td>JFK</td>
      <td>http://www.panynj.gov/CommutingTravel/airports...</td>
      <td>http://en.wikipedia.org/wiki/John_F._Kennedy_I...</td>
      <td>Manhattan, New York City, NYC, Idlewild</td>
      <td>1052075</td>
      <td>2010-01-23T11:27:54+00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3384</td>
      <td>KATL</td>
      <td>large_airport</td>
      <td>Hartsfield Jackson Atlanta International Airport</td>
      <td>33.636700</td>
      <td>-84.428101</td>
      <td>1026.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-GA</td>
      <td>Atlanta</td>
      <td>1</td>
      <td>KATL</td>
      <td>ATL</td>
      <td>ATL</td>
      <td>http://www.atlanta-airport.com/</td>
      <td>http://en.wikipedia.org/wiki/Hartsfield-Jackso...</td>
      <td>NaN</td>
      <td>2002475</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4185</td>
      <td>LFPG</td>
      <td>large_airport</td>
      <td>Charles de Gaulle International Airport</td>
      <td>49.012798</td>
      <td>2.550000</td>
      <td>392.0</td>
      <td>EU</td>
      <td>FR</td>
      <td>FR-J</td>
      <td>Paris</td>
      <td>1</td>
      <td>LFPG</td>
      <td>CDG</td>
      <td>NaN</td>
      <td>http://www.aeroportsdeparis.fr/</td>
      <td>http://en.wikipedia.org/wiki/Charles_de_Gaulle...</td>
      <td>PAR, Aéroport Roissy-Charles de Gaulle, Roissy...</td>
      <td>1127475</td>
      <td>2010-01-05T18:17:43+00:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3878</td>
      <td>KSFO</td>
      <td>large_airport</td>
      <td>San Francisco International Airport</td>
      <td>37.618999</td>
      <td>-122.375000</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CA</td>
      <td>San Francisco</td>
      <td>1</td>
      <td>KSFO</td>
      <td>SFO</td>
      <td>SFO</td>
      <td>http://www.flysfo.com/</td>
      <td>http://en.wikipedia.org/wiki/San_Francisco_Int...</td>
      <td>QSF, QBA</td>
      <td>1112475</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2513</td>
      <td>EHAM</td>
      <td>large_airport</td>
      <td>Amsterdam Airport Schiphol</td>
      <td>52.308601</td>
      <td>4.763890</td>
      <td>-11.0</td>
      <td>EU</td>
      <td>NL</td>
      <td>NL-NH</td>
      <td>Amsterdam</td>
      <td>1</td>
      <td>EHAM</td>
      <td>AMS</td>
      <td>NaN</td>
      <td>http://www.schiphol.nl/</td>
      <td>http://en.wikipedia.org/wiki/Amsterdam_Schipho...</td>
      <td>NaN</td>
      <td>1093275</td>
      <td>2011-10-05T23:26:15+00:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2212</td>
      <td>EDDF</td>
      <td>large_airport</td>
      <td>Frankfurt am Main International Airport</td>
      <td>50.033333</td>
      <td>8.570556</td>
      <td>364.0</td>
      <td>EU</td>
      <td>DE</td>
      <td>DE-HE</td>
      <td>Frankfurt-am-Main</td>
      <td>1</td>
      <td>EDDF</td>
      <td>FRA</td>
      <td>NaN</td>
      <td>http://www.frankfurt-airport.de/</td>
      <td>http://en.wikipedia.org/wiki/Frankfurt_Airport</td>
      <td>EDAF, Rhein-Main Air Base</td>
      <td>1144675</td>
      <td>2016-09-15T03:34:21+00:00</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3521</td>
      <td>KEWR</td>
      <td>large_airport</td>
      <td>Newark Liberty International Airport</td>
      <td>40.692501</td>
      <td>-74.168701</td>
      <td>18.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NJ</td>
      <td>Newark</td>
      <td>1</td>
      <td>KEWR</td>
      <td>EWR</td>
      <td>EWR</td>
      <td>http://www.panynj.gov/CommutingTravel/airports...</td>
      <td>http://en.wikipedia.org/wiki/Newark_Liberty_In...</td>
      <td>Manhattan, New York City, NYC</td>
      <td>1064475</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3631</td>
      <td>KLAS</td>
      <td>large_airport</td>
      <td>McCarran International Airport</td>
      <td>36.080101</td>
      <td>-115.152000</td>
      <td>2181.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NV</td>
      <td>Las Vegas</td>
      <td>1</td>
      <td>KLAS</td>
      <td>LAS</td>
      <td>LAS</td>
      <td>http://www.mccarran.com/</td>
      <td>http://en.wikipedia.org/wiki/McCarran_Internat...</td>
      <td>NaN</td>
      <td>1068475</td>
      <td>2010-07-05T16:48:39+00:00</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3488</td>
      <td>KDFW</td>
      <td>large_airport</td>
      <td>Dallas Fort Worth International Airport</td>
      <td>32.896801</td>
      <td>-97.038002</td>
      <td>607.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-TX</td>
      <td>Dallas-Fort Worth</td>
      <td>1</td>
      <td>KDFW</td>
      <td>DFW</td>
      <td>DFW</td>
      <td>http://www.dfwairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Dallas-Fort_Worth...</td>
      <td>QDF</td>
      <td>1203175</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3670</td>
      <td>KMCO</td>
      <td>large_airport</td>
      <td>Orlando International Airport</td>
      <td>28.429399</td>
      <td>-81.308998</td>
      <td>96.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-FL</td>
      <td>Orlando</td>
      <td>1</td>
      <td>KMCO</td>
      <td>MCO</td>
      <td>MCO</td>
      <td>http://www.orlandoairports.net/</td>
      <td>http://en.wikipedia.org/wiki/Orlando_Internati...</td>
      <td>Disney World,Epcot Center</td>
      <td>1044075</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3486</td>
      <td>KDEN</td>
      <td>large_airport</td>
      <td>Denver International Airport</td>
      <td>39.861698</td>
      <td>-104.672997</td>
      <td>5431.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-CO</td>
      <td>Denver</td>
      <td>1</td>
      <td>KDEN</td>
      <td>DEN</td>
      <td>DEN</td>
      <td>http://www.flydenver.com/</td>
      <td>http://en.wikipedia.org/wiki/Denver_Internatio...</td>
      <td>DVX, KVDX</td>
      <td>1103275</td>
      <td>2015-11-13T09:28:42+00:00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3685</td>
      <td>KMIA</td>
      <td>large_airport</td>
      <td>Miami International Airport</td>
      <td>25.793200</td>
      <td>-80.290604</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-FL</td>
      <td>Miami</td>
      <td>1</td>
      <td>KMIA</td>
      <td>MIA</td>
      <td>MIA</td>
      <td>http://www.miami-airport.com/</td>
      <td>http://en.wikipedia.org/wiki/Miami_Internation...</td>
      <td>MFW, South Florida</td>
      <td>1073075</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3602</td>
      <td>KIAD</td>
      <td>large_airport</td>
      <td>Washington Dulles International Airport</td>
      <td>38.944500</td>
      <td>-77.455803</td>
      <td>312.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-DC</td>
      <td>Washington</td>
      <td>1</td>
      <td>KIAD</td>
      <td>IAD</td>
      <td>IAD</td>
      <td>http://www.mwaa.com/dulles/</td>
      <td>http://en.wikipedia.org/wiki/Washington_Dulles...</td>
      <td>WAS</td>
      <td>1023875</td>
      <td>2010-01-23T11:27:53+00:00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3643</td>
      <td>KLGA</td>
      <td>large_airport</td>
      <td>La Guardia Airport</td>
      <td>40.777199</td>
      <td>-73.872597</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NY</td>
      <td>New York</td>
      <td>1</td>
      <td>KLGA</td>
      <td>LGA</td>
      <td>LGA</td>
      <td>http://www.panynj.gov/CommutingTravel/airports...</td>
      <td>http://en.wikipedia.org/wiki/LaGuardia_Airport</td>
      <td>Manhattan, New York City, NYC, Glenn H. Curtis...</td>
      <td>1030575</td>
      <td>2010-01-23T11:27:55+00:00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3772</td>
      <td>KPHX</td>
      <td>large_airport</td>
      <td>Phoenix Sky Harbor International Airport</td>
      <td>33.434299</td>
      <td>-112.012001</td>
      <td>1135.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-AZ</td>
      <td>Phoenix</td>
      <td>1</td>
      <td>KPHX</td>
      <td>PHX</td>
      <td>PHX</td>
      <td>http://phoenix.gov/skyharborairport/</td>
      <td>http://en.wikipedia.org/wiki/Phoenix_Sky_Harbo...</td>
      <td>NaN</td>
      <td>1060875</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2429</td>
      <td>EGKK</td>
      <td>large_airport</td>
      <td>London Gatwick Airport</td>
      <td>51.148102</td>
      <td>-0.190278</td>
      <td>202.0</td>
      <td>EU</td>
      <td>GB</td>
      <td>GB-ENG</td>
      <td>London</td>
      <td>1</td>
      <td>EGKK</td>
      <td>LGW</td>
      <td>NaN</td>
      <td>http://www.gatwickairport.com/</td>
      <td>http://en.wikipedia.org/wiki/London_Gatwick_Ai...</td>
      <td>LON, Crawley, Charlwood</td>
      <td>1049275</td>
      <td>2009-01-30T17:53:54+00:00</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3875</td>
      <td>KSEA</td>
      <td>large_airport</td>
      <td>Seattle Tacoma International Airport</td>
      <td>47.449001</td>
      <td>-122.308998</td>
      <td>433.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-WA</td>
      <td>Seattle</td>
      <td>1</td>
      <td>KSEA</td>
      <td>SEA</td>
      <td>SEA</td>
      <td>http://www.portseattle.org/seatac/</td>
      <td>http://en.wikipedia.org/wiki/Seattle-Tacoma_In...</td>
      <td>NaN</td>
      <td>1038675</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3422</td>
      <td>KBOS</td>
      <td>large_airport</td>
      <td>General Edward Lawrence Logan International Ai...</td>
      <td>42.364300</td>
      <td>-71.005203</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-MA</td>
      <td>Boston</td>
      <td>1</td>
      <td>KBOS</td>
      <td>BOS</td>
      <td>BOS</td>
      <td>http://www.massport.com/logan/</td>
      <td>http://en.wikipedia.org/wiki/Logan_Internation...</td>
      <td>General Edward Lawrence Logan International Ai...</td>
      <td>1035875</td>
      <td>2010-01-23T11:27:48+00:00</td>
    </tr>
    <tr>
      <th>21</th>
      <td>3771</td>
      <td>KPHL</td>
      <td>large_airport</td>
      <td>Philadelphia International Airport</td>
      <td>39.871899</td>
      <td>-75.241096</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-PA</td>
      <td>Philadelphia</td>
      <td>1</td>
      <td>KPHL</td>
      <td>PHL</td>
      <td>PHL</td>
      <td>http://www.phl.org/</td>
      <td>http://en.wikipedia.org/wiki/Philadelphia_Inte...</td>
      <td>NaN</td>
      <td>1031075</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1990</td>
      <td>CYYZ</td>
      <td>large_airport</td>
      <td>Lester B. Pearson International Airport</td>
      <td>43.677200</td>
      <td>-79.630600</td>
      <td>569.0</td>
      <td>NaN</td>
      <td>CA</td>
      <td>CA-ON</td>
      <td>Toronto</td>
      <td>1</td>
      <td>CYYZ</td>
      <td>YYZ</td>
      <td>YYZ</td>
      <td>http://www.gtaa.com/</td>
      <td>http://en.wikipedia.org/wiki/Toronto_Pearson_I...</td>
      <td>YTO, Toronto International Airport, Malton</td>
      <td>1040275</td>
      <td>2009-10-15T13:44:39+00:00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3483</td>
      <td>KDCA</td>
      <td>large_airport</td>
      <td>Ronald Reagan Washington National Airport</td>
      <td>38.852100</td>
      <td>-77.037697</td>
      <td>15.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-DC</td>
      <td>Washington</td>
      <td>1</td>
      <td>KDCA</td>
      <td>DCA</td>
      <td>DCA</td>
      <td>http://www.flyreagan.com/dca/reagan-national-a...</td>
      <td>http://en.wikipedia.org/wiki/Ronald_Reagan_Was...</td>
      <td>WAS</td>
      <td>1016175</td>
      <td>2016-12-13T09:13:00+00:00</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3604</td>
      <td>KIAH</td>
      <td>large_airport</td>
      <td>George Bush Intercontinental Houston Airport</td>
      <td>29.984400</td>
      <td>-95.341400</td>
      <td>97.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-TX</td>
      <td>Houston</td>
      <td>1</td>
      <td>KIAH</td>
      <td>IAH</td>
      <td>IAH</td>
      <td>http://www.fly2houston.com/iah</td>
      <td>http://en.wikipedia.org/wiki/George_Bush_Inter...</td>
      <td>QHO</td>
      <td>1055675</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3457</td>
      <td>KCLT</td>
      <td>large_airport</td>
      <td>Charlotte Douglas International Airport</td>
      <td>35.214001</td>
      <td>-80.943100</td>
      <td>748.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-NC</td>
      <td>Charlotte</td>
      <td>1</td>
      <td>KCLT</td>
      <td>CLT</td>
      <td>CLT</td>
      <td>http://www.charlotteairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Charlotte/Douglas...</td>
      <td>NaN</td>
      <td>1028075</td>
      <td>2009-01-30T14:52:52+00:00</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3497</td>
      <td>KDTW</td>
      <td>large_airport</td>
      <td>Detroit Metropolitan Wayne County Airport</td>
      <td>42.212399</td>
      <td>-83.353401</td>
      <td>645.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-MI</td>
      <td>Detroit</td>
      <td>1</td>
      <td>KDTW</td>
      <td>DTW</td>
      <td>DTW</td>
      <td>http://www.metroairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Detroit_Metropoli...</td>
      <td>DTT, Detroit Metro Airport</td>
      <td>1079375</td>
      <td>2008-06-13T14:30:04+00:00</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2218</td>
      <td>EDDM</td>
      <td>large_airport</td>
      <td>Munich International Airport</td>
      <td>48.353802</td>
      <td>11.786100</td>
      <td>1487.0</td>
      <td>EU</td>
      <td>DE</td>
      <td>DE-BY</td>
      <td>Munich</td>
      <td>1</td>
      <td>EDDM</td>
      <td>MUC</td>
      <td>NaN</td>
      <td>http://www.munich-airport.com/</td>
      <td>http://en.wikipedia.org/wiki/Munich_Internatio...</td>
      <td>Franz Josef Strauss International Airport, Flu...</td>
      <td>1026675</td>
      <td>2015-10-17T03:56:24+00:00</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3709</td>
      <td>KMSP</td>
      <td>large_airport</td>
      <td>Minneapolis-St Paul International/Wold-Chamber...</td>
      <td>44.882000</td>
      <td>-93.221802</td>
      <td>841.0</td>
      <td>NaN</td>
      <td>US</td>
      <td>US-MN</td>
      <td>Minneapolis</td>
      <td>1</td>
      <td>KMSP</td>
      <td>MSP</td>
      <td>MSP</td>
      <td>http://www.mspairport.com/</td>
      <td>http://en.wikipedia.org/wiki/Minneapolis-Saint...</td>
      <td>NaN</td>
      <td>1085275</td>
      <td>2009-12-17T08:53:11+00:00</td>
    </tr>
    <tr>
      <th>29</th>
      <td>4372</td>
      <td>LIRF</td>
      <td>large_airport</td>
      <td>Leonardo da Vinci–Fiumicino Airport</td>
      <td>41.800278</td>
      <td>12.238889</td>
      <td>13.0</td>
      <td>EU</td>
      <td>IT</td>
      <td>IT-62</td>
      <td>Rome</td>
      <td>1</td>
      <td>LIRF</td>
      <td>FCO</td>
      <td>RM11</td>
      <td>http://www.adr.it/portal/portal/adr/Fiumicino/...</td>
      <td>http://en.wikipedia.org/wiki/Leonardo_da_Vinci...</td>
      <td>ROM, Rome Fiumicino Airport, Fiumicino Airport</td>
      <td>1031875</td>
      <td>2016-10-08T00:37:02+00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>50599</th>
      <td>44389</td>
      <td>KZ-0071</td>
      <td>closed</td>
      <td>Karas Air Base</td>
      <td>48.737000</td>
      <td>58.042000</td>
      <td>NaN</td>
      <td>AS</td>
      <td>KZ</td>
      <td>KZ-AKT</td>
      <td>Zhem</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Emba Air Base, Аэродром Карась,Аэродром Эмба</td>
      <td>0</td>
      <td>2009-04-24T13:54:46+00:00</td>
    </tr>
    <tr>
      <th>50600</th>
      <td>44390</td>
      <td>KZ-0072</td>
      <td>closed</td>
      <td>Karas Air Base</td>
      <td>48.737000</td>
      <td>58.042000</td>
      <td>NaN</td>
      <td>AS</td>
      <td>KZ</td>
      <td>KZ-AKT</td>
      <td>Zhem</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Emba Air Base, Аэродром Карась, Аэродром Эмба</td>
      <td>0</td>
      <td>2009-04-24T13:54:54+00:00</td>
    </tr>
    <tr>
      <th>50601</th>
      <td>319812</td>
      <td>HSJ</td>
      <td>small_airport</td>
      <td>Shangjie Airport</td>
      <td>34.842153</td>
      <td>113.273983</td>
      <td>450.0</td>
      <td>AS</td>
      <td>CN</td>
      <td>CN-41</td>
      <td>Zhengzhou</td>
      <td>0</td>
      <td>NaN</td>
      <td>HSJ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-08-06T20:22:46+00:00</td>
    </tr>
    <tr>
      <th>50602</th>
      <td>43684</td>
      <td>UWKV</td>
      <td>medium_airport</td>
      <td>Zhigansk Airport</td>
      <td>66.796501</td>
      <td>123.361000</td>
      <td>292.0</td>
      <td>AS</td>
      <td>RU</td>
      <td>RU-SA</td>
      <td>Zhigansk</td>
      <td>1</td>
      <td>UEVV</td>
      <td>ZIX</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://en.wikipedia.org/wiki/Zhigansk_Airport</td>
      <td>УЕЖЖ, Аэропорт Жиганск</td>
      <td>0</td>
      <td>2015-09-20T23:58:06+00:00</td>
    </tr>
    <tr>
      <th>50603</th>
      <td>308462</td>
      <td>ZLZW</td>
      <td>medium_airport</td>
      <td>Zhongwei Shapotou Airport</td>
      <td>37.573125</td>
      <td>105.154454</td>
      <td>8202.0</td>
      <td>AS</td>
      <td>CN</td>
      <td>CN-64</td>
      <td>Zhongwei</td>
      <td>1</td>
      <td>ZLZW</td>
      <td>ZHY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Zhongwei_Shapotou...</td>
      <td>NaN</td>
      <td>0</td>
      <td>2013-04-10T00:28:33+00:00</td>
    </tr>
    <tr>
      <th>50604</th>
      <td>44386</td>
      <td>KZ-0070</td>
      <td>small_airport</td>
      <td>Zhosaly Airport</td>
      <td>45.514999</td>
      <td>64.084999</td>
      <td>335.0</td>
      <td>AS</td>
      <td>KZ</td>
      <td>KZ-KZY</td>
      <td>Zhosaly</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Dzhusaly Airport, Аэропорт Жосалы, Аэропорт Дж...</td>
      <td>0</td>
      <td>2009-04-24T13:34:01+00:00</td>
    </tr>
    <tr>
      <th>50605</th>
      <td>46369</td>
      <td>UA-0071</td>
      <td>closed</td>
      <td>Zhovti Vody Airstrip</td>
      <td>48.350503</td>
      <td>33.444271</td>
      <td>NaN</td>
      <td>EU</td>
      <td>UA</td>
      <td>UA-12</td>
      <td>Zhovti Vody</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Zholtyye Vody Airstrip, ЗПС Жовті Води, ВПП Жё...</td>
      <td>0</td>
      <td>2010-03-27T18:04:32+00:00</td>
    </tr>
    <tr>
      <th>50606</th>
      <td>46370</td>
      <td>UA-0072</td>
      <td>closed</td>
      <td>Volodimirivka Airstrip</td>
      <td>48.284035</td>
      <td>33.448949</td>
      <td>NaN</td>
      <td>EU</td>
      <td>UA</td>
      <td>UA-12</td>
      <td>Zhovti Vody</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ЗПС Володимирівка, ВПП Владимировка</td>
      <td>0</td>
      <td>2010-03-27T18:11:37+00:00</td>
    </tr>
    <tr>
      <th>50607</th>
      <td>44432</td>
      <td>UKKO</td>
      <td>medium_airport</td>
      <td>Ozerne Air Base</td>
      <td>50.158298</td>
      <td>28.738300</td>
      <td>761.0</td>
      <td>EU</td>
      <td>UA</td>
      <td>UA-18</td>
      <td>Zhytomyr</td>
      <td>0</td>
      <td>UKKO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://en.wikipedia.org/wiki/Ozerne_(air_base)</td>
      <td>Zhitomyr Ozernoye, Oziernoye, Ozernoe, Ozernoy...</td>
      <td>0</td>
      <td>2017-02-18T21:11:10+00:00</td>
    </tr>
    <tr>
      <th>50608</th>
      <td>308157</td>
      <td>DE-0040</td>
      <td>small_airport</td>
      <td>Zierenberg Airfield</td>
      <td>51.367000</td>
      <td>9.336000</td>
      <td>1450.0</td>
      <td>EU</td>
      <td>DE</td>
      <td>DE-HE</td>
      <td>Zierenberg</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2012-12-15T23:37:53+00:00</td>
    </tr>
    <tr>
      <th>50609</th>
      <td>44743</td>
      <td>RU-0197</td>
      <td>small_airport</td>
      <td>Zimovniki Air Base</td>
      <td>47.141998</td>
      <td>42.395000</td>
      <td>NaN</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-ROS</td>
      <td>Zimovniki</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Аэродром Зимовники</td>
      <td>0</td>
      <td>2009-08-17T10:56:36+00:00</td>
    </tr>
    <tr>
      <th>50610</th>
      <td>319653</td>
      <td>MX-0257</td>
      <td>small_airport</td>
      <td>El Pinocho Airport</td>
      <td>19.476550</td>
      <td>-100.433200</td>
      <td>6265.0</td>
      <td>NaN</td>
      <td>MX</td>
      <td>MX-MIC</td>
      <td>Zitácuaro</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PIO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-07-26T00:52:39+00:00</td>
    </tr>
    <tr>
      <th>50611</th>
      <td>43961</td>
      <td>CZ-0044</td>
      <td>heliport</td>
      <td>Bat’A Hospital Heliport</td>
      <td>49.226147</td>
      <td>17.705345</td>
      <td>738.0</td>
      <td>EU</td>
      <td>CZ</td>
      <td>CZ-ZL</td>
      <td>Zlín</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2009-03-27T18:08:04+00:00</td>
    </tr>
    <tr>
      <th>50612</th>
      <td>46129</td>
      <td>RU-0304</td>
      <td>medium_airport</td>
      <td>Kapustin Yar Airfield</td>
      <td>48.667000</td>
      <td>45.732000</td>
      <td>NaN</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-AST</td>
      <td>Znamensk</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Аэродром Капустин Яр</td>
      <td>0</td>
      <td>2010-01-09T12:30:12+00:00</td>
    </tr>
    <tr>
      <th>50613</th>
      <td>319421</td>
      <td>IT-0275</td>
      <td>small_airport</td>
      <td>Campo di Volo Arcobaleno</td>
      <td>44.496944</td>
      <td>11.243611</td>
      <td>NaN</td>
      <td>EU</td>
      <td>IT</td>
      <td>IT-45</td>
      <td>Zola Predosa (BO)</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>BO11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-07-10T05:35:40+00:00</td>
    </tr>
    <tr>
      <th>50614</th>
      <td>45182</td>
      <td>UA-0053</td>
      <td>closed</td>
      <td>Palmira Airfield</td>
      <td>49.763000</td>
      <td>32.123000</td>
      <td>NaN</td>
      <td>EU</td>
      <td>UA</td>
      <td>UA-71</td>
      <td>Zolotonosha</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Аэродром Пальмира</td>
      <td>0</td>
      <td>2009-11-18T14:53:00+00:00</td>
    </tr>
    <tr>
      <th>50615</th>
      <td>319121</td>
      <td>MX-0184</td>
      <td>heliport</td>
      <td>Jupiter I Platform Helipad</td>
      <td>19.520472</td>
      <td>-92.184506</td>
      <td>50.0</td>
      <td>NaN</td>
      <td>MX</td>
      <td>MX-CAM</td>
      <td>Zonda De Campeche</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>HJU</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-06-25T00:38:37+00:00</td>
    </tr>
    <tr>
      <th>50616</th>
      <td>43747</td>
      <td>EBZO</td>
      <td>heliport</td>
      <td>Zandvoorde heliport</td>
      <td>50.817280</td>
      <td>2.974564</td>
      <td>102.0</td>
      <td>EU</td>
      <td>BE</td>
      <td>BE-VWV</td>
      <td>Zonnebeke</td>
      <td>0</td>
      <td>EBZO</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2012-10-21T18:35:45+00:00</td>
    </tr>
    <tr>
      <th>50618</th>
      <td>43230</td>
      <td>GE-0002</td>
      <td>closed</td>
      <td>Zugdidi Airport</td>
      <td>42.489201</td>
      <td>41.818699</td>
      <td>NaN</td>
      <td>AS</td>
      <td>GE</td>
      <td>GE-SZ</td>
      <td>Zugdidi</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2009-08-17T13:52:57+00:00</td>
    </tr>
    <tr>
      <th>50620</th>
      <td>298761</td>
      <td>IT-0032</td>
      <td>small_airport</td>
      <td>Campo di Volo Il Grifo</td>
      <td>38.637255</td>
      <td>15.947750</td>
      <td>630.0</td>
      <td>EU</td>
      <td>IT</td>
      <td>IT-78</td>
      <td>Zungri</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>VVZUN</td>
      <td>http://www.aeroclubilgrifo.it</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2017-01-03T03:03:17+00:00</td>
    </tr>
    <tr>
      <th>50621</th>
      <td>308403</td>
      <td>ZUZY</td>
      <td>medium_airport</td>
      <td>Zunyi Xinzhou Airport</td>
      <td>27.589500</td>
      <td>107.000700</td>
      <td>2920.0</td>
      <td>AS</td>
      <td>CN</td>
      <td>CN-52</td>
      <td>Zunyi</td>
      <td>1</td>
      <td>ZUZY</td>
      <td>ZYI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://en.wikipedia.org/wiki/Zunyi_Xinzhou_Air...</td>
      <td>NaN</td>
      <td>0</td>
      <td>2013-03-30T12:14:29+00:00</td>
    </tr>
    <tr>
      <th>50622</th>
      <td>316165</td>
      <td>LDZE</td>
      <td>small_airport</td>
      <td>Zvekovac</td>
      <td>45.823056</td>
      <td>16.500000</td>
      <td>374.0</td>
      <td>EU</td>
      <td>HR</td>
      <td>HR-01</td>
      <td>Zvekovac</td>
      <td>0</td>
      <td>LDZE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://www.akdubrava.net</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2015-11-02T23:33:02+00:00</td>
    </tr>
    <tr>
      <th>50623</th>
      <td>44702</td>
      <td>RU-0193</td>
      <td>heliport</td>
      <td>Pushkino Heliport</td>
      <td>56.042000</td>
      <td>37.978001</td>
      <td>NaN</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-MOS</td>
      <td>Zverosovkhoz</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>http://www.agusta.ru/</td>
      <td>NaN</td>
      <td>Вертодром Пушкино</td>
      <td>0</td>
      <td>2010-02-25T11:06:38+00:00</td>
    </tr>
    <tr>
      <th>50624</th>
      <td>43804</td>
      <td>LOKZ</td>
      <td>heliport</td>
      <td>Zwatzhof Heliport</td>
      <td>46.975555</td>
      <td>14.274722</td>
      <td>2500.0</td>
      <td>EU</td>
      <td>AT</td>
      <td>AT-2</td>
      <td>Zwatzhof</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2009-03-23T20:54:13+00:00</td>
    </tr>
    <tr>
      <th>50625</th>
      <td>43816</td>
      <td>LOAZ</td>
      <td>heliport</td>
      <td>Zwettl Heliport</td>
      <td>48.599998</td>
      <td>15.164167</td>
      <td>1900.0</td>
      <td>EU</td>
      <td>AT</td>
      <td>AT-3</td>
      <td>Zwettl</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2009-03-23T20:54:37+00:00</td>
    </tr>
    <tr>
      <th>50626</th>
      <td>43708</td>
      <td>NL-0013</td>
      <td>heliport</td>
      <td>Isala Klinieken. Locatie Sophia Heliport</td>
      <td>52.513874</td>
      <td>6.128000</td>
      <td>6.0</td>
      <td>EU</td>
      <td>NL</td>
      <td>NL-OV</td>
      <td>Zwolle</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>EH0029</td>
      <td>0</td>
      <td>2009-03-21T16:38:26+00:00</td>
    </tr>
    <tr>
      <th>50627</th>
      <td>315523</td>
      <td>RU-0438</td>
      <td>small_airport</td>
      <td>Zyryanka West Airport</td>
      <td>65.736700</td>
      <td>150.705000</td>
      <td>150.0</td>
      <td>EU</td>
      <td>RU</td>
      <td>RU-SA</td>
      <td>Zyryanka</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://en.wikipedia.org/wiki/Zyryanka_West_Ai...</td>
      <td>ZCQ6, Зырянка (Западный)</td>
      <td>0</td>
      <td>2015-09-23T00:19:01+00:00</td>
    </tr>
    <tr>
      <th>50628</th>
      <td>318187</td>
      <td>KZ-0081</td>
      <td>small_airport</td>
      <td>Zubovsk Airport</td>
      <td>49.800103</td>
      <td>84.214661</td>
      <td>NaN</td>
      <td>AS</td>
      <td>KZ</td>
      <td>KZ-VOS</td>
      <td>Zyryanovsk</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2016-05-31T03:11:45+00:00</td>
    </tr>
    <tr>
      <th>50629</th>
      <td>42246</td>
      <td>BITH</td>
      <td>closed</td>
      <td>Sauðanes Airport</td>
      <td>66.250603</td>
      <td>-15.267200</td>
      <td>15.0</td>
      <td>EU</td>
      <td>IS</td>
      <td>IS-6</td>
      <td>Þórshöfn</td>
      <td>0</td>
      <td>BITH</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>https://en.wikipedia.org/wiki/%C3%9E%C3%B3rsh%...</td>
      <td>THO</td>
      <td>0</td>
      <td>2015-08-20T20:43:01+00:00</td>
    </tr>
    <tr>
      <th>50630</th>
      <td>315917</td>
      <td>BG-0027</td>
      <td>small_airport</td>
      <td>Gabrovo</td>
      <td>42.948200</td>
      <td>25.359980</td>
      <td>NaN</td>
      <td>EU</td>
      <td>BG</td>
      <td>BG-07</td>
      <td>Габрово</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2015-10-20T12:54:39+00:00</td>
    </tr>
  </tbody>
</table>
<p>39368 rows × 20 columns</p>
</div>
</div>


</div>
</div>
</div>

- Subsetting (querying but return dataframe not just indices)

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
gb = df.groupby('type')
```
</div>

</div>

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
gb.all()
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">



<div markdown="0" class="output output_html">
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>continent</th>
      <th>elevation_ft</th>
      <th>gps_code</th>
      <th>home_link</th>
      <th>iata_code</th>
      <th>id</th>
      <th>ident</th>
      <th>iso_country</th>
      <th>iso_region</th>
      <th>keywords</th>
      <th>last_updated</th>
      <th>latitude_deg</th>
      <th>local_code</th>
      <th>longitude_deg</th>
      <th>municipality</th>
      <th>name</th>
      <th>scheduled_service</th>
      <th>score</th>
      <th>wikipedia_link</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>balloonport</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>closed</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>heliport</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>large_airport</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>medium_airport</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>seaplane_base</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>small_airport</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
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
small = df[df.type=='small_airport']
medium = df[df.type=='medium_airport']
large = df[df.type=='large_airport']
```
</div>

</div>

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
len(small)
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
32875
```


</div>
</div>
</div>

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
len(medium)
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
4530
```


</div>
</div>
</div>

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
len(large)
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
574
```


</div>
</div>
</div>

- spatial join - airports by countries

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
countries_shp = ps.pdio.read_files("data/airports/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp")
```
</div>

</div>

- derived features - point sequence to line for the routes
- spatial join - does route pass through a country
- crs: contextily example, 
- knn analysis - find most isolated airport
- voronoi - whats my closest airport
- dissolve - dissovle boundaries in europe

## Vignette: House Prices

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
df = pd.read_csv('data/sandiego/listings.csv')
len(df)
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
6608
```


</div>
</div>
</div>

<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
df.columns
```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">


{:.output_data_text}
```
Index(['id', 'listing_url', 'scrape_id', 'last_scraped', 'name', 'summary',
       'space', 'description', 'experiences_offered', 'neighborhood_overview',
       'notes', 'transit', 'access', 'interaction', 'house_rules',
       'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url',
       'host_id', 'host_url', 'host_name', 'host_since', 'host_location',
       'host_about', 'host_response_time', 'host_response_rate',
       'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url',
       'host_picture_url', 'host_neighbourhood', 'host_listings_count',
       'host_total_listings_count', 'host_verifications',
       'host_has_profile_pic', 'host_identity_verified', 'street',
       'neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
       'smart_location', 'country_code', 'country', 'latitude', 'longitude',
       'is_location_exact', 'property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
       'price', 'weekly_price', 'monthly_price', 'security_deposit',
       'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
       'maximum_nights', 'calendar_updated', 'has_availability',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365', 'calendar_last_scraped', 'number_of_reviews',
       'first_review', 'last_review', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 'requires_license',
       'license', 'jurisdiction_names', 'instant_bookable',
       'cancellation_policy', 'require_guest_profile_picture',
       'require_guest_phone_verification', 'calculated_host_listings_count',
       'reviews_per_month'],
      dtype='object')
```


</div>
</div>
</div>

- keyword table join (census)
(keyword comes from spatial join with polygon shown below)

- groupby: avg house price by census polygon
- buffer: deriving dummies for houses within x of an amenity
- spatial join: create keyword that we use for the table
- raster/clip with shape: elevation or pollution by tract, or by house, or  noise
- voronoi - what's my closest coffee shop

- Sets: union, intersection, difference: point out that these are really implied by the buffer used to define regimes (intersection dummy = 1, difference dummy=0)

message is, if you have the column in the table use it, but many cases you do not have the column and need to go the spatial join route
