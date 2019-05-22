import os

ourpath = os.path.abspath(os.path.dirname(__file__))
datapath = os.path.join(ourpath, '../data')


def san_diego_tracts():
    return os.path.join(datapath, 'sandiego/sd_tracts_acs_clean.shp') 

def san_diego_airbnbs():
    raise NotImplementedError
    return os.path.join(datapath, '...')

def texas():
    return os.path.join(datapath, 'texas.shp')

def mexico():
    return os.path.join(datapath, 'mexicojoin.shp')

def brexit():
    return os.path.join(datapath, 'brexit_vote.csv')

def lads():
    f = ('Local_Authority_Districts_December_2016_Generalised_Clipped_Boundaries_in_the_UK_WGS84/'\
         'Local_Authority_Districts_December_2016_Generalised_Clipped_Boundaries_in_the_UK_WGS84.shp')
    return os.path.join(datapath, f)

def san_diego_neighborhoods():
    return os.path.join(datapath, 'airbnb/neighbourhoods.geojson')

def regression_airbnbs():
    return os.path.join(datapath, 'airbnb/regression_db.geojson') 

def texas():
    return os.path.join(datapath, 'texas.shp')

def mexico():
    return os.path.join(datapath, 'mexicojoin.shp')