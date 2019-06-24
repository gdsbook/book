from pysal.viz import mapclassify


# facade for geopandas and pysal2.0rc

schemes = ['Quantiles', 'Equal_Interval', 'Maximum_Breaks', 'Fisher_Jenks']
dispatcher = {}
for scheme in schemes:
    dispatcher[scheme.lower()] = eval('mapclassify.classifiers.{scheme}'.format(scheme=scheme))

def choropleth(df, column, scheme='Quantiles', k=5, cmap='BluGrn', legend=False,  \
               edgecolor='white', linewidth=0.1, alpha=0.75, ax=None):
    """
    Choropleth mapping based on geopandas and pysal2.0rc
    
    Parameters
    ----------
    
    df: geopandas GeoDataFrame
    
    column: string
            column name for attribute that is to be mapped
            
    scheme: string
            Name of mapclassify classification scheme
            
    k:  int
        number of classes for choropleth
        
    cmap: string
          name of colormap from matplotlib
          
    legend: Boolean
            Show legend (True)
            
    edgecolor: string
             Color of polygon edges
    
    linewidth: float
            width of edges
            
    alpha: float
           transparency
            
    ax: matplotlib.pyplot plt axis
    
    
    """
    classified = dispatcher[scheme.lower()](df[column], k=k)
    legend = [ '%.3f'%cut  for cut in classified.bins]
    labels = [legend[ybi] for ybi in classified.yb]
    ax = df.assign(cl=labels).plot(column='cl', categorical=True, \
                                   cmap=cmap, legend=legend, 
                                   edgecolor=edgecolor, linewidth=linewidth, \
                                   alpha=alpha, ax=ax)
    return ax
    
