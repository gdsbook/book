import glob
import nbformat

token = 'filterwarnings(\\"ignore\\")'

mpl = "\"matplotlib.rcParams['figure.dpi'] = 300"

replace = f"{token}\n{mpl}"


replace 
nbs = glob.glob("latex300/*.ipynb")

for nbf in nbs:
    nb = nbformat.read(nbf, as_version=4)
    cell0 = nb.cells[0]
    if 'source' in cell0:
        old = cell0['source']
        print(old)
        new = old+"\nimport matplotlib\nmatplotlib.rcParams['figure.dpi']=300"
        print(new)
        nb.cells[0]['source']=new
    nbformat.write(nb, nbf)
