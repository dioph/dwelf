# dwelf
Stellar parameter determination based on Spot Modeling
## Pre-requisites
Because `dwelf` uses `MultiNest`, make sure you have installed 
_blas_, _lapack_ and _atlas_:

    $ sudo apt-get install libblas3 liblapack3 libatlas3-base

Then you should get and compile `MultiNest`:
    
    $ git clone https://github.com/JohannesBuchner/MultiNest
    $ cd MultiNest/build
    $ cmake ..
    $ make

And make sure your environment has `LD_LIBRARY_PATH` pointing to `MultiNest/lib`
    
## Installation
In order to get the latest version from source:

    $ git clone https://github.com/dioph/dwelf.git
    $ cd dwelf
    $ python setup.py install

## Example using CheetahModeler

```python
from dwelf import *
from astropy.io import ascii

filename = PACKAGEDIR + '/data/kappaCeti2003.csv'
kappa2003 = ascii.read(filename)

model = CheetahModeler(rmin=0.85, rmax=1.05, inc_min=30, inc_max=80, Peq_min=8.0, Peq_max=10.5, 
                       lat_min=-10, lat_max=60, burn=250, n_walkers=60, n_steps=2500,
                       v_min=4.5, v_max=5.5, threshratio=1.1)
model.x = kappa2003['time']
model.y = kappa2003['flux']
params = model.fit()

print(model.bestps)
for p in params:
    print('{0:6.2f} -- {1:6.2f}'.format(p[0]-p[2], p[0]+p[1]))
v_min = sum(model.vsini(params[0][0]-params[0][2], params[1][0]+params[1][1]))/2
v_max = sum(model.vsini(params[0][0]+params[0][1], params[1][0]-params[1][2]))/2
print('v sin i: {0:.2f} -- {1:.2f}'.format(v_min, v_max))
model.plot_min()
plot_mcmc(model.samples)
```
