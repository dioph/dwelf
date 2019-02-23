# dwelf
Stellar parameter determination based on Spot Modeling
## Pre-requisites
Because `dwelf` uses `MultiNest`, make sure you have installed 
_blas_, _lapack_ and _atlas_:

    $ sudo apt-get install libblas{3,-dev} liblapack{3,-dev} libatlas3-base

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
t, y, dy = kappa2003['time'], kappa2003['flux'], kappa2003['alternate_err']

model = CheetahModeler(t, y, dy=dy, inc_min=30, inc_max=80, Peq_min=8.0, Peq_max=10.5,
                       lat_min=-10, lat_max=60, burn=250, n_walkers=60, n_steps=2500,
                       v_min=4.5, v_max=5.5, rmin=0.85, rmax=1.05, threshratio=1.1)
params = model.fit()
# print best fit solutions
print(model.bestps)
# print probable intervals (16th -- 84th percentiles)
for p in params:
    print('{0:06.02f} -- {1:06.02f}'.format(p[0]-p[2], p[0]+p[1]))
# print v sin i probable interval
v_min = sum(model.vsini(params[0][0]-params[0][2], params[1][0]+params[1][1]))/2
v_max = sum(model.vsini(params[0][0]+params[0][1], params[1][0]-params[1][2]))/2
print('v sin i: {0:.02f} -- {1:.02f}'.format(v_min, v_max))
# plot best fits
model.plot_min()
# plot MCMC triangle confusogram
plot_mcmc(model.samples)
```
## Example using MaculaModeler

```python
from dwelf import *
from astropy.io import ascii
import numpy as np

filename = PACKAGEDIR + '/data/kappaCeti.csv'
kappa = ascii.read(filename)
t, y, dy = kappa['time'], kappa['flux'], kappa['alternate_err']

d2r = np.pi/180
nspots = 7

inc = np.array([30, 80]) * d2r
Peq = np.array([8.0, 10.5])
k2 = np.array([-.75, .75])
k4 = 0.
c = np.array([0., .684, 0., 0.])
d = np.array([0., .684, 0., 0.])

alpha = np.array([[0, 30] for _ in range(nspots)]) * d2r
fspot = np.array([.22 for _ in range(nspots)])
tmax = np.array([t[0], t[0], t[419], t[419], t[419], t[700], t[700]])
life = np.array([100 for _ in range(nspots)])
ingress = np.array([0 for _ in range(nspots)])
egress = np.array([0 for _ in range(nspots)])

U = np.array([[.99, 1.01], [.99, 1.015], [.99, 1.01]])
B = np.array([1, 1, 1])

tstart = np.array([1400, 1500, 2000])
tend = np.array([1500, 2000, 2200])

wsini = np.array([4.5, 5.5]) / (.95 * 695700)

model = MaculaModeler(t, y, nspots, dy=dy, inc=inc, Peq=Peq, k2=k2, k4=k4, c=c, d=d,
                      wsini=wsini, alpha=alpha, fspot=fspot, tmax=tmax, life=life,
                      ingress=ingress, egress=egress, U=U, B=B, tstart=tstart, tend=tend)
                      
res = model.multinest(verbose=True, importance_nested_sampling=False, evidence_tolerance=100)
```
