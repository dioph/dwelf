from numpy import *
from dwelf_class import Modeler
import cleanest
import matplotlib.pyplot as plt
import kplr

# attempt to model Kepler-17 lightcurve from quarter 1
model = Modeler(rmin=1.02, rmax=1.08, inc_min=30, inc_max=80, Teq_min=10.0, Teq_max=12.5,
				n_spots=2, v_min=3.7, v_max=4.7, threshratio=1.1, savefile="savetest.txt")
	
kic = kplr.API().star(10619192)
hdu = kic.get_light_curves()[0].open()
tbl = hdu[1].data
model.x = tbl['time']
model.y = tbl['sap_flux']

params = model.fit()
print(model.bestps)
for p in params:
	print('{0:6.2f} -- {1:6.2f}'.format(p[0]-p[2], p[0]+p[1]))
v_min = sum(model.vsini(params[0][0]-params[0][2], params[1][0]+params[1][1]))/2
v_max = sum(model.vsini(params[0][0]+params[0][1], params[1][0]-params[1][2]))/2
print('v sin i: {0:.2f} -- {1:.2f}'.format(v_min, v_max))
model.plot_min()
plt.show()
model.plot_mcmc()

model.y, bp = cleanest.join(model.x, model.y, 2)

model.savefile = None

params = model.fit()
print(model.bestps)
print('---CLEANEST---')
for p in params:
	print('{0:6.2f} -- {1:6.2f}'.format(p[0]-p[2], p[0]+p[1]))
v_min = sum(model.vsini(params[0][0]-params[0][2], params[1][0]+params[1][1]))/2
v_max = sum(model.vsini(params[0][0]+params[0][1], params[1][0]-params[1][2]))/2
print('v sin i: {0:.2f} -- {1:.2f}'.format(v_min, v_max))
model.plot_min()
plt.show()
model.plot_mcmc()
