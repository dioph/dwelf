from numpy import *
from dwelf_class import Modeler
import cleanest
import kplr

# attempt to model Kepler-17 lightcurve
model = Modeler(rmin=1.02, rmax=1.08, inc_min=30, Teq_min=10.0, Teq_max=12.5, 
				k_min=-0.75, k_max=0.75, v_min=3.7, v_max=4.7, n_steps=2000)
				
kic = kplr.API().star(10619192)
hdu = kic.get_light_curves()[0].open()
tbl = hdu[1].data
model.x = tbl['time']
model.y = tbl['sap_flux']

params = model.fit()
for p in params:
	print('{0:.2f} -- {1:.2f}'.format(p[0]-p[2], p[0]+p[1]))
v_min = sum(model.vsini(params[0][0]-params[0][2], params[1][0]+params[1][1]))/2
v_max = sum(model.vsini(params[0][0]+params[0][1], params[1][0]-params[1][2]))/2
print('v sin i: {0:.2f} -- {1:.2f}'.format(v_min, v_max))
model.plot_min()
model.plot_mcmc()

model.y, bp = cleanest.join(model.x, model.y, 2)

params = model.fit()
print('---CLEANEST---')
for p in params:
	print('{0:.2f} -- {1:.2f}'.format(p[0]-p[2], p[0]+p[1]))
v_min = sum(model.vsini(params[0][0]-params[0][2], params[1][0]+params[1][1]))/2
v_max = sum(model.vsini(params[0][0]+params[0][1], params[1][0]-params[1][2]))/2
print('v sin i: {0:.2f} -- {1:.2f}'.format(v_min, v_max))
model.plot_min()
model.plot_mcmc()
