from numpy import *
from dwelf_class import Modeler
import cleanest
import matplotlib.pyplot as plt

# attempt to model kappa Ceti lightcurve from 2003
model = Modeler(rmin=0.85, rmax=1.05, inc_min=30, inc_max=80, Teq_min=8.0, Teq_max=10.5, lat_min=-10, lat_max=60,
				burn=300, n_walkers=300, n_steps=3000, v_min=4.5, v_max=5.5, threshratio=1.1, n_max=1)
				
text = open('kappa.txt')
model.x = array([])
model.y = array([])
for line in text:
	if line == '\n':
		continue
	line = line.strip().split('\t')
	if float(line[0]) > 1700:
		continue
	model.x = append(model.x, float(line[0]))
	model.y = append(model.y, float(line[1]))
	

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
