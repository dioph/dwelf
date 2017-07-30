from numpy import *
from dwelf_class import Modeler
import cleanest

# attempt to model kappa Ceti lightcurve from 2003
model = Modeler(rmin=0.85, rmax=1.05, inc_min=30, inc_max=80, Teq_min=8.0, Teq_max=10.5, 
				k_min=-0.75, k_max=0.75, rad_max=15, v_min=4, v_max=6)
				
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
for p in params:
	print('{0:.2f} -- {1:.2f}'.format(p[0]-p[2], p[0]+p[1]))
model.plot_min()
model.plot_mcmc()

model.y, bp = cleanest.join(model.x, model.y, 2)
params = model.fit()
print('---CLEANEST---')
for p in params:
	print('{0:.2f} -- {1:.2f}'.format(p[0]-p[2], p[0]+p[1]))
model.plot_min()
model.plot_mcmc()
