from numpy import *
from matplotlib import pyplot as plt
from scipy.optimize import leastsq
from scipy.cluster.vq import whiten, kmeans2
from astropy.stats import LombScargle as ls
from time import process_time
import cleanest
import warnings
warnings.filterwarnings('ignore')

global inc_min, inc_max, inc_n
global Teq_min, Teq_max, Teq_n
global alpha_min, alpha_max, alpha_n
global lat_min, lat_max, lat_n
global lon_min, lon_max, lon_n
global rad_min, rad_max, rad_n
#0.45 0.3 0.67
def eker(time, bps, l1=0.7, l2=0.0, ir=0.29):
	d2r = pi / 180
	
	inc_deg = bps[0]
	period = bps[1] / (1 - bps[2] * sin(bps[3] * d2r)**2)
	lat_deg = bps[3]
	lon_deg = bps[4]
	rad_deg = bps[5]
	
	limb1 = l1
	limb2 = l2
	iratio = ir

	inc = inc_deg * d2r
	lam = lon_deg * d2r
	bet = lat_deg * d2r
	rad = rad_deg * d2r
	
	cosrad = cos(rad)
	sinrad = sin(rad)
	
	phase = time / period
	phi = 2.0 * pi * phase
	nphi = len(phi)
	
	costhe0 = cos(inc) * sin(bet) + sin(inc) * cos(bet) * cos(phi-lam)
	sinthe0 = sqrt(1.0 - costhe0**2)
	the0 = arccos(costhe0)
	
	jf = flatnonzero(the0 <= pi/2-rad)
	nf = len(jf)
	jg = flatnonzero(logical_and(the0 > pi/2-rad, the0 <= pi/2))
	ng = len(jg)
	jc = flatnonzero(logical_and(the0 > pi/2, the0 <= pi/2+rad))
	nc = len(jc)
	jo = flatnonzero(the0 > pi/2+rad)
	no = len(jo)
	
	ic = zeros(nphi)
	il = zeros(nphi)
	iq = zeros(nphi)
	
	# FULL
	
	if nf >= 1:
		costhe0_f = costhe0[jf]
		sinthe0_f = sinthe0[jf]
		
		ic[jf] = pi * sin(rad)**2 * costhe0_f
		il[jf] = 2*pi/3 * (1 - cosrad**3) - pi * cosrad * sinrad**2 * sinthe0_f**2
		iq[jf] = pi/2 * (1 - cosrad**4) * costhe0_f**3 + 3*pi/4 * sinrad**4 * costhe0_f * sinthe0_f**2
		
	# GIBBOUS
	
	if ng >= 1:
		the0_g = the0[jg]
		costhe0_g = costhe0[jg]
		sinthe0_g = sinthe0[jg]
		
		cosphi0_g = - 1.0 / ( tan(the0_g) * tan(rad) )
		rad0_g = abs( the0_g - pi/2 )
		
		phi0_g = arccos(cosphi0_g)
		sinphi0_g = sqrt(1.0 - cosphi0_g**2)
		cosrad0_g = cos(rad0_g)
		sinrad0_g = sin(rad0_g)
		
		k1_g = ((pi - phi0_g) / 4) * (cosrad0_g**4 - cosrad**4)
		k2_g = (sinphi0_g / 8) * ( rad0_g - rad + 0.5 * ( sin(2*rad)	* cos(2*rad) - sin(2*rad0_g) * cos(2*rad0_g) ) )
		k3_g = (1.0 / 8) * (pi - phi0_g - sinphi0_g * cosphi0_g) * (sinrad**4 - sinrad0_g**4)
		k4_g = - (sinphi0_g - sinphi0_g**3 / 3) * ( (3.0 / 8) * (rad - rad0_g) + (1.0 / 16) * ( sin(2*rad)	* (cos(2*rad)	- 4) - sin(2*rad0_g) * (cos(2*rad0_g) - 4) ) )

		cl_g = ((pi - phi0_g) / 3) * (cosrad**3 - cosrad0_g**3) * (1 - 3*costhe0_g**2) - (pi - phi0_g - sinphi0_g * cosphi0_g) * (cosrad - cosrad0_g) * sinthe0_g**2 - (4.0 / 3) * sinphi0_g * (sinrad**3 - sinrad0_g**3) * sinthe0_g * costhe0_g - (1.0 / 3) * sinphi0_g * cosphi0_g * (cosrad**3 - cosrad0_g**3) * sinthe0_g**2
		cq_g = 2 * costhe0_g**3 * k1_g + 6 * costhe0_g**2 * sinthe0_g * k2_g + 6 * costhe0_g * sinthe0_g**2 * k3_g + 2 * sinthe0_g**3 * k4_g

		ic[jg] = phi0_g * costhe0_g * sinrad**2 - arcsin(cosrad / sinthe0_g) - 0.5 * sinthe0_g * sinphi0_g * sin(2*rad) + pi/2

		il[jg] = 2*pi/3 * (1 - cosrad**3) - pi * cosrad * sinrad**2 * sinthe0_g**2 - cl_g
		iq[jg] = pi/2 * (1 - cosrad**4) * costhe0_g**3 + 3*pi/4 * sinrad**4 * costhe0_g * sinthe0_g**2 - cq_g
		
	# CRESCENT
	
	if nc >= 1:
		the0_c = the0[jc]
		costhe0_c = costhe0[jc]
		sinthe0_c = sinthe0[jc]
		
		cosphi0_c = - 1.0 / ( tan(the0_c) * tan(rad) )
		rad0_c = abs( the0_c - pi/2 )

		phi0_c = arccos(cosphi0_c)
		sinphi0_c = sqrt(1.0 - cosphi0_c**2)
		cosrad0_c = cos(rad0_c)
		sinrad0_c = sin(rad0_c)

		k1_c = (phi0_c / 4) * (cosrad0_c**4 - cosrad**4)
		k2_c = - (sinphi0_c / 8) * ( rad0_c - rad + 0.5 * ( sin(2*rad)	* cos(2*rad) - sin(2*rad0_c) * cos(2*rad0_c) ) )
		k3_c = (1.0 / 8) * (phi0_c + sinphi0_c * cosphi0_c) * (sinrad**4 - sinrad0_c**4)
		k4_c = (sinphi0_c - sinphi0_c**3 / 3) * ( (3.0 / 8) * (rad - rad0_c) + (1.0 / 16) * ( sin(2*rad)	* (cos(2*rad)	- 4) - sin(2*rad0_c) * (cos(2*rad0_c) - 4) ) )
		
		cq_c = 2 * costhe0_c**3 * k1_c + 6 * costhe0_c**2 * sinthe0_c * k2_c + 6 * costhe0_c * sinthe0_c**2 * k3_c + 2 * sinthe0_c**3 * k4_c

		ic[jc] = phi0_c * costhe0_c * sinrad**2 - arcsin(cosrad / sinthe0_c) - 0.5 * sinthe0_c * sinphi0_c * sin(2*rad) + pi/2

		il[jc] = (phi0_c / 3) * (cosrad**3 - cosrad0_c**3) * (1 - 3 * costhe0_c**2) - (phi0_c + sinphi0_c * cosphi0_c) * (cosrad - cosrad0_c) * sinthe0_c**2 + (4.0 / 3) * sinphi0_c * (sinrad**3 - sinrad0_c**3) * sinthe0_c * costhe0_c + (1.0 / 3) * sinphi0_c * cosphi0_c * (cosrad**3 - cosrad0_c**3) * sinthe0_c**2

		iq[jc] = cq_c
		
	# OCCULTED
	
	if no >=1:
		ic[jo] = 0.0
		il[jo] = 0.0
		iq[jo] = 0.0
	
	lc = 1.0 + (iratio - 1.0) / (pi * (1.0 - limb1/3.0 + limb2/6.0)) * ((1.0 - limb1 + limb2)*ic + (limb1 - 2.0 * limb2)*il + limb2*iq)
	return lc

def spacedvals(lower, upper, nvals):
	spacing = (upper - lower)/double(nvals)
	return arange(lower+(spacing/2.0), upper, spacing)

def model(x, params, l1=0.7, l2=0.0, ir=0.29):
	star_params = params[0:3]
	lc = eker(x, append(star_params, params[3:6]), l1, l2, ir)
	for j in range(6, len(params), 3):
		lc += eker(x, append(star_params, params[j:j+3]), l1, l2, ir) - 1
	return lc

def llsq(x, y, p0s, n_iter=0, star_params=[]):
	opts = []
	sses = []
	p0s2 = []
	for p0 in p0s:
		try:
			p1 = p0[len(star_params):]
			fps, iem = leastsq(eps, p1, args=(x, y, star_params), maxfev=n_iter)
			if any(isnan(fps)) == False:
				opts.append(append(star_params, fps))
				sses.append(sse(fps, x, y, star_params))
				p0s2.append(p0)
		except:
			pass
			
	return opts, sses, p0s2

def vsini(i, T, rmin=0.85, rmax=1.05):
	return [2*pi*rmin*695700*sin(i*pi/180)/(T*3600*24), 2*pi*rmax*695700*sin(i*pi/180)/(T*3600*24)]

def eps(params, x, y, star_params=[], vmin=4, vmax=6):
	global inc_min, inc_max, inc_n
	global Teq_min, Teq_max, Teq_n
	global alpha_min, alpha_max, alpha_n
	global lat_min, lat_max, lat_n
	global lon_min, lon_max, lon_n
	global rad_min, rad_max, rad_n
	
	params = append(star_params, params)
	comp_min = [inc_min, Teq_min, alpha_min, lat_min, lon_min, rad_min]
	comp_max = [inc_max, Teq_max, alpha_max, lat_max, lon_max, rad_max]
	
	diff = y - model(x, params)
	
	v = vsini(params[0], params[1])
	if v[1] < vmin or v[0] > vmax:
		return 10
	
	for j in range(len(params)):
		if j >= 6:
			if (params[j] < comp_min[j-3] or params[j] > comp_max[j-3]):
				return 10
		else:
			if (params[j] < comp_min[j] or params[j] > comp_max[j]):
				return 10
	
	if sum(isnan(diff)) != 0:
		return 10
		
	return diff

def sse(params, x, y, star_params=[]):
	return sum(eps(params, x, y, star_params)**2)

def singlefit(x, y, p0s, star_params=[], n_iter=12, n_clusters=18):
	opts, sses, p0s = llsq(x, y, p0s, n_iter, star_params=star_params)
	stups = sorted(zip(sses, opts, p0s), key=lambda x: x[0])
	sses, opts, p0s = zip(*stups)

	optsmat = whiten(array(opts[:int(0.75*len(opts))]))
	centroid, label = kmeans2(optsmat, n_clusters, iter=20, minit='points')
	label = list(label)
	p0s = [opts[label.index(i)] for i in range(n_clusters) if i in label]

	opts, sses, p0s = llsq(x, y, p0s, star_params=star_params)
	stups = sorted(zip(sses, opts, p0s), key=lambda x: x[0])
	sses, opts, p0s = zip(*stups)
	
	return opts[:min(10, len(opts))]

def multifit(x, y, p0s):
	t1 = process_time()
	opts1 = singlefit(x, y, p0s)
	t2 = process_time()
	print('PRIMEIRO FIT: {0:.3f} s'.format(t2-t1))
	p = []
	for p1 in opts1:
		y_r = y - model(x, p1) + 1
		y_r /= max(y_r)
		opts2 = singlefit(x, y_r, p0s, star_params=p1[0:3])
		for p2 in opts2:
			p.append(append(p1, p2[3:]))
	t3 = process_time()
	print('SEGUNDO FIT: {0:.3f} s'.format(t3-t2))
	opts, sses, p0s = llsq(x, y, p)
	t4 = process_time()
	print('SIMULFIT: {0:.3f} s'.format(t4-t3))
	stups = sorted(zip(sses, opts, p0s), key=lambda x: x[0])
	sses, opts, p0s = zip(*stups)
	print('TOTAL: {0:.3f} s'.format(t4-t1))
	return opts

def sst(x, y):
	m = sum(y)/len(y)
	return sum((y-m)**2)

def check(x, y, p):
	for j in p:
		print('{0:.3f} '.format(j), end='')
	else:
		print('sse={0:.5f}'.format(sse(p,x,y)))
	print('R={0:.5f}'.format(1-sse(p,x,y)/sst(x,y)))

def fit(lc, star_info=[[0,90,3],[1,45,3],[-0.6,0.6,3]], spots_info=[[-90,90,3],[0,360,3],[5,25,3]], clean=0):
	global inc_min, inc_max, inc_n
	global Teq_min, Teq_max, Teq_n
	global alpha_min, alpha_max, alpha_n
	global lat_min, lat_max, lat_n
	global lon_min, lon_max, lon_n
	global rad_min, rad_max, rad_n
	
	x, y = lc
	k = int(len(y)/500)
	if len(y) > 500:
		chosen = [i % k == 0 for i in range(len(y))] #random.permutation(len(y))[:500]
		x = x[chosen]
		y = y[chosen]
		'''stups = sorted(zip(x, y), key=lambda x: x[0])
		x, y = zip(*stups)'''
	p = linspace(0.5, 50.5, 1001)
	f = 1. / p
	'''a = ls(x, y).power(f)
	plt.plot(1/f, a)
	plt.show()'''
	inc, Teq, alpha = star_info
	inc_min, inc_max, inc_n = inc
	Teq_min, Teq_max, Teq_n = Teq
	alpha_min, alpha_max, alpha_n = alpha
	
	n_spots = len(spots_info)
	
	lat, lon, rad = spots_info
	lat_min, lat_max, lat_n = lat
	lon_min, lon_max, lon_n = lon
	rad_min, rad_max, rad_n = rad
	
	inc_vals = spacedvals(inc_min, inc_max, inc_n)
	Teq_vals = spacedvals(Teq_min, Teq_max, Teq_n)
	alpha_vals = spacedvals(alpha_min, alpha_max, alpha_n)
	
	lat_vals = spacedvals(lat_min, lat_max, lat_n)
	lon_vals = spacedvals(lon_min, lon_max, lon_n)
	rad_vals = spacedvals(rad_min, rad_max, rad_n)
	
	p0s = list([a,b,c,d,e,f] for a in inc_vals for b in Teq_vals for c in alpha_vals for d in lat_vals for e in lon_vals for f in rad_vals)
	if clean > 0:
		yc, bp = cleanest.join(x, y, clean)
	else:
		yc = y
		bp = []
	opts = multifit(x, yc, p0s)
	a1 = ls(x, y).power(f)
	a2 = ls(x, yc).power(f)
	
	#ans = [60.6, 8.784, 0.087, 32.4, 298.5, 11.75, 37.2, 104.7, 5.95]
	#ans = [60.6, 8.784, 0.087, 58.4, 162, 9.74, 49.6, 292, 8.55]
	
	check(x, y, opts[0])
	#check(x, y, ans)
	
	plt.subplot(321)
	plt.plot(x, y, 'r.')
	plt.plot(x, model(x, opts[0]), 'b')
	#plt.plot(x, model(x, ans), 'k')
	plt.grid()
	plt.subplot(322)
	plt.ylim([0.0, 1.0])
	plt.plot(1/f, a1, 'r')
	for b in bp:
		plt.plot([b, b],[0, a1[int((b-0.5)/0.05 + 1)]],'b--')
	plt.subplot(323)
	plt.plot(x, yc, 'r.')
	plt.plot(x, model(x, opts[0]), 'b')
	#plt.plot(x, model(x, ans), 'k')
	plt.grid()
	plt.subplot(324)
	plt.ylim([0.0, 1.0])
	plt.plot(1/f, a2, 'r')
	for b in bp:
		plt.plot([b, b],[0, a2[int((b-0.5)/0.05 + 1)]],'b--')
	plt.subplot(325)
	plt.plot(x, 1+y-model(x, opts[0]), 'k')
	plt.grid()
	plt.subplot(326)
	plt.ylim([0.0, 1.0])
	plt.plot(1/f, ls(x, 1+y-model(x, opts[0])).power(f), 'k')
	plt.show()
	
	return opts[0]
