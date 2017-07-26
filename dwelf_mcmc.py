from numpy import *
import scipy.optimize as op
import emcee, corner
import matplotlib.pyplot as plt
from time import process_time
import dwelf
import warnings
warnings.filterwarnings('ignore')

def eker(time, params):
	d2r = pi / 180
	limb1, limb2, iratio, inc_deg, Teq, k, lat_deg, lon_deg, rad_deg = params
	inc = inc_deg * d2r
	lam = lon_deg * d2r
	bet = lat_deg * d2r
	rad = rad_deg * d2r
	period = Teq / (1 - k * sin(bet)**2)
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

'''
params:		bounds:		given:
[0] l1		0,1
[1] l2		0,1
[2] ir		0,1
[3]	inc		0,90		R_sun = 695700
[4]	Teq 	1,45		v = 2 * pi * R * sin(i)/(86400*T)
[5] k		-0.6,0.6	k = (T2-T1) / (T2*sin(lat2)**2 - T1*sin(lat1)**2)
[6]	lat1	-90,90		Teq = T * (1 - k * sin(lat)**2)
[7]	lon1	0,360		
[8]	rad1	5,25
[9]	lat2	-90,90
[10]lon2	0,360
[11]rad2	5,25
'''

def spacedvals(lower, upper, nvals):
	spacing = (upper - lower)/double(nvals)
	return arange(lower+(spacing/2.0), upper, spacing)

def solve(x, params):
	n_spots = int((len(params)-6)/3)
	y = eker(x, params[:9])
	for i in range(1, n_spots):
		y += -1 + eker(x, append(params[:6], params[3*(i+2):3*(i+3)]))
	return y

def lnprior(params):
	n_spots = int((len(params)-6)/3)
	l1,l2,ir,inc,Teq,k = params[:6]
	if not (0 <= l1 < 1 and 0 <= l2 < 1 and 0 < ir < 1 and 0 < inc < 90 and 1 < Teq < 45 and -0.6 < k < 0.6):
		return -inf
	for i in range(n_spots):
		if not (-90 < params[3*i+6] < 90 and 0 < params[3*i+7] < 360 and 5 < params[3*i+8] < 25):
			return -inf
	return 0.0

def chi(params, x, y):
	diff = y - solve(x, params)
	return sum(diff**2)

def lnprob(params, x, y):
	lp = lnprior(params)
	if not isfinite(lp):
		return -inf
	return lp - chi(params, x, y)

def mcmc(x, y, p0s):
	ndim = len(p0s[0])
	nwalkers = len(p0s)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y))
	print("Running MCMC...")
	t1 = process_time()
	sampler.run_mcmc(p0s, 500)
	print("Done.")
	t2 = process_time()
	print('Took {0:.3f} seconds'.format(t2-t1))
	return sampler.chain

def fit(x, y):
	n_walkers = 100
	'''p0s = list( [l1,l2,ir,inc,Teq,k,lat1,lon1,rad1,lat2,lon2,rad2] \
	  for l1 in spacedvals(0,1,1) for l2 in spacedvals(0,1,1) for ir in spacedvals(0,1,1) \
	  for inc in spacedvals(0,90,1) for Teq in spacedvals(1,45,2) for k in spacedvals(-0.6,0.6,1) \
	  for lat1 in spacedvals(-90,90,2) for lon1 in spacedvals(0,360,1) for rad1 in spacedvals(5,25,2) \
	  for lat2 in spacedvals(-90,90,2) for lon2 in spacedvals(0,360,1) for rad2 in spacedvals(5,25,2) )'''
	result = append([0.7, 0.0, 0.29], dwelf.fit([x,y], clean=2))
	p0s = [result + (i/(n_walkers*5)) * random.randn(12) for i in range(n_walkers)]
	chain = mcmc(x, y, p0s)
	burn = 50
	samples = chain[:, burn:, :].reshape((-1, 12))
	corner.corner(samples)
	plt.show()
	l1,l2,ir,inc,Teq,k,lat1,lon1,rad1,lat2,lon2,rad2 = 	map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*percentile(samples, [16,50,84], axis=0)))
	return l1,l2,ir,inc,Teq,k,lat1,lon1,rad1,lat2,lon2,rad2
	
	
