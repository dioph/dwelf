from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.cluster.vq import whiten, kmeans2
import emcee, corner
from time import process_time
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class Modeler(object):
	def __init__(self, l1=0.70, l2=0.00, ir=0.29, x=linspace(0,45,500), y=ones(500), rmin=1.0, rmax=1.0,
				inc_min=0, inc_max=90, Teq_min=1, Teq_max=45, k_min=-0.6, k_max=0.6, lat_min=-90, lat_max=90,
				lon_min=0, lon_max=360, rad_min=5, rad_max=25, stdv=0.003, n_spots=2, n_iter=20, n_clusters=30,
				burn=50, n_walkers=120, n_steps=1000, v_min=0, v_max=10):
		self.l1 = l1
		self.l2 = l2
		self.ir = ir
		self.x = x
		self.y = y
		self.rmin = rmin
		self.rmax = rmax
		self.inc_min = inc_min; self.inc_max = inc_max
		self.Teq_min = Teq_min; self.Teq_max = Teq_max
		self.k_min = k_min; self.k_max = k_max
		self.lat_min = lat_min; self.lat_max = lat_max
		self.lon_min =  lon_min; self.lon_max = lon_max
		self.rad_min = rad_min; self.rad_max = rad_max
		self.stdv = stdv
		self.n_spots = n_spots
		self.n_iter = n_iter
		self.n_clusters = n_clusters
		self.n_dim = 3 + 3*n_spots
		self.burn = burn
		self.n_walkers = n_walkers
		self.n_steps = n_steps
		self.v_min = v_min; self.v_max = v_max
		
	def eker(self, theta):
		d2r = pi / 180
		limb1, limb2, iratio = self.l1, self.l2, self.ir
		inc_deg, Teq, k, lat_deg, lon_deg, rad_deg = theta
		inc = inc_deg * d2r
		lam = lon_deg * d2r
		bet = lat_deg * d2r
		rad = rad_deg * d2r
		period = Teq / (1 - k * sin(bet)**2)
		cosrad = cos(rad)
		sinrad = sin(rad)
		phase = self.x / period
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
		
	def solve(self, theta):
		ndim = len(theta)
		nspots = int((ndim-3)/3)
		y = self.eker(theta[:6])
		for i in range(2, nspots+1):
			y += -1 + self.eker(append(theta[:3], theta[3*i:3*(i+1)]))
		return y
	
	def normalize(self):
		self.x -= min(self.x)
		good = logical_and(~isnan(self.y), self.x <= 45)
		self.x = self.x[good]
		self.y = self.y[good]
		self.y /= max(self.y)
		k = int(len(self.y)/500)
		if len(self.y) > 500:
			chosen = [i % k == 0 for i in range(len(self.y))]
			self.x = self.x[chosen]
			self.y = self.y[chosen]
	
	def vsini(self, i, T):
		return [50.592731692185623*self.rmin*sin(i*pi/180)/T, 50.592731692185623*self.rmax*sin(i*pi/180)/T]
	
	def lnprior(self, theta):
		ndim = len(theta)
		nspots = int((ndim-3)/3)
		inc, Teq, k = theta[:3]
		v = self.vsini(inc, Teq)
		if self.v_max < v[0] or v[1] < self.v_min:
			return -inf
		if not (self.inc_min < inc < self.inc_max and self.Teq_min < Teq < self.Teq_max and self.k_min < k < self.k_max):
			return -inf
		for i in range(1, nspots+1):
			if not (self.lat_min < theta[3*i] < self.lat_max and self.lon_min < theta[3*i+1] < self.lon_max and self.rad_min < theta[3*i+2] < self.rad_max):
				return -inf
		return 0.0
	
	def chi(self, theta, star_params=[]):
		theta = append(star_params, theta)
		self.diff = self.y - self.solve(theta)
		return sum(self.diff**2) / self.stdv
	
	def lnprob(self, theta):
		lp = self.lnprior(theta)
		if not isfinite(lp):
			return -inf
		return lp - self.chi(theta)
		
	def eps(self, theta, star_params=[]):
		theta = append(star_params, theta)
		lp = self.lnprior(theta)
		if not isfinite(lp):
			return ones(len(self.y))
		return self.y - self.solve(theta)
		
	def llsq(self, p0s, n_iter=0, star_params=[]):
		opts = []
		sses = []
		for p0 in p0s:
			p1 = p0[len(star_params):]
			fps, ier = leastsq(self.eps, p1, args=(star_params), maxfev=n_iter)
			if any(isnan(fps)) == False:
				opts.append(append(star_params, fps))
				sses.append(self.chi(fps, star_params))
		return opts, sses
		
	def singlefit(self, p0s, star_params=[]):
		opts, sses = self.llsq(p0s, self.n_iter, star_params=star_params)
		stups = sorted(zip(sses, opts), key=lambda x: x[0])
		sses, opts = zip(*stups)
		optsmat = whiten(array(opts[:int(0.75*len(opts))]))
		centroid, label = kmeans2(optsmat, self.n_clusters, iter=20, minit='points')
		label = list(label)
		p0s = [opts[label.index(i)] for i in range(self.n_clusters) if i in label]
		opts, sses = self.llsq(p0s, star_params=star_params)
		stups = sorted(zip(sses, opts), key=lambda x: x[0])
		sses, opts = zip(*stups)
		return opts[:min(6, len(opts))]
	
	def multifit(self, p0s):
		t1 = process_time()
		opts1 = self.singlefit(p0s)
		t2 = process_time()
		print('FIRST FIT: {0:.2f} s'.format(t2-t1))
		for i in range(1, self.n_spots):
			p = []
			for p1 in opts1:
				y_r = self.y
				self.y = y_r - self.solve(p1) + 1
				opts2 = self.singlefit(p0s, star_params=p1[:3])
				self.y = y_r
				for p2 in opts2:
					p.append(append(p1, p2[3:]))
			t3 = process_time()
			print('MULTI FIT #{1}: {0:.2f} s'.format(t3-t2, i))
			opts, sses = self.llsq(p)
			t4 = process_time()
			print('SIMULFIT #{1}: {0:.2f} s'.format(t4-t3, i))
			stups = sorted(zip(sses, opts), key=lambda x: x[0])
			sses, opts = zip(*stups)
			opts1 = opts
		print('TOTAL: {0:.2f} s'.format(t4-t1))
		return opts[:min(6, len(opts))]
		
	def spacedvals(self):
		p0s = []
		mins = [self.inc_min, self.Teq_min, self.k_min, self.lat_min, self.lon_min, self.rad_min]
		maxs = [self.inc_max, self.Teq_max, self.k_max, self.lat_max, self.lon_max, self.rad_max]
		for i in range(6):
			p0s.append(arange(mins[i]+(maxs[i]-mins[i])/6.0, maxs[i], (maxs[i]-mins[i])/3.0))
		return list(product(*p0s))
		
	def minimize(self):	
		self.normalize()
		p0s = self.spacedvals()
		opts = self.multifit(p0s)
		self.yf = [self.solve(theta) for theta in opts]
		return opts
		
	def mcmc(self, p0s):
		sampler = emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.lnprob)
		print("Running MCMC...")
		t1 = process_time()
		sampler.run_mcmc(p0s, self.n_steps)
		print("Done.")
		t2 = process_time()
		print('Took {0:.3f} seconds'.format(t2-t1))
		return sampler.chain
		
	def fit(self):
		result = self.minimize()
		n = len(result)
		if len(shape(result)) == 1:
			p0s = [result + .1 * random.randn(self.n_dim) for i in range(self.n_walkers)]
		else:
			p0s = [result[j] + .1 * random.randn(self.n_dim) for i in range(int(self.n_walkers/n)) for j in range(n)]
		self.chain = self.mcmc(p0s)
		self.samples = self.chain[:, self.burn:, :].reshape((-1, self.n_dim))
		p = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*percentile(self.samples, [16,50,84], axis=0)))
		return list(p)
	
	def plot_min(self):
		plt.plot(self.x , self.y, 'r.')
		colors = "bgycmk"
		for i in range(len(self.yf)):
			plt.plot(self.x, self.yf[i], colors[i])
		plt.show()
		
	def plot_mcmc(self):
		lab = []
		fig, axes = plt.subplots(self.n_spots+1, 3, sharex=True)	
		axes[0][0].plot(self.chain[:, :, 0].T, color="k", alpha=0.4); axes[0][0].set_ylabel("$i$"); lab.append("$i$")
		axes[0][1].plot(self.chain[:, :, 1].T, color="k", alpha=0.4); axes[0][1].set_ylabel("$P_{eq}$"); lab.append("$P_{eq}$")
		axes[0][2].plot(self.chain[:, :, 2].T, color="k", alpha=0.4); axes[0][2].set_ylabel("$k$"); lab.append("$k$")
		for i in range(1, self.n_spots+1):
			axes[i][0].plot(self.chain[:, :, 3*i].T, color="k", alpha=0.4); axes[i][0].set_ylabel("$\\beta_{0}$".format(i)); lab.append("$\\beta_{0}$".format(i))
			axes[i][1].plot(self.chain[:, :, 3*i+1].T, color="k", alpha=0.4); axes[i][1].set_ylabel("$\lambda_{0}$".format(i)); lab.append("$\lambda_{0}$".format(i))
			axes[i][2].plot(self.chain[:, :, 3*i+2].T, color="k", alpha=0.4); axes[i][2].set_ylabel("$R_{0}$".format(i)); lab.append("$R_{0}$".format(i))
		corner.corner(self.samples, labels=lab)
		plt.show()
		
