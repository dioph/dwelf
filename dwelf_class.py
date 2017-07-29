from numpy import *
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.cluster.vq import whiten, kmeans2
import emcee, corner

class Modeler(object):
	def __init__(self):
		self.l1 = 0.70
		self.l2 = 0.00
		self.ir = 0.29
		self.x = linspace(0,45,500)
		self.y = ones(500)
		self.rmin = 1.0
		self.rmax = 1.0
		self.inc_min = 0; self.inc_max = 90
		self.Teq_min = 1; self.Teq_max = 45
		self.k_min = -0.6; self.k_max = 0.6
		self.lat_min = -90; self.lat_max = 90
		self.lon_min =  0; self.lon_max = 360
		self.rad_min = 5; self.rad_max = 25
		self.stdv = 0.003
		
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
		
	def solve(self, theta):
		self.ndim = len(theta)
		n_spots = int((self.ndim-3)/3)
		self.y = self.eker(theta[:6])
		for i in range(2, n_spots+1):
			self.y += -1 + self.eker(append(theta[:3], theta[3*i:3*(i+1)]))
	
	def vsini(self, i, T):
		return [50.592731692185623*self.rmin*sin(i*pi/180)/T, 50.592731692185623*self.rmax*sin(i*pi/180)/T]
	
	def lnprior(self, theta):
		self.ndim = len(theta)
		n_spots = int((self.ndim-3)/3)
		inc, Teq, k = theta[:3]
		if not (self.inc_min < inc < self.inc_max and self.Teq_min < Teq < self.Teq_max and self.k_min < k < self.k_max):
			return -inf
		for i in range(1, n_spots+1):
			if not (self.lat_min < theta[3*i] < self.lat_max and self.lon_min < theta[3*i+1] < self.lon_max and self.rad_min < theta[3*i+2] < self.rad_max):
				return -inf
		return 0.0
	
	def chi(self, theta):
		self.diff = self.y - self.solve(theta)
		return sum(self.diff**2) / self.stdv
	
	def lnprob(self, theta):
		lp = self.lnprior(theta)
		if not isfinite(lp):
			return -inf
		return lp - self.chi(theta)
		
	def normalize(self):
		self.x -= min(self.x)
		good = logical_and(~isnan(self.y), self.x <= 45)
		self.x = self.x[good]
		self.y = self.y[good]
		self.y /= max(self.y)
	
	def eps(self, theta):
		pass
		
