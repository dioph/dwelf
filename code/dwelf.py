from numpy import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import leastsq
from scipy.cluster.vq import whiten, kmeans2
import emcee
import corner
import time
from itertools import product
import sys
import warnings

warnings.filterwarnings('ignore')


class Modeler(object):
    def __init__(self, l1=0.68, l2=0.00, ir=0.22, x=linspace(0, 45, 500), y=ones(500), rmin=0.5, rmax=1.5,
                 inc_min=0, inc_max=90, Teq_min=1, Teq_max=45, k_min=-0.6, k_max=0.6, lat_min=-90, lat_max=90,
                 lon_min=0, lon_max=360, rad_min=5, rad_max=30, stdv=0.001, n_spots=2, n_iter=20, n_clusters=30,
                 burn=100, n_walkers=120, n_steps=1000, v_min=0, v_max=10, threshratio=2, n_temps=1, thin=1,
                 n_spaced=3, savefile=None):
        """
        class constructor
        """
        self.l1 = l1    # linear limb-darkening
        self.l2 = l2    # quadratic limb-darkening
        self.ir = ir    # spot-to-photosphere intensity ratio
        self.x = x  # time array
        self.y = y  # flux array
        self.rmin = rmin; self.rmax = rmax  # radius (in solar units)
        self.inc_min = inc_min; self.inc_max = inc_max  # stellar inclination
        self.Teq_min = Teq_min; self.Teq_max = Teq_max  # period at equator
        self.k_min = k_min; self.k_max = k_max  # differential rotation
        self.lat_min = lat_min; self.lat_max = lat_max  # spot latitude
        self.lon_min = lon_min; self.lon_max = lon_max  # spot longitude
        self.rad_min = rad_min; self.rad_max = rad_max  # spot radius (in degrees)
        self.stdv = stdv    # arbitrary denominator to normalize chi closer to 1.0
        self.n_spots = n_spots  # number of spots
        self.n_iter = n_iter    # number of initial iterations of the L-M algorithm during the fitting process
        self.n_clusters = n_clusters    # number of clusters found by kmeans to simplify fitting process
        self.n_dim = 3 + 3 * n_spots    # number of parameters to fit (3 star params + 3 spot params for each spot)
        self.burn = burn    # burn-in period of MCMC (recommended: 0.1 * n_steps/thin)
        self.n_walkers = n_walkers  # number of walkers in MCMC (multiple of every integer 1..6)
        self.n_steps = n_steps  # number of steps in MCMC
        self.v_min = v_min; self.v_max = v_max  # v sin i (in km/s)
        self.threshratio = threshratio  # multiplying factor to determine threshold of acceptable chi
        self.thin = thin    # thinning factor used in MCMC
        self.n_temps = n_temps  # number of temperatures used in Parallel Tempering
        self.n_spaced = n_spaced    # number of spaced values per parameter
        self.savefile = savefile    # name of file to save MCMC progress

    def eker(self, theta):
        """
        analytic model of one circular spot based on equations in Eker (1994, ApJ, 420, 373)
        theta[0..5]:	star params (inc, Teq, k) + spot params (lat, lon, rad)
        """
        limb1, limb2, iratio = self.l1, self.l2, self.ir
        inc_deg, Teq, k, lat_deg, lon_deg, rad_deg = theta
        # convert angles from degrees to radians
        inc = radians(inc_deg)
        lam = radians(lon_deg)
        bet = radians(lat_deg)
        rad = radians(rad_deg)
        period = Teq / (1 - k * sin(bet) ** 2)  # differential rotation
        # useful scalar quantities
        cosrad = cos(rad)
        sinrad = sin(rad)
        # rotational phases
        phase = self.x / period
        phi = 2.0 * pi * phase
        nphi = len(phi)
        # angle the0 between two vectors originating from spot center:
        # 1) normal to stellar surface, directed away from center of star
        # 2) directed towards the observer
        costhe0 = cos(inc) * sin(bet) + sin(inc) * cos(bet) * cos(phi - lam)
        sinthe0 = sqrt(1.0 - costhe0 ** 2)
        the0 = arccos(costhe0)
        # find phases when spot is full, gibbous, crescent or occulted
        jf = flatnonzero(the0 <= pi / 2 - rad)
        nf = len(jf)
        jg = flatnonzero(logical_and(the0 > pi / 2 - rad, the0 <= pi / 2))
        ng = len(jg)
        jc = flatnonzero(logical_and(the0 > pi / 2, the0 <= pi / 2 + rad))
        nc = len(jc)
        jo = flatnonzero(the0 > pi / 2 + rad)
        no = len(jo)
        # allocate arrays for integrals
        ic = zeros(nphi)    # constant intensity term
        il = zeros(nphi)    # linear intensity term
        iq = zeros(nphi)    # quadratic intensity term
        #
        # FULL (entirely visible)
        #
        if nf >= 1:
            costhe0_f = costhe0[jf]
            sinthe0_f = sinthe0[jf]
            ic[jf] = pi * sin(rad) ** 2 * costhe0_f
            il[jf] = 2 * pi / 3 * (1 - cosrad ** 3) - \
                pi * cosrad * sinrad ** 2 * sinthe0_f ** 2
            iq[jf] = pi / 2 * (1 - cosrad ** 4) * costhe0_f ** 3 + \
                3 * pi / 4 * sinrad ** 4 * costhe0_f * sinthe0_f ** 2
        #
        # GIBBOUS (more than half visible)
        #
        if ng >= 1:
            the0_g = the0[jg]
            costhe0_g = costhe0[jg]
            sinthe0_g = sinthe0[jg]
            cosphi0_g = - 1.0 / (tan(the0_g) * tan(rad))
            rad0_g = abs(the0_g - pi / 2)
            phi0_g = arccos(cosphi0_g)
            sinphi0_g = sqrt(1.0 - cosphi0_g ** 2)
            cosrad0_g = cos(rad0_g)
            sinrad0_g = sin(rad0_g)
            k1_g = ((pi - phi0_g) / 4) * (cosrad0_g ** 4 - cosrad ** 4)
            k2_g = (sinphi0_g / 8) * (rad0_g - rad + 0.5 * (sin(2 * rad)
                                                            * cos(2 * rad) - sin(2 * rad0_g) * cos(2 * rad0_g)))
            k3_g = (1.0 / 8) * (pi - phi0_g - sinphi0_g *
                                cosphi0_g) * (sinrad ** 4 - sinrad0_g ** 4)
            k4_g = - (sinphi0_g - sinphi0_g ** 3 / 3) * ((3.0 / 8) * (rad - rad0_g) + (1.0 / 16)
                                                     * (sin(2 * rad) * (cos(2 * rad) - 4) - sin(2 * rad0_g) * (cos(2 * rad0_g) - 4)))
            cl_g = ((pi - phi0_g) / 3) * (cosrad ** 3 - cosrad0_g ** 3) * (1 - 3 * costhe0_g ** 2) - (pi - phi0_g - sinphi0_g * cosphi0_g) * (cosrad - cosrad0_g) * sinthe0_g ** 2 - \
                (4.0 / 3) * sinphi0_g * (sinrad ** 3 - sinrad0_g ** 3) * sinthe0_g * costhe0_g - \
                (1.0 / 3) * sinphi0_g * cosphi0_g * \
                (cosrad ** 3 - cosrad0_g ** 3) * sinthe0_g ** 2
            cq_g = 2 * costhe0_g ** 3 * k1_g + 6 * costhe0_g ** 2 * sinthe0_g * \
                k2_g + 6 * costhe0_g * sinthe0_g ** 2 * k3_g + 2 * sinthe0_g ** 3 * k4_g
            ic[jg] = phi0_g * costhe0_g * sinrad ** 2 - \
                arcsin(cosrad / sinthe0_g) - 0.5 * sinthe0_g * \
                sinphi0_g * sin(2 * rad) + pi / 2
            il[jg] = 2 * pi / 3 * (1 - cosrad ** 3) - pi * \
                cosrad * sinrad ** 2 * sinthe0_g ** 2 - cl_g
            iq[jg] = pi / 2 * (1 - cosrad ** 4) * costhe0_g ** 3 + 3 * \
                pi / 4 * sinrad ** 4 * costhe0_g * sinthe0_g ** 2 - cq_g
        #
        # CRESCENT (less than half visible)
        #
        if nc >= 1:
            the0_c = the0[jc]
            costhe0_c = costhe0[jc]
            sinthe0_c = sinthe0[jc]
            cosphi0_c = - 1.0 / (tan(the0_c) * tan(rad))
            rad0_c = abs(the0_c - pi / 2)
            phi0_c = arccos(cosphi0_c)
            sinphi0_c = sqrt(1.0 - cosphi0_c ** 2)
            cosrad0_c = cos(rad0_c)
            sinrad0_c = sin(rad0_c)
            k1_c = (phi0_c / 4) * (cosrad0_c ** 4 - cosrad ** 4)
            k2_c = - (sinphi0_c / 8) * (rad0_c - rad + 0.5 * (sin(2 * rad)
                                                              * cos(2 * rad) - sin(2 * rad0_c) * cos(2 * rad0_c)))
            k3_c = (1.0 / 8) * (phi0_c + sinphi0_c * cosphi0_c) * \
                (sinrad ** 4 - sinrad0_c ** 4)
            k4_c = (sinphi0_c - sinphi0_c ** 3 / 3) * ((3.0 / 8) * (rad - rad0_c) + (1.0 / 16)
                                                       * (sin(2 * rad) * (cos(2 * rad) - 4) - sin(2 * rad0_c) * (cos(2 * rad0_c) - 4)))
            cq_c = 2 * costhe0_c ** 3 * k1_c + 6 * costhe0_c ** 2 * sinthe0_c * \
                k2_c + 6 * costhe0_c * sinthe0_c ** 2 * k3_c + 2 * sinthe0_c ** 3 * k4_c
            ic[jc] = phi0_c * costhe0_c * sinrad ** 2 - \
                arcsin(cosrad / sinthe0_c) - 0.5 * sinthe0_c * \
                sinphi0_c * sin(2 * rad) + pi / 2
            il[jc] = (phi0_c / 3) * (cosrad ** 3 - cosrad0_c ** 3) * (1 - 3 * costhe0_c ** 2) - (phi0_c + sinphi0_c * cosphi0_c) * (cosrad - cosrad0_c) * sinthe0_c ** 2 + (
                4.0 / 3) * sinphi0_c * (sinrad ** 3 - sinrad0_c ** 3) * sinthe0_c * costhe0_c + (1.0 / 3) * sinphi0_c * cosphi0_c * (cosrad ** 3 - cosrad0_c ** 3) * sinthe0_c ** 2
            iq[jc] = cq_c
        #
        # OCCULTED (back of the star)
        #
        if no >= 1:
            ic[jo] = 0.0
            il[jo] = 0.0
            iq[jo] = 0.0
        # calculate lightcurve (equation 12c from Eker, 1994)
        lc = 1.0 + (iratio - 1.0) / (pi * (1.0 - limb1 / 3.0 + limb2 / 6.0)) * (
            (1.0 - limb1 + limb2) * ic + (limb1 - 2.0 * limb2) * il + limb2 * iq)
        return lc

    def solve(self, theta):
        """
        models any number of spots
        theta:	star params (inc, Teq, k) + spot params (lat, lon, rad) for each spot
        """
        ndim = len(theta)
        nspots = int((ndim - 3) / 3)
        y = self.eker(theta[:6])
        for i in range(2, nspots + 1):
            y += -1 + self.eker(append(theta[:3], theta[3*i:3*(i+1)]))
        return y

    def normalize(self):
        """
        x and y corrections before any fitting takes place
        """
        self.x -= min(self.x)   # time array starts at epoch 0 and is restricted to be <= 45
        good = ~isnan(self.y)   #logical_and(~isnan(self.y), self.x <= 45)  # removed NaNs
        self.x = self.x[good]
        self.y = self.y[good]
        self.y /= max(self.y)   # flux normalized
        m = median(self.y)
        rms = sqrt(mean(square(self.y/m-1)))
        r1 = m + m*sqrt(2)*rms
        self.y /= r1
        k = int(len(self.y) / 500)
        if len(self.y) > 500:
            chosen = [i % k == 0 for i in
                      range(len(self.y))]   # maximum number of samples is 500 for faster computation
            self.x = self.x[chosen]
            self.y = self.y[chosen]

    def vsini(self, i, T):
        """
        calculates minimum and maximum v sin i given inclination and period at equator
        considers the radius limits in solar units
        """
        # solar radius = 695700 km
        # 1 day = 86400 seconds
        # 2 * pi * 695700 / 86400 = 50.592731692185623
        return [50.592731692185623 * self.rmin * sin(i * pi / 180) / T,
                50.592731692185623 * self.rmax * sin(i * pi / 180) / T]

    def lnprior(self, theta):
        """
        log of prior probability of theta (limits the search scope)
        -inf if out of defined limits (0%)
        0.0 otherwise (100%)
        """
        ndim = len(theta)
        nspots = int((ndim - 3) / 3)
        inc, Teq, k = theta[:3]
        v = self.vsini(inc, Teq)
        if self.v_max < v[0] or v[1] < self.v_min:
            return -inf # restrict v sin i
        if not (self.inc_min < inc < self.inc_max and self.Teq_min < Teq < self.Teq_max and self.k_min < k < self.k_max):
            return -inf # restrict star params
        for i in range(1, nspots+1):
            if not (self.lat_min < theta[3*i] < self.lat_max and self.lon_min < theta[3*i + 1]
             < self.lon_max and self.rad_min < theta[3 * i + 2] < self.rad_max):
                return -inf # restrict spot params
        return 0.0

    def chi(self, theta, star_params=None):
        """
        negative log of likelihood function (sum squared error)
        """
        if star_params is None:
            star_params = []
        theta = append(star_params, theta)
        diff = self.y - self.solve(theta)
        return sum(diff ** 2) / self.stdv

    def lnprob(self, theta):
        """
        log of posterior probability of theta (prior * likelihood)
        """
        lp = self.lnprior(theta)
        if not isfinite(lp):
            return -inf
        return lp - self.chi(theta)

    def eps(self, theta, star_params=None):
        """
        returns array of residuals between lightcurve and fit (unless lnprior == -inf)
        """
        if star_params is None:
            star_params = []
        theta = append(star_params, theta)
        lp = self.lnprior(theta)
        if not isfinite(lp):
            return ones_like(self.y)
        return self.y - self.solve(theta)

    def llsq(self, p0s, n_iter=0, star_params=None):
        """
        runs Levenberg-Marquardt algorithm for each initial point p0
        star_params required for multiple spots fitting (fixed star)
        """
        if star_params is None:
            star_params = []
        opts = []
        sses = []
        for p0 in p0s:
            p1 = p0[len(star_params):]
            fps, __ = leastsq(self.eps, p1, args=(star_params), maxfev=n_iter)
            if not any(isnan(fps)):
                opts.append(append(star_params, fps))
                sses.append(self.chi(fps, star_params))
        return opts, sses

    def singlefit(self, p0s, star_params=None):
        """
        fits one single spot to current lightcurve
        returns best fits with chi <= threshold (max 6 for faster computation)
        """
        if star_params is None:
            star_params = []
        # initial fit with few (n_iter) iterations
        opts, sses = self.llsq(p0s, self.n_iter, star_params=star_params)
        # sort fits with respect to chi
        sses, opts = zip(*sorted(zip(sses, opts), key=lambda x: x[0]))
        # let all parameters have same variance (enable clustering)
        optsmat = whiten(array(opts[:int(0.75 * len(opts))]))
        # find (n_clusters) centroids using kmeans
        __, label = kmeans2(optsmat, self.n_clusters, iter=20, minit='points')
        label = list(label)
        # new corresponding initial points
        p0s = [opts[label.index(i)] for i in range(self.n_clusters) if i in label]
        # final fit with full iterations
        opts, sses = self.llsq(p0s, star_params=star_params)
        # sort fits with respect to chi
        sses, opts = zip(*sorted(zip(sses, opts), key=lambda x: x[0]))
        threshold = sses[0] * self.threshratio
        # fits with chi <= given threshold
        bestps = array(opts)[array(sses) <= threshold]
        return bestps[:min(6, len(bestps))]

    def multifit(self, p0s):
        """
        fits multiple spots using greedy algorithm
        """
        t1 = time.perf_counter()
        opts1 = self.singlefit(p0s) # fit first (hopefully larger) spot
        t2 = time.perf_counter()
        print('FIRST FIT: {0:.2f} s'.format(t2-t1))
        opts = array([])
        for i in range(1, self.n_spots):
            t2 = time.perf_counter()
            p = []
            for p1 in opts1:
                y_r = self.y
                # let current lightcurve be the residual from previously fitted spots
                self.y = y_r - self.solve(p1) + 1
                opts2 = self.singlefit(p0s, star_params=p1[:3])
                self.y = y_r    # retrieve original lightcurve
                for p2 in opts2:
                    p.append(append(p1, p2[3:]))
            t3 = time.perf_counter()
            print('MULTIFIT #{1}: {0:.2f} s'.format(t3-t2, i))
            # for each new spot, do a simultaneous fit of all parameters so far
            opts, sses = self.llsq(p)
            t4 = time.perf_counter()
            print('SIMULFIT #{1}: {0:.2f} s'.format(t4-t3, i))
            # sort fits with respect to chi
            sses, opts = zip(*sorted(zip(sses, opts), key=lambda x: x[0]))
            opts1 = opts    # opts stores all spots fitted so far
        t4 = time.perf_counter()
        print('TOTAL: {0:.2f} s'.format(t4-t1))
        return opts

    def spacedvals(self):
        """
        defines (n_spaced)**6 initial points spaced in allowed parameter region
        """
        p0s = []
        mins = [self.inc_min, self.Teq_min, self.k_min,
                self.lat_min, self.lon_min, self.rad_min]
        maxs = [self.inc_max, self.Teq_max, self.k_max,
                self.lat_max, self.lon_max, self.rad_max]
        for i in range(6):
            p0s.append(arange(mins[i] + (maxs[i] - mins[i]) / (2 * self.n_spaced), maxs[i],
                              (maxs[i] - mins[i]) / self.n_spaced))
        q = list(product(*p0s))
        random.shuffle(q)
        return q

    def minimize(self):
        """
        find parameters of local minima of chi
        saves best fits for plotting if required
        """
        self.normalize()
        p0s = self.spacedvals()
        if self.n_spots > 1:
            opts = self.multifit(p0s)
        else:
            opts = self.singlefit(p0s)
        self.yf = [self.solve(theta) for theta in opts]
        self.bestps = opts
        return opts

    def mcmc(self, p0s):
        """
        runs Monte Carlo Markov Chain algorithm to determine uncertainties
        """
        sampler = emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.lnprob)
        print("Running MCMC...")
        t1 = time.perf_counter()
        if self.savefile:
            f = open(self.savefile, "w")
            f.close()
        # progress bar:
        width = 30
        for i, result in enumerate(sampler.sample(
                p0s, iterations=self.n_steps, thin=self.thin)):
            n = int((width + 1) * float(i) / self.n_steps)
            sys.stdout.write(
                "\r[{0}{1}] {2:.1f}%".format('#'*n, ' '*(width-n), 100*float(i+1)/self.n_steps))
            if self.savefile:
                position = result[0]
                f = open(self.savefile, "a")
                for k in range(position.shape[0]):
                    f.write("{0:4d} {1:s}\n".format(
                        k, " ".join([str(pos) for pos in position[k]])))
                f.close()
        sys.stdout.write("\n")
        print("Done.")
        t2 = time.perf_counter()
        print('Took {0:.3f} seconds'.format(t2-t1))
        return sampler

    def ptmcmc(self, p0s):
        """
        runs Parallel Tempering Monte Carlo Markov Chain algorithm to determine multi-modal uncertainties
        """
        def logl(theta): return -self.chi(theta)
        sampler = emcee.PTSampler(self.n_temps, self.n_walkers, self.n_dim, logl, self.lnprior)
        print("Running MCMC...")
        t1 = time.perf_counter()
        if self.savefile:
            f = open(self.savefile, "w")
            f.close()
        # progress bar:
        width = 30
        for i, result in enumerate(sampler.sample(
                p0s, iterations=self.n_steps, thin=self.thin)):
            n = int((width + 1) * float(i) / self.n_steps)
            sys.stdout.write(
                "\r[{0}{1}] {2:.1f}%".format('#'*n, ' '*(width-n), 100*float(i+1)/self.n_steps))
            if self.savefile:
                position = result[0]
                f = open(self.savefile, "a")
                for k in range(position.shape[0]):
                    f.write("{0:4d} {1:s}\n".format(
                        k, " ".join([str(pos) for pos in position[k]])))
                f.close()
        sys.stdout.write("\n")
        print("Done.")
        t2 = time.perf_counter()
        print('Took {0:.3f} seconds'.format(t2-t1))
        return sampler

    def fit(self):
        """
        quick way to run whole fitting + MCMC process
        returns list of obtained parameters and uncertainties
        """
        result = self.minimize()
        n = min(len(result), self.n_temps)
        self.n_temps = n
        if n == 1:
            # initialize walkers in a ball around best fit
            p0s = [result[0] + .1 * random.randn(self.n_dim) for i in range(self.n_walkers)]
            self.sampler = self.mcmc(p0s)
            self.chain = self.sampler.chain
        else:
            # initialize walkers in n balls around best fits
            p0s = [[result[j] + .1 * random.randn(self.n_dim) for i in range(self.n_walkers)] for j in range(n)]
            self.sampler = self.ptmcmc(p0s)
            self.chain = self.sampler.chain[0]
        # cut burn-in period
        self.samples = self.chain[:, self.burn:, :].reshape((-1, self.n_dim))
        # 16th and 84th percentiles give the marginalized distributions  
        p = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                zip(*percentile(self.samples, [16, 50, 84], axis=0)))
        return list(p)

    def plot_min(self):
        """
        nice plot of best fits
        """
        colors = "bgycmk"
        for i in range(shape(self.yf)[0]):
            plt.figure()
            label = ""
            for j in self.bestps[i]:
                label += str("{0:.2f}".format(j)) + ", "
            label = label[:-2]
            patch = mpatches.Patch(color=colors[i], label=label)
            plt.legend(handles=[patch])
            plt.plot(self.x, self.y, 'r.')
            plt.plot(self.x, self.yf[i], colors[i], linewidth=2)
            plt.grid()
            plt.minorticks_on()
        plt.show()

    def plot_mcmc(self):
        """
        MCMC standard visualization
        """
        lab = []
        fig, axes = plt.subplots(self.n_spots+1, 3, sharex=True)
        axes[0][0].plot(self.chain[:, :, 0].T, color="k", alpha=0.4)
        axes[0][0].set_ylabel("$i$")
        lab.append("$i$")
        axes[0][1].plot(self.chain[:, :, 1].T, color="k", alpha=0.4)
        axes[0][1].set_ylabel("$P_{eq}$")
        lab.append("$P_{eq}$")
        axes[0][2].plot(self.chain[:, :, 2].T, color="k", alpha=0.4)
        axes[0][2].set_ylabel("$k$")
        lab.append("$k$")
        for i in range(1, self.n_spots+1):
            axes[i][0].plot(self.chain[:, :, 3 * i].T, color="k", alpha=0.4)
            axes[i][0].set_ylabel("$\\beta_{0}$".format(i))
            lab.append("$\\beta_{0}$".format(i))
            axes[i][1].plot(self.chain[:, :, 3 * i + 1].T,
                            color="k", alpha=0.4)
            axes[i][1].set_ylabel("$\lambda_{0}$".format(i))
            lab.append("$\lambda_{0}$".format(i))
            axes[i][2].plot(self.chain[:, :, 3 * i + 2].T,
                            color="k", alpha=0.4)
            axes[i][2].set_ylabel("$R_{0}$".format(i))
            lab.append("$R_{0}$".format(i))
        corner.corner(self.samples[:,:3], labels=lab[:3], quantiles=[0.16, 0.5, 0.84], show_titles=True,
                      title_kwargs={"fontsize": 12})
        for i in range(1, self.n_spots+1):
            corner.corner(self.samples[:,3*i:3*(i+1)], labels=lab[3*i:3*(i+1)], quantiles=[0.16, 0.5, 0.84], show_titles=True,
                      title_kwargs={"fontsize": 12})
        plt.show()
