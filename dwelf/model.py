import time
from itertools import product

import emcee
import matplotlib.patches as mpatches
from scipy.cluster.vq import whiten, kmeans2
from scipy.optimize import leastsq
from tqdm.auto import tqdm

from .utils import *
from . import MPLSTYLE


class Modeler(object):
    def __init__(self, l1=0.68, l2=0.00, ir=0.22, x=np.arange(0, 50, .1), y=np.ones(500), rmin=0.5, rmax=1.5,
                 inc_min=0, inc_max=90, Teq_min=1, Teq_max=50, k_min=-0.6, k_max=0.6, lat_min=-90, lat_max=90,
                 lon_min=0, lon_max=360, rad_min=5, rad_max=30, stdv=0.001, n_spots=2, n_iter=20, n_clusters=30,
                 burn=100, n_walkers=120, n_steps=1000, v_min=0, v_max=10, threshratio=2, n_temps=1, thin=1,
                 n_spaced=3, savefile=None):
        """Class constructor

        Attributes
        ----------
        l1: linear limb-darkening coefficient
        l2: quadratic limb-darkening coefficient
        ir: spot-to-photosphere intensity ratio
        x: time array
        y: flux array
        rmin, rmax: stellar radius bounds (solar radius)
        inc_min, inc_max: stellar inclination bounds (deg)
        Teq_min, Teq_max: equatorial rotation period bounds (day)
        k_min, k_max: differential rotation coefficient bounds
        lat_min, lat_max: spot latitude bounds (deg)
        lon_min, lon_max: spot longitude bounds (deg)
        rad_min, rad_max: spot radius bounds (deg)
        stdv: stellar activity (# TODO: deprecate)
        n_spots: number of spots
        n_iter: number of initial iterations of the L-M algorithm during the fitting process
        n_clusters: number of clusters found by kmeans to simplify fitting process
        burn: burn-in period of MCMC (recommended: 0.1 * n_steps/thin)
        n_walkers: number of walkers in MCMC (multiple of every integer 1..6)
        v_min, v_max: v sin i bounds (km/s)
        n_steps: number of MCMC steps
        threshratio: multiplying factor to determine threshold of acceptable chi
        thin: thinning MCMC factor
        n_temps: number of temperatures used in Parallel Tempering
        n_spaced: number of spaced values per parameter
        savefile: name of file to save MCMC progress
        """
        self.l1 = l1
        self.l2 = l2
        self.ir = ir
        self.x = x
        self.y = y
        self.rmin = rmin
        self.rmax = rmax
        self.inc_min = inc_min
        self.inc_max = inc_max
        self.Teq_min = Teq_min
        self.Teq_max = Teq_max
        self.k_min = k_min
        self.k_max = k_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.rad_min = rad_min
        self.rad_max = rad_max
        self.stdv = stdv
        self.n_spots = n_spots
        self.n_iter = n_iter
        self.n_clusters = n_clusters
        self.n_dim = 3 + 3 * n_spots
        self.burn = burn
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.v_min = v_min
        self.v_max = v_max
        self.threshratio = threshratio
        self.thin = thin
        self.n_temps = n_temps
        self.n_spaced = n_spaced
        self.savefile = savefile

    def solve(self, theta):
        """Models any number of spots

        Parameters
        ----------
        theta:	star params (inc, Teq, k) + spot params (lat, lon, rad) for each spot

        Returns
        -------
        y: sum of flux contributions from each spot
        """
        ndim = len(theta)
        nspots = int((ndim - 3) / 3)
        y = np.ones_like(self.x)
        for i in range(1, nspots + 1):
            y += -1 + eker(np.append(theta[:3], theta[3 * i:3 * (i + 1)]))
        return y

    def normalize(self, n_bins=500):
        """time and flux validations before any fitting takes place
        """
        # time array starts at epoch 0
        self.x -= np.min(self.x)
        # removed NaNs
        good = ~np.isnan(self.y)
        self.x = self.x[good]
        self.y = self.y[good]
        # maximum number of samples is 500 for faster computation
        cadence = np.max(self.x) / n_bins
        f = np.zeros(n_bins)
        t = np.zeros(n_bins)
        for i in range(n_bins):
            t[i] = i * cadence
            f[i] = np.mean(self.y[np.logical_and(self.x < (i + 1) * cadence, self.x > i * cadence)])
        self.x = t[~np.isnan(f)]
        self.y = f[~np.isnan(f)]
        # normalize flux
        if np.abs(np.mean(self.y)) > .01:
            self.y /= np.mean(self.y)

    def vsini(self, i, T):
        """Calculates minimum and maximum v sin i given inclination and period at equator.
        Considers bounds in solar units.
        (1 solar radius = 695700 km)

        Parameters
        ----------
        i: stellar inclination
        T: equatorial rotation period

        Returns
        -------
        vmin, vmax: v sin i bounds (km/s)
        """
        vmin = 2 * np.pi * 695700 * self.rmin * np.sin(i * np.pi / 180) / (T * 86400)
        vmax = vmin * self.rmax / self.rmin
        return vmin, vmax

    def lnprior(self, theta):
        """log of prior probability of theta (limits the search scope)

        Parameters
        ----------
        theta:	star params (inc, Teq, k) + spot params (lat, lon, rad) for each spot

        Returns
        -------
        -inf if out of defined limits (0%)
        0.0 otherwise (100%)
        """
        ndim = len(theta)
        nspots = int((ndim - 3) / 3)
        inc, Teq, k = theta[:3]
        v = self.vsini(inc, Teq)
        # restrict v sin i
        if self.v_max < v[0] or v[1] < self.v_min:
            return -np.inf
        # restrict star params
        if not (self.inc_min < inc < self.inc_max and self.Teq_min < Teq < self.Teq_max and
                self.k_min < k < self.k_max):
            return -np.inf
        # restrict spot params
        for i in range(1, nspots + 1):
            if not (self.lat_min < theta[3 * i] < self.lat_max and self.lon_min < theta[3 * i + 1] < self.lon_max and
                    self.rad_min < theta[3 * i + 2] < self.rad_max):
                return -np.inf
        return 0.0

    def chi(self, theta, star_params=None):
        """Negative log of likelihood function (sum squared error)
        """
        if star_params is None:
            star_params = []
        theta = np.append(star_params, theta)
        error = self.y - self.solve(theta)
        return np.sum(np.square(error)) / self.stdv

    def lnprob(self, theta):
        """log of posterior probability of theta (prior * likelihood)
        """
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - self.chi(theta)

    def eps(self, theta, star_params=None):
        """Returns array of residuals between light curve and fit (unless lnprior == -inf)
        """
        if star_params is None:
            star_params = []
        theta = np.append(star_params, theta)
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return np.ones_like(self.y)
        return self.y - self.solve(theta)

    def llsq(self, p0s, n_iter=0, star_params=None):
        """Runs Levenberg-Marquardt algorithm for each initial point p0

        Parameters
        ----------
        p0s: array of initial guesses
        n_iter: maximum number of function evaluations for the L-M algorithm
        star_params: (inc, Teq, k), required for multiple spots fitting (fixed star)

        Returns
        -------
        opts: array of best fits for each initial guess
        sses: array of chi values for each best fit
        """
        if star_params is None:
            star_params = []
        opts = []
        sses = []
        for p0 in p0s:
            p1 = p0[len(star_params):]
            fps = leastsq(func=self.eps, x0=p1, args=star_params, maxfev=n_iter)
            if not any(np.isnan(fps)):
                opts.append(np.append(star_params, fps))
                sses.append(self.chi(fps, star_params))
        return opts, sses

    def singlefit(self, p0s, star_params=None):
        """Fits one single spot to current light curve

        Parameters
        ----------
        p0s: array of initial guesses
        star_params: (inc, Teq, k)

        Returns
        -------
        bestps: array of best fits with chi <= threshold (limits to 6 for faster computation)
        """
        if star_params is None:
            star_params = []
        # initial fit with few (n_iter) iterations
        opts, sses = self.llsq(p0s, self.n_iter, star_params=star_params)
        # sort fits with respect to chi
        sses, opts = zip(*sorted(zip(sses, opts), key=lambda x: x[0]))
        # let all parameters have same variance (enable clustering)
        optsmat = whiten(np.array(opts))
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
        bestps = np.array(opts)[np.array(sses) <= threshold]
        return bestps[:min(6, len(bestps))]

    def multifit(self, p0s, verbose=True):
        """Fits multiple spots using greedy algorithm

        Parameters
        ----------
        p0s: array of initial guesses
        verbose: whether to print time elapsed messages
        """
        t1 = time.perf_counter()
        # fit first (hopefully larger) spot
        opts1 = self.singlefit(p0s)
        t2 = time.perf_counter()
        if verbose:
            print('FIRST FIT: {0:.2f} s'.format(t2 - t1))
        opts = np.array([])
        for i in range(1, self.n_spots):
            t2 = time.perf_counter()
            p = []
            for p1 in opts1:
                y_r = self.y
                # let current light curve be the residual from previously fitted spots
                self.y = y_r - self.solve(p1) + 1
                opts2 = self.singlefit(p0s, star_params=p1[:3])
                # retrieve original light curve
                self.y = y_r
                for p2 in opts2:
                    p.append(np.append(p1, p2[3:]))
            t3 = time.perf_counter()
            if verbose:
                print('MULTIFIT #{1}: {0:.2f} s'.format(t3 - t2, i))
            # for each new spot, do a simultaneous fit of all parameters so far
            opts, sses = self.llsq(p)
            t4 = time.perf_counter()
            if verbose:
                print('SIMULFIT #{1}: {0:.2f} s'.format(t4 - t3, i))
            # sort fits with respect to chi
            sses, opts = zip(*sorted(zip(sses, opts), key=lambda x: x[0]))
            # opts stores all spots fitted so far
            opts1 = opts
        t4 = time.perf_counter()
        if verbose:
            print('TOTAL: {0:.2f} s'.format(t4 - t1))
        return opts

    def spacedvals(self, method='default'):
        """Defines (n_spaced)**6 initial points spaced in allowed parameter region
        """
        p0s = []
        mins = np.array([self.inc_min, self.Teq_min, self.k_min,
                         self.lat_min, self.lon_min, self.rad_min])
        maxs = np.array([self.inc_max, self.Teq_max, self.k_max,
                         self.lat_max, self.lon_max, self.rad_max])
        if method == 'random':
            p0s = np.random.rand(self.n_spaced, 6)
            q = np.zeros_like(p0s)
            q = p0s * (maxs - mins) + mins
        else:
            for i in range(6):
                p0s.append(np.arange(mins[i] + (maxs[i] - mins[i]) / (2 * self.n_spaced), maxs[i],
                                     (maxs[i] - mins[i]) / self.n_spaced))
            q = list(product(*p0s))
            np.random.shuffle(q)
        return q

    def minimize(self):
        """Find parameters of local minima of chi
        Saves best fits for plotting if required.
        """
        self.normalize()
        p0s = self.spacedvals(method='random')
        if self.n_spots > 1:
            opts = self.multifit(p0s)
        else:
            opts = self.singlefit(p0s)
        self.yf = [self.solve(theta) for theta in opts]
        self.bestps = opts
        return opts

    def mcmc(self, p0s):
        """Runs Monte Carlo Markov Chain algorithm to determine uncertainties
        """
        sampler = emcee.EnsembleSampler(self.n_walkers, self.n_dim, self.lnprob)
        if self.savefile:
            f = open(self.savefile, "w")
            f.close()
        for result in tqdm(sampler.sample(p0s, iterations=self.n_steps, thin=self.thin), total=self.n_steps):
            if self.savefile:
                position = result[0]
                f = open(self.savefile, "a")
                for k in range(position.shape[0]):
                    f.write("{0:4d} {1:s}\n".format(k, " ".join([str(pos) for pos in position[k]])))
                f.close()
        return sampler

    def ptmcmc(self, p0s):
        """Runs Parallel Tempering Monte Carlo Markov Chain algorithm to determine multi-modal uncertainties
        """

        def logl(theta):
            return -self.chi(theta)

        sampler = emcee.PTSampler(self.n_temps, self.n_walkers, self.n_dim, logl, self.lnprior)
        if self.savefile:
            f = open(self.savefile, "w")
            f.close()
        for result in tqdm(sampler.sample(p0s, iterations=self.n_steps, thin=self.thin), total=self.n_steps):
            if self.savefile:
                position = result[0]
                f = open(self.savefile, "a")
                for k in range(position.shape[0]):
                    f.write("{0:4d} {1:s}\n".format(k, " ".join([str(pos) for pos in position[k]])))
                f.close()
        return sampler

    def fit(self):
        """Quick way to run whole fitting + MCMC process

        Returns
        -------
        p: list of obtained parameters and uncertainties
        """
        result = self.minimize()
        n = min(len(result), self.n_temps)
        self.n_temps = n
        if n == 1:
            # initialize walkers in a ball around best fit
            p0s = [result[0] + .1 * np.random.randn(self.n_dim) for i in range(self.n_walkers)]
            self.sampler = self.mcmc(p0s)
            self.chain = self.sampler.chain
        else:
            # initialize walkers in n balls around best fits
            p0s = [[result[j] + .1 * np.random.randn(self.n_dim) for i in range(self.n_walkers)] for j in range(n)]
            self.sampler = self.ptmcmc(p0s)
            self.chain = self.sampler.chain[0]
        # cut burn-in period
        self.samples = self.chain[:, self.burn:, :].reshape((-1, self.n_dim))
        # 16th and 84th percentiles give the marginalized distributions  
        p = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(self.samples, [16, 50, 84], axis=0)))
        return list(p)

    def plot_min(self):
        """Nice plot of best fits
        """
        colors = "bgycmk"
        with plt.style.context(MPLSTYLE):
            for i in range(np.shape(self.yf)[0]):
                plt.figure()
                label = ""
                for j in self.bestps[i]:
                    label += str("{0:.2f}".format(j)) + ", "
                label = label[:-2]
                patch = mpatches.Patch(color=colors[i], label=label)
                plt.legend(handles=[patch])
                plt.plot(self.x, self.y, 'r.')
                plt.plot(self.x, self.yf[i], colors[i], linewidth=2)
            plt.show()
