import time
from itertools import product

import emcee
import matplotlib.patches as mpatches
from pymultinest.solve import solve
from scipy.cluster.vq import whiten, kmeans2
from scipy.optimize import leastsq, minimize
from tqdm.auto import tqdm

from dwelf import MPLSTYLE
from dwelf.priors import *
from dwelf.utils import *


class MaculaModeler(object):
    def __init__(self, t, y, nspots, dy=None,
                 Pvec=None, k2=None, k4=None, c=None, d=None, same_limb=False,
                 lon=None, lat=None, alpha=None, fspot=None, tmax=None, life=None, ingress=None, egress=None,
                 U=None, B=None, tstart=None, tend=None):
        """Class constructor

        Attributes
        ----------
        t: time array
        y: flux array
        nspots: number of spots
        dy: flux uncertainties
        Pvec: 2-D array containing rotation period at equator and stellar inclination
        k2: 2nd-order differential rotation coefficient
        k4: 4th-order differential rotation coefficient
        c: stellar limb-darkening coefficients
        d: spot limb-darkening coefficients
        same_limb: whether to always assume c==d in the model
        lon: spot longitudes (rad)
        lat: spot latitudes (rad)
        alpha: spot radius (rad)
        fspot: spot-to-photosphere intensity ratio
        tmax: time of greatest spot area
        life: spot lifetimes
        ingress: spot ingress times
        egress: spot egress times
        U: unspotted surface flux value
        B: instrumental blending factor
        tstart: start time for each of the stitched curves
        tend: end time for each of the stitched curves
        """
        self.t = t
        self.y = y
        self.dy = dy
        self.nspots = nspots

        self.Pvec = Pvec
        self.k2 = k2
        self.k4 = k4
        self.c = c
        self.d = d
        self.same_limb = same_limb

        self.lon = lon
        self.lat = lat
        self.alpha = alpha
        self.fspot = fspot
        self.tmax = tmax
        self.life = life
        self.ingress = ingress
        self.egress = egress

        self.U = U
        self.B = B
        self.tstart = tstart
        self.tend = tend

        if self.dy is None:
            self.dy = np.ones_like(self.y)

        # validate star params
        if self.Pvec is None:
            self.Pvec = Uniform(ndim=2, xmin=(0, 0), xmax=(1, 50))
        if self.k2 is None:
            self.k2 = Uniform(-1, 1)
        if self.k4 is None:
            self.k4 = Uniform(-1, 1)

        if self.c is None:
            self.c = Uniform(ndim=4, xmin=(-1, -1, -1, -1), xmax=(1, 1, 1, 1))
        if self.d is None:
            self.d = Uniform(ndim=4, xmin=(-1, -1, -1, -1), xmax=(1, 1, 1, 1))

        if self.same_limb:
            self.d = SameAs(self.c)

        # validate spot params
        if self.lon is None:
            self.lon = Uniform(ndim=self.nspots, xmin=-np.pi, xmax=np.pi)
        if self.lat is None:
            self.lat = Uniform(ndim=self.nspots, xmin=-np.pi / 2, xmax=np.pi / 2)
        if self.alpha is None:
            self.alpha = Uniform(ndim=self.nspots, xmin=0, xmax=np.pi / 4)
        if self.fspot is None:
            self.fspot = Uniform(ndim=self.nspots, xmin=0, xmax=2)
        if self.tmax is None:
            self.tmax = Uniform(ndim=self.nspots, xmin=t[0], xmax=t[-1])
        if self.life is None:
            self.life = Uniform(ndim=self.nspots, xmin=0, xmax=t[-1] - t[0])
        if self.ingress is None:
            self.ingress = Uniform(ndim=self.nspots, xmin=0, xmax=t[-1] - t[0])
        if self.egress is None:
            self.egress = Uniform(ndim=self.nspots, xmin=0, xmax=t[-1] - t[0])

        # validate inst params
        self.mmax = np.size(self.tstart)
        if self.U is None:
            self.U = Uniform(ndim=self.mmax, xmin=.9, xmax=1.1)
        if self.B is None:
            self.B = Uniform(ndim=self.mmax, xmin=.9, xmax=1.1)

        if self.tstart is None:
            self.tstart = np.array([self.t[0] - .01])
        if self.tend is None:
            self.tend = np.array([self.t[-1] + .01])

        # list of fitted variable names and dictionary of fixed parameter values
        self.fixed_params = {}
        self.fit_names = []
        for key, val in self.parameters.items():
            if isinstance(val, Dirac):
                self.fixed_params[key] = val
            else:
                self.fit_names.append(key)

        self.ndim = 0
        for v in self.parameters.values():
            self.ndim += v.n_inputs

    @property
    def parameters(self):
        return {**self.star_pars, **self.spot_pars, **self.inst_pars}

    @property
    def star_pars(self):
        return dict(Pvec=self.Pvec, k2=self.k2, k4=self.k4, c=self.c, d=self.d)

    @property
    def spot_pars(self):
        return dict(lon=self.lon, lat=self.lat, alpha=self.alpha, fspot=self.fspot,
                    tmax=self.tmax, life=self.life, ingress=self.ingress, egress=self.egress)

    @property
    def inst_pars(self):
        return dict(U=self.U, B=self.B)

    def sample(self, x):
        x = np.asarray(x)
        theta = []
        assert x.size == self.ndim, "Mismatch in dimension"
        i, j = 0, 0
        for key, val in self.parameters.items():
            j += val.n_inputs
            theta = np.append(theta, val.sample(*x[i:j]))
            i = j
        theta[0] = np.arcsin(theta[0])
        return theta

    def predict(self, t, theta):
        """Calculates the model flux for given parameter values

        Parameters
        ----------
        t: time array
        theta: parameter vector

        Returns
        -------
        yf: model flux
        """
        theta = np.asarray(theta)
        assert theta.size == 12 + self.nspots * 8 + self.mmax * 2, "Parameter vector with wrong size"
        theta_star = theta[:12]
        theta_spot = theta[12:12 + self.nspots * 8]
        theta_inst = theta[12 + self.nspots * 8:]
        yf = macula(t, theta_star, theta_spot, theta_inst, tstart=self.tstart, tend=self.tend)
        return yf

    def chi(self, theta):
        """Chi squared of parameters given a set of observations

        Parameters
        ----------
        theta: parameter vector

        Returns
        -------
        sse: sum of squared errors weighted by observation uncertainties
        """
        yf = self.predict(self.t, theta)
        sse = np.sum(np.square((yf - self.y) / self.dy))
        return sse

    def lnprob(self, theta):
        """Log posterior probability function to be maximized

        Parameters
        ----------
        theta: parameter vector

        Returns
        -------
        log likelihood when parameters are within prior bounds (NINF otherwise)
        """
        # TODO: add prior function to Prior class
        n = self.t.size
        c = - .5 * n * np.log(2 * np.pi) - .5 * np.log(self.dy).sum()
        return c - .5 * self.chi(theta)

    def grad_chi(self, theta):
        """Gradient of log likelihood

        Parameters
        ----------
        theta: parameter vector

        Returns
        -------
        dy: partial derivatives of log likelihood with respect to each parameter
        """
        theta = np.asarray(theta)
        assert theta.size == 12 + self.nspots * 8 + self.mmax * 2, "Parameter vector with wrong size"
        theta_star = theta[:12]
        theta_spot = theta[12:12 + self.nspots * 8]
        theta_inst = theta[12 + self.nspots * 8:]

        result = macula(self.t, theta_star, theta_spot, theta_inst, tstart=self.tstart, tend=self.tend,
                        full_output=True, derivatives=True)
        yf = result[0]
        dy_star = result[1]
        dy_spot = result[2]
        dy_inst = result[3]
        # TODO: review this whole function!
        dic_star = {}
        for i, key in enumerate(self.star_pars.keys()):
            dic_star[key] = dy_star[:, i]
        dic_spot = {}
        for i, key in enumerate(self.spot_pars.keys()):
            dic_spot[key] = dy_spot[:, i]
        dic_inst = {}
        for i, key in enumerate(self.inst_pars.keys()):
            dic_inst[key] = dy_inst[:, i]
        dic_full = {**dic_star, **dic_spot, **dic_inst}
        dy = []
        for key in self.fit_names:
            if key in self.spot_pars.keys():
                for j in range(self.nspots):
                    dy.append(np.sum(2 * (yf - self.y) * dic_full[key][:, j]) / np.std(self.y))
            elif key in self.inst_pars.keys():
                for j in range(self.mmax):
                    dy.append(np.sum(2 * (yf - self.y) * dic_full[key][:, j]) / np.std(self.y))
            else:
                dy.append(np.sum(2 * (yf - self.y) * dic_full[key]) / np.std(self.y))
        return np.array(dy)

    def minimize(self, n=1000):
        """Searches for local minima from a set of random initial points

        Parameters
        ----------
        n: number of initial points

        Returns
        -------
        opts: best fit starting at each point sorted by chi^2
        sses: corresponding chi^2 value for each best fit
        """
        # TODO: review this whole funciton
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        ndim = ranges.size
        theta = self.sample(np.random.rand(n, self.ndim))
        opts, sses = [], []
        for p0 in tqdm(theta):
            results = minimize(fun=self.chi, x0=p0, method='L-BFGS-B', jac=self.grad_chi, bounds=self.bounds)
            opts.append(results.x)
            sses.append(self.chi(results.x))
        mask = np.isfinite(sses)
        sses = np.array(sses)[mask]
        opts = np.array(opts)[mask]
        sorted_ids = np.argsort(sses)
        sses = sses[sorted_ids]
        opts = opts[sorted_ids]
        return opts, sses

    def mcmc(self, theta, nwalkers=60, nsteps=2500, burn=250):
        """Runs Markov Chain Monte Carlo to sample the posterior distribution

        Parameters
        ----------
        theta: starting value of parameter vector
        nwalkers: number of walkers/chains
        nsteps: number of steps in each chain
        burn: number of burn-in steps

        Returns
        -------
        samples: samples from the posterior distribution
        """
        ndim = len(theta)
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)
        p0 = theta + np.random.randn(nwalkers, ndim) * 1e-2 * ranges
        for _ in tqdm(sampler.sample(p0, iterations=nsteps), total=nsteps):
            pass
        samples = sampler.chain[:, burn:, :].reshape(-1, ndim)
        return samples

    def multinest(self, sampling_efficiency=.01, const_efficiency_mode=True, n_live_points=4000, **kwargs):
        """Runs MultiNest to sample the posterior distribution

        Parameters
        ----------
        sampling_efficiency: defines the sampling efficiency (default=0.01)
        const_efficiency_mode: whether to run in constant efficiency mode (default=True)
        n_live_points: number of live points (default=4000)
        **kwargs:
            importance_nested_sampling: whether to activate INS (default=True)
            evidence_tolerance: evidence tolerance factor (default=0.5)
            outputfiles_basename: root for MultiNest output files (default="chains/1-")
            verbose: updates on sampling progress (default=False)

        Returns
        -------
        results: data returned by pymultinest.solve.solve
        """

        def prior(cube):
            return self.sample(cube)

        def logl(cube):
            n = self.t.size
            c = - .5 * n * np.log(2 * np.pi) - .5 * np.log(self.dy).sum()
            return c - .5 * self.chi(cube)

        results = solve(LogLikelihood=logl, Prior=prior, n_dims=self.ndim, sampling_efficiency=sampling_efficiency,
                        const_efficiency_mode=const_efficiency_mode, n_live_points=n_live_points, **kwargs)
        return results


class CheetahModeler(object):
    def __init__(self, x, y, dy=None, l1=0.68, l2=0.00, ir=0.22, rmin=0.5, rmax=1.5, inc_min=0, inc_max=90,
                 Peq_min=1, Peq_max=50, k_min=-0.6, k_max=0.6, lat_min=-90, lat_max=90, lon_min=0, lon_max=360,
                 rad_min=5, rad_max=30, n_spots=2, n_iter=20, n_clusters=30, burn=100, n_walkers=120, n_steps=1000,
                 v_min=0, v_max=10, threshratio=2, n_temps=1, thin=1, n_spaced=3, savefile=None):
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
        Peq_min, Peq_max: equatorial rotation period bounds (day)
        k_min, k_max: differential rotation coefficient bounds
        lat_min, lat_max: spot latitude bounds (deg)
        lon_min, lon_max: spot longitude bounds (deg)
        rad_min, rad_max: spot radius bounds (deg)
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
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.dy = np.asarray(dy)
        self.rmin = rmin
        self.rmax = rmax
        self.inc_min = inc_min
        self.inc_max = inc_max
        self.Peq_min = Peq_min
        self.Peq_max = Peq_max
        self.k_min = k_min
        self.k_max = k_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.rad_min = rad_min
        self.rad_max = rad_max
        self.n_spots = n_spots
        self.n_iter = n_iter
        self.n_clusters = n_clusters
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

        if self.dy is None:
            self.dy = np.ones_like(self.y)

        self.n_dim = 3 + 3 * n_spots
        self.yf = []
        self.bestps = np.array([])
        self.samples = np.array([])

    def solve(self, theta):
        """Models any number of spots

        Parameters
        ----------
        theta:	star params (inc, Peq, k) + spot params (lat, lon, rad) for each spot

        Returns
        -------
        y: sum of flux contributions from each spot
        """
        ndim = len(theta)
        nspots = int((ndim - 3) / 3)
        y = np.ones_like(self.x)
        for i in range(1, nspots + 1):
            y += -1 + eker(self.x, np.append(theta[:3], theta[3 * i:3 * (i + 1)]), l1=self.l1, l2=self.l2, ir=self.ir)
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
        self.dy = self.dy[good]
        # maximum number of samples is 500 for faster computation
        binsize = np.size(self.x) // n_bins
        if binsize > 1:
            n_bins = np.size(self.x) // binsize
            indexes = np.array_split(np.arange(len(self.x)), n_bins)
            self.x = np.array([np.mean(self.x[a]) for a in indexes])
            self.y = np.array([np.mean(self.y[a]) for a in indexes])
            self.dy = np.array([np.sqrt(np.nansum(self.dy[a] ** 2)) for a in indexes]) / binsize
        # normalize flux
        self.dy /= np.max(self.y)
        self.y /= np.max(self.y)

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
        theta:	star params (inc, Peq, k) + spot params (lat, lon, rad) for each spot

        Returns
        -------
        -inf if out of defined limits (0%)
        0.0 otherwise (100%)
        """
        ndim = len(theta)
        nspots = int((ndim - 3) / 3)
        inc, Peq, k = theta[:3]
        v = self.vsini(inc, Peq)
        # restrict v sin i
        if self.v_max < v[0] or v[1] < self.v_min:
            return -np.inf
        # restrict star params
        if not (self.inc_min < inc < self.inc_max and self.Peq_min < Peq < self.Peq_max and
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
        return np.sum(np.square(error / self.dy))

    def lnprob(self, theta):
        """log of posterior probability of theta (prior * likelihood)
        """
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        n = self.x.size
        c = - .5 * n * np.log(2 * np.pi) - .5 * np.log(self.dy).sum()
        return c - .5 * self.chi(theta)

    def eps(self, theta, star_params=None):
        """Returns array of residuals between light curve and fit (unless lnprior == -inf)
        """
        if star_params is None:
            star_params = []
        theta = np.append(star_params, theta)
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return np.ones_like(self.y) * 1e90
        return (self.y - self.solve(theta)) / self.dy

    def llsq(self, p0s, n_iter=0, star_params=None):
        """Runs Levenberg-Marquardt algorithm for each initial point p0

        Parameters
        ----------
        p0s: array of initial guesses
        n_iter: maximum number of function evaluations for the L-M algorithm
        star_params: (inc, Peq, k), required for multiple spots fitting (fixed star)

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
            fps = leastsq(func=self.eps, x0=p1, args=star_params, maxfev=n_iter)[0]
            if not any(np.isnan(fps)):
                opts.append(np.append(star_params, fps))
                sses.append(self.chi(fps, star_params))
        return opts, sses

    def singlefit(self, p0s, star_params=None):
        """Fits one single spot to current light curve

        Parameters
        ----------
        p0s: array of initial guesses
        star_params: (inc, Peq, k)

        Returns
        -------
        bestps: array of best fits with chi <= threshold (limits to 6 for faster computation)
        """
        if star_params is None:
            star_params = []
        # initial fit with few (n_iter) iterations
        opts, sses = self.llsq(p0s, self.n_iter, star_params=star_params)
        # sort fits with respect to chi
        mask = np.isfinite(sses)
        sses = np.asarray(sses)[mask]
        opts = np.asarray(opts)[mask]
        sorted_ids = np.argsort(sses)
        opts = opts[sorted_ids]
        # let all parameters have same variance (enable clustering)
        optsmat = whiten(np.asarray(opts))
        # find (n_clusters) centroids using kmeans
        __, label = kmeans2(optsmat, self.n_clusters, iter=20, minit='points')
        label = list(label)
        # new corresponding initial points
        p0s = [opts[label.index(i)] for i in range(self.n_clusters) if i in label]
        # final fit with full iterations
        opts, sses = self.llsq(p0s, star_params=star_params)
        # sort fits with respect to chi
        mask = np.isfinite(sses)
        sses = np.asarray(sses)[mask]
        opts = np.asarray(opts)[mask]
        sorted_ids = np.argsort(sses)
        sses = sses[sorted_ids]
        opts = opts[sorted_ids]
        threshold = sses[0] * self.threshratio
        # fits with chi <= given threshold
        bestps = np.asarray(opts)[np.asarray(sses) <= threshold]
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
            mask = np.isfinite(sses)
            sses = np.asarray(sses)[mask]
            opts = np.asarray(opts)[mask]
            sorted_ids = np.argsort(sses)
            opts = opts[sorted_ids]
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
        mins = np.array([self.inc_min, self.Peq_min, self.k_min, self.lat_min, self.lon_min, self.rad_min])
        maxs = np.array([self.inc_max, self.Peq_max, self.k_max, self.lat_max, self.lon_max, self.rad_max])
        if method == 'random':
            m = np.ones(self.n_spaced ** 6, dtype=bool)
            nbad = m.sum()
            q = np.inf * np.ones((self.n_spaced ** 6, 6))
            while nbad > 0:
                p0s = np.random.rand(self.n_spaced ** 6, 6)
                q[m] = p0s[m] * (maxs - mins) + mins
                lp = np.array([self.lnprior(p) for p in q])
                m = ~np.isfinite(lp)
                nbad = m.sum()
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
        if self.savefile is not None:
            f = open(self.savefile, "w")
            f.close()
        for result in tqdm(sampler.sample(p0s, iterations=self.n_steps, thin=self.thin), total=self.n_steps):
            if self.savefile is not None:
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
        if self.savefile is not None:
            f = open(self.savefile, "w")
            f.close()
        for result in tqdm(sampler.sample(p0s, iterations=self.n_steps, thin=self.thin), total=self.n_steps):
            if self.savefile is not None:
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
            p0s = [result[0] + .1 * np.random.randn(self.n_dim) for _ in range(self.n_walkers)]
            sampler = self.mcmc(p0s)
            chain = sampler.chain
        else:
            # initialize walkers in n balls around best fits
            p0s = [[result[j] + .1 * np.random.randn(self.n_dim) for _ in range(self.n_walkers)] for j in range(n)]
            sampler = self.ptmcmc(p0s)
            chain = sampler.chain[0]
        # cut burn-in period
        self.samples = chain[:, self.burn:, :].reshape((-1, self.n_dim))
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
