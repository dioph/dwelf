import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from _macula import maculamod


def eker(t, theta, l1=.68, l2=0., ir=.22):
    """Analytic model of one circular spot based on equations in Eker (1994, ApJ, 420, 373)

    Parameters
    ----------
    t: time array
    theta:	star params (inc, Peq, k) + spot params (lat, lon, rad)
    l1: linear limb-darkening coefficient
    l2: quadratic limb-darkening coefficient
    ir: spot-to-photosphere intensity ratio

    Returns
    -------
    y: flux for given spot configuration at corresponding time samples
    """
    limb1, limb2, iratio = l1, l2, ir
    inc_deg, Peq, k, lat_deg, lon_deg, rad_deg = theta
    # convert angles from degrees to radians
    inc = np.radians(inc_deg)
    lam = np.radians(lon_deg)
    bet = np.radians(lat_deg)
    rad = np.radians(rad_deg)
    period = Peq / (1 - k * np.sin(bet) ** 2)
    # useful scalar quantities
    cosrad = np.cos(rad)
    sinrad = np.sin(rad)
    # rotational phases
    phase = t / period
    phi = 2.0 * np.pi * phase
    nphi = len(phi)
    # angle the0 between two vectors originating from spot center:
    # 1) normal to stellar surface, directed away from center of star
    # 2) directed towards the observer
    costhe0 = np.cos(inc) * np.sin(bet) + np.sin(inc) * np.cos(bet) * np.cos(phi - lam)
    sinthe0 = np.sqrt(1.0 - costhe0 ** 2)
    the0 = np.arccos(costhe0)
    # find phases when spot is full, gibbous, crescent or occulted
    jf = np.flatnonzero(the0 <= np.pi / 2 - rad)
    nf = len(jf)
    jg = np.flatnonzero(np.logical_and(the0 > np.pi / 2 - rad, the0 <= np.pi / 2))
    ng = len(jg)
    jc = np.flatnonzero(np.logical_and(the0 > np.pi / 2, the0 <= np.pi / 2 + rad))
    nc = len(jc)
    jo = np.flatnonzero(the0 > np.pi / 2 + rad)
    no = len(jo)
    # allocate arrays for integrals
    ic = np.zeros(nphi)  # constant intensity term
    il = np.zeros(nphi)  # linear intensity term
    iq = np.zeros(nphi)  # quadratic intensity term
    #
    # FULL (entirely visible)
    #
    if nf >= 1:
        costhe0_f = costhe0[jf]
        sinthe0_f = sinthe0[jf]
        ic[jf] = np.pi * np.sin(rad) ** 2 * costhe0_f
        il[jf] = 2 * np.pi / 3 * (1 - cosrad ** 3) - np.pi * cosrad * sinrad ** 2 * sinthe0_f ** 2
        iq[jf] = np.pi / 2 * (1 - cosrad ** 4) * costhe0_f ** 3 + \
            3 * np.pi / 4 * sinrad ** 4 * costhe0_f * sinthe0_f ** 2
    #
    # GIBBOUS (more than half visible)
    #
    if ng >= 1:
        the0_g = the0[jg]
        costhe0_g = costhe0[jg]
        sinthe0_g = sinthe0[jg]
        cosphi0_g = - 1.0 / (np.tan(the0_g) * np.tan(rad))
        rad0_g = abs(the0_g - np.pi / 2)
        phi0_g = np.arccos(cosphi0_g)
        sinphi0_g = np.sqrt(1.0 - cosphi0_g ** 2)
        cosrad0_g = np.cos(rad0_g)
        sinrad0_g = np.sin(rad0_g)
        k1_g = ((np.pi - phi0_g) / 4) * (cosrad0_g ** 4 - cosrad ** 4)
        k2_g = (sinphi0_g / 8) * (rad0_g - rad + 0.5 * (np.sin(2 * rad) * np.cos(2 * rad) -
                                                        np.sin(2 * rad0_g) * np.cos(2 * rad0_g)))
        k3_g = (1.0 / 8) * (np.pi - phi0_g - sinphi0_g * cosphi0_g) * (sinrad ** 4 - sinrad0_g ** 4)
        k4_g = - (sinphi0_g - sinphi0_g ** 3 / 3) * ((3.0 / 8) * (rad - rad0_g) + (1.0 / 16) *
                                                     (np.sin(2 * rad) * (np.cos(2 * rad) - 4) -
                                                      np.sin(2 * rad0_g) * (np.cos(2 * rad0_g) - 4)))
        cl_g = ((np.pi - phi0_g) / 3) * (cosrad ** 3 - cosrad0_g ** 3) * (1 - 3 * costhe0_g ** 2) - \
               (np.pi - phi0_g - sinphi0_g * cosphi0_g) * (cosrad - cosrad0_g) * sinthe0_g ** 2 - \
               (4.0 / 3) * sinphi0_g * (sinrad ** 3 - sinrad0_g ** 3) * sinthe0_g * costhe0_g - \
               (1.0 / 3) * sinphi0_g * cosphi0_g * (cosrad ** 3 - cosrad0_g ** 3) * sinthe0_g ** 2
        cq_g = 2 * costhe0_g ** 3 * k1_g + 6 * costhe0_g ** 2 * sinthe0_g * k2_g + \
            6 * costhe0_g * sinthe0_g ** 2 * k3_g + 2 * sinthe0_g ** 3 * k4_g
        ic[jg] = phi0_g * costhe0_g * sinrad ** 2 - np.arcsin(cosrad / sinthe0_g) - \
            0.5 * sinthe0_g * sinphi0_g * np.sin(2 * rad) + np.pi / 2
        il[jg] = 2 * np.pi / 3 * (1 - cosrad ** 3) - np.pi * cosrad * sinrad ** 2 * sinthe0_g ** 2 - cl_g
        iq[jg] = np.pi / 2 * (1 - cosrad ** 4) * costhe0_g ** 3 + 3 * np.pi / 4 * \
            sinrad ** 4 * costhe0_g * sinthe0_g ** 2 - cq_g
    #
    # CRESCENT (less than half visible)
    #
    if nc >= 1:
        the0_c = the0[jc]
        costhe0_c = costhe0[jc]
        sinthe0_c = sinthe0[jc]
        cosphi0_c = - 1.0 / (np.tan(the0_c) * np.tan(rad))
        rad0_c = abs(the0_c - np.pi / 2)
        phi0_c = np.arccos(cosphi0_c)
        sinphi0_c = np.sqrt(1.0 - cosphi0_c ** 2)
        cosrad0_c = np.cos(rad0_c)
        sinrad0_c = np.sin(rad0_c)
        k1_c = (phi0_c / 4) * (cosrad0_c ** 4 - cosrad ** 4)
        k2_c = - (sinphi0_c / 8) * (rad0_c - rad + 0.5 * (np.sin(2 * rad) * np.cos(2 * rad) -
                                                          np.sin(2 * rad0_c) * np.cos(2 * rad0_c)))
        k3_c = (1.0 / 8) * (phi0_c + sinphi0_c * cosphi0_c) * (sinrad ** 4 - sinrad0_c ** 4)
        k4_c = (sinphi0_c - sinphi0_c ** 3 / 3) * ((3.0 / 8) * (rad - rad0_c) + (1.0 / 16) *
                                                   (np.sin(2 * rad) * (np.cos(2 * rad) - 4) -
                                                    np.sin(2 * rad0_c) * (np.cos(2 * rad0_c) - 4)))
        cq_c = 2 * costhe0_c ** 3 * k1_c + 6 * costhe0_c ** 2 * sinthe0_c * k2_c + \
            6 * costhe0_c * sinthe0_c ** 2 * k3_c + 2 * sinthe0_c ** 3 * k4_c
        ic[jc] = phi0_c * costhe0_c * sinrad ** 2 - np.arcsin(cosrad / sinthe0_c) - \
            0.5 * sinthe0_c * sinphi0_c * np.sin(2 * rad) + np.pi / 2
        il[jc] = (phi0_c / 3) * (cosrad ** 3 - cosrad0_c ** 3) * (1 - 3 * costhe0_c ** 2) - \
                 (phi0_c + sinphi0_c * cosphi0_c) * (cosrad - cosrad0_c) * sinthe0_c ** 2 + \
                 (4.0 / 3) * sinphi0_c * (sinrad ** 3 - sinrad0_c ** 3) * sinthe0_c * costhe0_c + \
                 (1.0 / 3) * sinphi0_c * cosphi0_c * (cosrad ** 3 - cosrad0_c ** 3) * sinthe0_c ** 2
        iq[jc] = cq_c
    #
    # OCCULTED (back of the star)
    #
    if no >= 1:
        ic[jo] = 0.0
        il[jo] = 0.0
        iq[jo] = 0.0
    # calculate light curve (equation 12c from Eker, 1994)
    y = 1.0 + (iratio - 1.0) / (np.pi * (1.0 - limb1 / 3.0 + limb2 / 6.0)) * \
        ((1.0 - limb1 + limb2) * ic + (limb1 - 2.0 * limb2) * il + limb2 * iq)
    return y


def plot_mcmc(samples, labels=None, priors=None, ptrue=None, precision=None, nbins=30, s=1.0):
    """Plots a Giant Triangle Confusogram

    Parameters
    ----------
    samples: 2-D array, shape (N, ndim)
        Samples from ndim variables to be plotted in the GTC
    labels: list of strings, optional
        List of names for each variable (size ndim)
    priors: list of callables, optional
        List of prior functions for the variables distributions (size ndim)
    ptrue: list of floats, optional
        List of true estimates for each parameter
    precision: list of ints, optional
        List of decimal places to write down for each parameter. Defaults to 2
    nbins: int, optional
        Number of bins to be used in 1D and 2D histograms. Defaults to 30
    s: float, optional
        Standard deviation of Gaussian filter applied to smooth histograms. Defaults to 1.0
    """
    p = map(lambda v: (v[1], v[1] - v[0], v[2] - v[1]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    p = list(p)
    ndim = samples.shape[-1]
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # TODO: pyplot style context
    grid = plt.GridSpec(ndim, ndim, wspace=0.0, hspace=0.0)
    handles = []

    if precision is None:
        precision = 2 * np.ones(ndim, dtype=int)
    if ptrue is None:
        ptrue = np.array([None for _ in range(ndim)])

    # PLOT 1D
    for i in range(ndim):
        ax = fig.add_subplot(grid[i, i])
        H, edges = np.histogram(samples[:, i], bins=nbins, density=True)
        centers = (edges[1:] + edges[:-1]) / 2
        data = ndimage.gaussian_filter1d((centers, H), sigma=s)
        data[1] /= data[1].sum()
        l1, = ax.plot(data[0], data[1], 'b-', lw=1, label='posterior')
        if priors is not None:
            pr = priors[i](centers)
            pr /= pr.sum()
            l2, = ax.plot(centers, pr, 'k-', lw=1, label='prior')
        l3 = ax.axvline(p[i][0], color='k', ls='--', label='median')
        mask = np.logical_and(centers - p[i][0] <= p[i][2], p[i][0] - centers <= p[i][1])
        ax.fill_between(centers[mask], np.zeros(mask.sum()), data[1][mask], color='b', alpha=0.3)
        if ptrue[i] is not None:
            l4 = ax.axvline(ptrue[i], color='gray', lw=1.5, label='true')
        if i < ndim - 1:
            ax.set_xticks([])
        else:
            ax.tick_params(rotation=45)
        ax.set_yticks([])
        ax.set_ylim(0)
        if labels is not None:
            ax.set_title('{0} = {1:.{4}f}$^{{+{2:.{4}f}}}_{{-{3:.{4}f}}}$'.format(labels[i], p[i][0], p[i][2], p[i][1],
                                                                                  precision[i]))

    handles.append(l1)
    try:
        handles.append(l2)
    except UnboundLocalError:
        pass
    try:
        handles.append(l3)
    except UnboundLocalError:
        pass
    try:
        handles.append(l4)
    except UnboundLocalError:
        pass

    # PLOT 2D
    nbins_flat = np.linspace(0, nbins ** 2, nbins ** 2)
    for i in range(ndim):
        for j in range(i):
            ax = fig.add_subplot(grid[i, j])
            H, xi, yi = np.histogram2d(samples[:, j], samples[:, i], bins=nbins)
            extents = [xi[0], xi[-1], yi[0], yi[-1]]
            H /= H.sum()
            H_order = np.sort(H.flat)
            H_cumul = np.cumsum(H_order)
            tmp = np.interp([.0455, .3173, 1.0], H_cumul, nbins_flat)
            chainlevels = np.interp(tmp, nbins_flat, H_order)
            data = ndimage.gaussian_filter(H.T, sigma=s)
            xbins = (xi[1:] + xi[:-1]) / 2
            ybins = (yi[1:] + yi[:-1]) / 2
            ax.contourf(xbins, ybins, data, levels=chainlevels, colors=['#1f77b4', '#52aae7', '#85ddff'], alpha=0.3)
            ax.contour(data, chainlevels, extent=extents, colors='b')
            if ptrue[i] is not None:
                ax.axhline(ptrue[i], color='gray', lw=1.5)
            if ptrue[j] is not None:
                ax.axvline(ptrue[j], color='gray', lw=1.5)
            if i < ndim - 1:
                ax.set_xticks([])
            else:
                ax.tick_params(rotation=45)
            if j > 0:
                ax.set_yticks([])
            else:
                ax.tick_params(rotation=45)
    fig.legend(handles=handles)
    return fig


def macula(t, theta_star, theta_spot, theta_inst, derivatives=False, temporal=False, tdeltav=False,
           full_output=False, tstart=None, tend=None):
    """Wrapper for macula FORTRAN routine.

    Parameters
    ----------
    theta_star: array_like
        Array of 12 parameters describing star:
        ! ------------------------------------------------------------------------------
        ! Theta_star(j) = Parameter vector for the star's intrinsic parameters
        ! ------------------------------------------------------------------------------
        ! Istar 	= Theta_star(1)		! Inclination of the star [rads]
        ! Peq 		= Theta_star(2)		! Rot'n period of the star's equator [d]
        ! kappa2 	= Theta_star(3)		! Quadratic differential rotation coeff
        ! kappa4 	= Theta_star(4)		! Quartic differential rotation coeff
        ! c1 		= Theta_star(5)		! 1st of four-coeff stellar LD terms
        ! c2 		= Theta_star(6)		! 2nd of four-coeff stellar LD terms
        ! c3 		= Theta_star(7)		! 3rd of four-coeff stellar LD terms
        ! c4 		= Theta_star(8)		! 4th of four-coeff stellar LD terms
        ! d1 		= Theta_star(9)		! 1st of four-coeff spot LD terms
        ! d2	 	= Theta_star(10)	! 2nd of four-coeff spot LD terms
        ! d3	 	= Theta_star(11)	! 3rd of four-coeff spot LD terms
        ! d4	 	= Theta_star(12)	! 4th of four-coeff spot LD terms

    theta_spot: array_like
        Array of spot parameters, shape (8, Nspot)
        ! ------------------------------------------------------------------------------
        ! Theta_spot(j,k) = Parameters of the k^th spot
        ! ------------------------------------------------------------------------------
        ! Lambda0(k) 	= Theta_spot(1,k)	! Longitude of spot at time tref(k)
        ! Phi0(k) 	= Theta_spot(2,k)	! Latitude of spot at time tref(k)
        ! alphamax(k)	= Theta_spot(3,k)	! Angular spot size at time tmax(k)
        ! fspot(k)	= Theta_spot(4,k)	! Spot-to-star flux contrast of spot k
        ! tmax(k)	= Theta_spot(5,k)	! Time at which spot k is largest
        ! life(k)	= Theta_spot(6,k)	! Lifetime of spot k (FWFM) [days]
        ! ingress(k)	= Theta_spot(7,k)	! Ingress duration of spot k [days]
        ! egress(k)	= Theta_spot(8,k)	! Egress duration of spot k  [days]

    theta_inst: array_like
        Nuisance/instrumental parameters for each of 'm' data sets.
        ! ------------------------------------------------------------------------------
        ! Theta_inst(j,m) = Instrumental/nuisance parameters
        ! ------------------------------------------------------------------------------
        ! U(m) 		= Theta_inst(1,m)	! Baseline flux level for m^th data set
        ! B(m) 		= Theta_inst(2,m)	! Blend factor for m^th data set

    derivatives: bool (optional)
        Whether to calculate derivatives.
    temporal: bool (optional)
        Whether to calculate temporal derivatives
    tdeltav: bool (optional)
        Whether to calculate transit depth variations
    full_output: bool (optional)
        If True, then return all output; otherwise just return model flux.
    tstart, tend: array-like (optional)
        Time stamp at start/end of each of 'm' data sets
    """

    if tstart is None:
        tstart = t[0] - .01
    if tend is None:
        tend = t[-1] + .01
    res = maculamod.macula(t, derivatives, temporal, tdeltav, theta_star, theta_spot, theta_inst, tstart, tend)
    if full_output:
        return res
    else:
        return res[0]


def triangulate(poly):
    triangles = []
    for i in range(poly.shape[0]-2):
        triangles.append(np.array([poly[i], poly[i+1], poly[-1]]))
    return np.array(triangles)


def triangle_area(triang):
    A, B, C = triang
    area = np.abs(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))/2
    return area


def change_coord(q1, q2, triang):
    A, B, C = triang
    v = (1 - np.sqrt(q1)) * A + np.sqrt(q1) * (1 - q2) * B + q2 * np.sqrt(q1) * C
    return v


def quadratic_limb(q1, q2):
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return u1, u2


def cubic_limb(q1, q2, q3):
    c2 = (q1 ** (1 / 3) / 12) * (28 * (9 - 5 * np.sqrt(2)) + 3 * np.sqrt(q2) *
                                 (-6 * np.cos(2 * np.pi * q3) + (3 + 10 * np.sqrt(2) * np.sin(2 * np.pi * q3))))
    c3 = (q1 ** (1 / 3) / 9) * (-632 + 396 * np.sqrt(q2)
                                + 3 * np.sqrt(q2) * (4 - 21 * np.sqrt(2)) * np.sin(2 * np.pi * q3))
    c4 = (q1 ** (1 / 3) / 12) * (28 * (9 - 5 * np.sqrt(2)) + 3 * np.sqrt(q2) *
                                 (6 * np.cos(2 * np.pi * q3) + (3 + 10 * np.sqrt(2) * np.sin(2 * np.pi * q3))))
    return c2, c3, c4


def polygon_intersection(poly1, poly2):
    poly = []
    for p in poly1:
        if is_inside(p, poly2):
            add_point(poly, p)
    for p in poly2:
        if is_inside(p, poly1):
            add_point(poly, p)
    for i in range(poly1.shape[0]):
        if i == poly1.shape[0] - 1:
            nxt = 0
        else:
            nxt = i + 1
        points = get_intersection_points(poly1[i], poly1[nxt], poly2)
        for p in points:
            add_point(poly, p)
    return sort_clockwise(np.array(poly))


def get_intersection_point(l1p1, l1p2, l2p1, l2p2):
    a1 = l1p2[1] - l1p1[1]
    b1 = l1p1[0] - l1p2[0]
    c1 = a1 * l1p1[0] + b1 * l1p1[1]
    a2 = l2p2[1] - l2p1[1]
    b2 = l2p1[0] - l2p2[0]
    c2 = a2 * l2p1[0] + b2 * l2p1[1]
    det = a1 * b2 - a2 * b1
    if np.abs(det) < 1e-9:
        return np.nan
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    online1 = ((min(l1p1[0], l1p2[0]) < x) or (np.abs(min(l1p1[0], l1p2[0]) - x) < 1e-9)) \
        and ((max(l1p1[0], l1p2[0]) > x) or (np.abs(max(l1p1[0], l1p2[0]) - x) < 1e-9)) \
        and ((min(l1p1[1], l1p2[1]) < y) or (np.abs(min(l1p1[1], l1p2[1]) - y) < 1e-9)) \
        and ((max(l1p1[1], l1p2[1]) > y) or (np.abs(max(l1p1[1], l1p2[1]) - y) < 1e-9))
    online2 = ((min(l2p1[0], l2p2[0]) < x) or (np.abs(min(l2p1[0], l2p2[0]) - x) < 1e-9)) \
        and ((max(l2p1[0], l2p2[0]) > x) or (np.abs(max(l2p1[0], l2p2[0]) - x) < 1e-9)) \
        and ((min(l2p1[1], l2p2[1]) < y) or (np.abs(min(l2p1[1], l2p2[1]) - y) < 1e-9)) \
        and ((max(l2p1[1], l2p2[1]) > y) or (np.abs(max(l2p1[1], l2p2[1]) - y) < 1e-9))
    if online1 and online2:
        return np.array([x, y])

    return np.nan


def get_intersection_points(p1, p2, poly):
    intersection = []
    for i in range(poly.shape[0]):
        if i == poly.shape[0] - 1:
            nxt = 0
        else:
            nxt = i + 1
        ip = get_intersection_point(p1, p2, poly[i], poly[nxt])
        if np.all(np.isfinite(ip)):
            add_point(intersection, ip)
    return np.array(intersection)


def add_point(l, p):
    for q in l:
        if np.abs(q[0] - p[0]) < 1e-9 and np.abs(q[1] - p[1]) < 1e-9:
            break
    else:
        l.append(p)


def is_inside(p, poly):
    i = 0
    j = poly.shape[0] - 1
    res = False
    for i in range(poly.shape[0]):
        if (poly[i, 1] > p[1]) != (poly[j, 1] > p[1]) \
          and p[0] < (poly[j, 0] - poly[i, 0]) * (p[1] - poly[i, 1]) / (poly[j, 1] - poly[i, 1]) + poly[i, 0]:
            res = not res
        j = i
        i = i + 1
    return res


def sort_clockwise(points):
    center = np.mean(points, axis=0)
    new_points = sorted(points, key=lambda p: -np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(new_points)
