import numpy as np
import matplotlib.pyplot as plt


def eker(t, theta, l1=.68, l2=0., ir=.22):
    """Analytic model of one circular spot based on equations in Eker (1994, ApJ, 420, 373)

    Parameters
    ----------
    t: time array
    theta:	star params (inc, Teq, k) + spot params (lat, lon, rad)
    l1: linear limb-darkening coefficient
    l2: quadratic limb-darkening coefficient
    ir: spot-to-photosphere intensity ratio

    Returns
    -------
    y: flux for given spot configuration at corresponding time samples
    """
    limb1, limb2, iratio = l1, l2, ir
    inc_deg, Teq, k, lat_deg, lon_deg, rad_deg = theta
    # convert angles from degrees to radians
    inc = np.radians(inc_deg)
    lam = np.radians(lon_deg)
    bet = np.radians(lat_deg)
    rad = np.radians(rad_deg)
    period = Teq / (1 - k * np.sin(bet) ** 2)
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

