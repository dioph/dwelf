import numpy as np
from scipy.stats import norm

from dwelf.utils import triangulate, triangle_area, polygon_intersection


class Prior(object):
    def __init__(self):
        self.n_inputs = 1
        self.last_sampled = None

    def sample(self, *args):
        self.last_sampled = args
        return args


class SameAs(Prior):
    def __init__(self, prior):
        super(SameAs, self).__init__()
        self.n_inputs = 0
        self.prior = prior

    def sample(self, *args):
        self.last_sampled = self.prior.last_sampled
        return self.last_sampled


# UNIVARIATE DISTRIBUTIONS


class Dirac(Prior):
    def __init__(self, x0):
        super(Dirac, self).__init__()
        self.n_inputs = 0
        self.x0 = x0

    def sample(self, *args):
        self.last_sampled = self.x0
        return self.x0


class Uniform(Prior):
    def __init__(self, xmin=0., xmax=1., ndim=1):
        super(Uniform, self).__init__()
        self.n_inputs = ndim
        self.xmin = np.asarray(xmin)
        self.xmax = np.asarray(xmax)

    def sample(self, *args):
        x = np.asarray(args)
        self.last_sampled = x * (self.xmax - self.xmin) + self.xmin
        return self.last_sampled


class SineUniform(Prior):
    def __init__(self, sinxmin=0., sinxmax=1., ndim=1):
        super(SineUniform, self).__init__()
        self.n_inputs = ndim
        self.sinxmin = np.asarray(sinxmin)
        self.sinxmax = np.asarray(sinxmax)

    def sample(self, *args):
        sinx = np.asarray(args)
        self.last_sampled = np.arcsin(sinx * (self.sinxmax - self.sinxmin) + self.sinxmin)
        return self.last_sampled


class LogUniform(Prior):
    def __init__(self, logxmin=0., logxmax=1., ndim=1):
        super(LogUniform, self).__init__()
        self.n_inputs = ndim
        self.logxmin = np.asarray(logxmin)
        self.logxmax = np.asarray(logxmax)

    def sample(self, *args):
        logx = np.asarray(args)
        self.last_sampled = np.exp(logx * (self.logxmax - self.logxmin) + self.logxmin)
        return self.last_sampled


class Normal(Prior):
    def __init__(self, mu=0., sd=1., ndim=1):
        super(Normal, self).__init__()
        self.n_inputs = ndim
        self.mu = np.asarray(mu)
        self.sd = np.asarray(sd)

    def sample(self, *args):
        x = np.asarray(args)
        self.last_sampled = norm(self.mu, self.sd).ppf(x)
        return self.last_sampled

class LogNormal(Prior):
    def __init__(self, logmu=0., logsd=1., ndim=1):
        super(LogNormal, self).__init__()
        self.n_inputs = ndim
        self.logmu = np.asarray(logmu)
        self.logsd = np.asarray(logsd)

    def sample(self, *args):
        logx = np.asarray(args)
        self.last_sampled = np.exp(norm(self.logmu, self.logsd).ppf(logx))
        return self.last_sampled

# BIVARIATE DISTRIBUTIONS


class Triangular(Prior):
    def __init__(self, triangle):
        super(Triangular, self).__init__()
        self.n_inputs = 2
        self.triangle = triangle

    def sample(self, *args):
        q1, q2 = args
        A, B, C = self.triangle
        self.last_sampled = (1 - np.sqrt(q1)) * A + np.sqrt(q1) * (1 - q2) * B + q2 * np.sqrt(q1) * C
        return self.last_sampled


class Polygon(Prior):
    def __init__(self, poly):
        super(Polygon, self).__init__()
        self.n_inputs = 2
        self.triangles = triangulate(poly)
        area = np.sum([triangle_area(t) for t in self.triangles])
        self.relative_area = np.array([triangle_area(t) / area for t in self.triangles])
        area_limits = np.cumsum(self.relative_area)
        self.area_limits = np.append(0, area_limits)

    def sample(self, *args):
        q1, q2 = args
        group = np.searchsorted(self.area_limits, q2) - 1
        triangle = self.triangles[group]
        q2 = (q2 - self.area_limits[group]) / self.relative_area[group]
        self.last_sampled = Triangular(triangle).sample(q1, q2)
        return self.last_sampled


# LDC DISTRIBUTIONS


class Quadratic(Prior):
    def __init__(self, amin=0., amax=2., bmin=-1., bmax=1.):
        super(Quadratic, self).__init__()
        self.n_inputs = 2
        poly1 = np.array([[amin, bmin], [amin, bmax], [amax, bmax], [amax, bmin]])
        poly2 = np.array([[0, 0], [0, 1], [2, -1]])
        self.poly = polygon_intersection(poly1, poly2)

    def sample(self, *args):
        q1, q2 = args
        a, b = Polygon(self.poly).sample(q1, q2)
        c1 = 0.
        c2 = a + 2 * b
        c3 = 0.
        c4 = -b
        self.last_sampled = np.array([c1, c2, c3, c4])
        return self.last_sampled


class ThreeParam(Prior):
    def __init__(self):
        super(ThreeParam, self).__init__()
        self.n_inputs = 3

    def sample(self, *args):
        q1, q2, q3 = args
        c1 = 0.
        c2 = (q1 ** (1 / 3) / 12) * (28 * (9 - 5 * np.sqrt(2)) + 3 * np.sqrt(q2) *
                                     (-6 * np.cos(2 * np.pi * q3) + (3 + 10 * np.sqrt(2) * np.sin(2 * np.pi * q3))))
        c3 = (q1 ** (1 / 3) / 9) * (-632 + 396 * np.sqrt(q2)
                                    + 3 * np.sqrt(q2) * (4 - 21 * np.sqrt(2)) * np.sin(2 * np.pi * q3))
        c4 = (q1 ** (1 / 3) / 12) * (28 * (9 - 5 * np.sqrt(2)) + 3 * np.sqrt(q2) *
                                     (6 * np.cos(2 * np.pi * q3) + (3 + 10 * np.sqrt(2) * np.sin(2 * np.pi * q3))))
        self.last_sampled = np.array([c1, c2, c3, c4])
        return self.last_sampled
