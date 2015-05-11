import numpy
import collections
from copy import deepcopy
import scipy.stats
from scipy.stats import wishart ## NOTE: Requires scipy 0.16 development version
from numpy.linalg import inv

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Individual distribution stochastics
# Each of these must propose and store.
# Here c, cx, cy store the info needed to recover the matrix
# the cov is stored as cov, and is the inverse of what the wishart makes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CircularNormal2D(object):
    """ Circular covariance """
    def __init__(self, c=None):

        self.df = 1

        if c is None: self.c = wishart.rvs(self.df, numpy.eye(self.df))  # let's draw from a 1-D wishart for this
        else:         self.c = c

        self.cov = inv(numpy.diag([self.c, self.c]))

    def compute_prior(self):
        return wishart.logpdf(self.c, self.df, numpy.eye(self.df))

    def propose(self):
        val = wishart.rvs(self.df, self.c)
        fb = wishart.logpdf(val, self.df, self.c) - wishart.logpdf(self.c, self.df, val)
        return CircularNormal2D(c=val), fb
        #return type(self)()

class AlignedNormal2D(object):
    """ Aligned to one axis """
    def __init__(self, cx=None, cy=None):

        self.df = 1

        if cx is None:  self.cx = wishart.rvs(self.df, numpy.eye(self.df))  # draw these separately and use a diagonal
        else:           self.cx = cx

        if cy is None:  self.cy = wishart.rvs(self.df, numpy.eye(self.df))
        else:           self.cy = cy

        self.cov = inv(numpy.diag([self.cx,self.cy]))

    def compute_prior(self):
        return wishart.logpdf(self.cx, self.df, numpy.eye(self.df)) + wishart.logpdf(self.cy, self.df, numpy.eye(1))

    def propose(self):
        vx = wishart.rvs(self.df, self.cx)
        vy = wishart.rvs(self.df, self.cy)
        fb =   wishart.logpdf(vx, self.df, self.cx) + wishart.logpdf(vy, self.df, self.cy) - \
             ( wishart.logpdf(self.cx, self.df, vx) + wishart.logpdf(self.cy, self.df, vy) )
        return AlignedNormal2D(cx=vx, cy=vy), fb


class FreeNormal2D(object):
    """ Free wishart distribution """
    def __init__(self, c=None):

        self.df = 2

        if c is None:  self.c = wishart.rvs(self.df, numpy.eye(self.df))
        else:          self.c = c

        self.cov = inv(self.c)

    def compute_prior(self):
        return wishart.logpdf(self.c, self.df, numpy.eye(self.df))

    def propose(self):
        val = wishart.rvs(self.df, self.c)
        fb = wishart.logpdf(val, self.df, self.c) - wishart.logpdf(self.c, self.df, val)
        return FreeNormal2D(c=val), fb


class DirichletSample(object):

    def __init__(self, n, alpha=5.0):
        if alpha is None:
            alpha = numpy.ones(n)*alpha
        elif not isinstance(alpha, collections.Iterable):
            alpha = numpy.array([alpha] * n)

        self.alpha = alpha
        self.n = n
        self.p = numpy.random.dirichlet(alpha)

    def compute_prior(self):
        return scipy.stats.dirichlet.logpdf(self.p, self.alpha)

    def propose(self):
        """This just proposes from the prior  """

        p = type(self)(self.n, alpha=self.alpha)

        return p, p.compute_prior() - self.compute_prior()
