import numpy
from copy import deepcopy
import numpy

import scipy.stats
from scipy.stats import wishart ## NOTE: Requires scipy 0.16 development version
from numpy.linalg import inv

## Matrix rotation
m90 = numpy.matrix([[0,1],[-1,0]]) # the rotation matrix -- 90 degrees
def rot90(m):
    """ 90 degree rotation of m1 """
    return (m90*m)*(m90.transpose())

# Some stochastics
def random_invwishart(n):
    """
        Returns the sample and its log probability under the prior
    """

    df = n # the dimension

    m = wishart.rvs(df, numpy.eye(n))
    lp = wishart.logpdf(m, df, numpy.eye(n))

    if n == 1:
        return 1.0/m, lp
    else:
        return inv(m), lp

def rdirichlet(n, a):
    """ Returns a dirichlet sample and its lp """
    alpha = numpy.array([a]*n)
    x = numpy.random.dirichlet(alpha)
    lp = scipy.stats.dirichlet.logpdf(x, alpha)
    return x, lp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Individual distribution stochastics
# Each of these must propose and store a .lp giving its log probability of having been proposed
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CircularNormal2D(object):
    """ Circular covariance """
    def __init__(self):
        c, lp = random_invwishart(1) # let's draw from a 1-D wishart for this
        self.cov = numpy.diag([c,c])
        self.lp = lp

    def propose(self):
        return type(self)()

class AlignedNormal2D(object):
    """ Aligned to one axis """
    def __init__(self):
        cx, lpx = random_invwishart(1) # draw these separately and use a diagonal
        cy, lpy = random_invwishart(1)
        self.cov = numpy.diag([cx,cy])
        self.lp = lpx + lpy

    def propose(self):
        return type(self)()

class FreeNormal2D(object):
    """ Free wishart distribution """
    def __init__(self):
        c, lp = random_invwishart(2)
        self.cov = c
        self.lp = lp

    def propose(self):
        return type(self)()


class DirichletSample(object):

    def __init__(self, n, alpha=2.0):
        if alpha is None:
            alpha = numpy.ones(n)*alpha
        self.alpha, self.n = alpha, n
        self.p, self.lp = rdirichlet(n, alpha)
    def propose(self):
        return type(self)(self.n, alpha=self.alpha)
