
import numpy
import random
from copy import deepcopy
from numpy import log
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from Distributions import *

origin = numpy.zeros(2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  A few helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def zipa(*args):
    # zip with assertion of equal length
    l0=len(args[0])
    assert all([len(a) == l0 for a in args])

    return zip(*args)

## Matrix rotation
m90 = numpy.matrix([[0,-1],[1,0]]) # the rotation matrix -- 90 degrees
def rot90(m):
    """ 90 degree rotation of m """
    return (m90*m)*(m90.transpose())

# For scaling the wishart
WSD=1.0
def scaleW(W, s1, s2):
    """ Returns VQV  """
    V = numpy.abs(numpy.diag([s1, s2])) # abs makes it a folded normal
    return numpy.dot(numpy.dot(V, W), V)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Superclass for hypotheses
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class H(object):
    """ A class to represent mixture hypotheses using a list of covariance matrices """

    def __init__(self):
        raise NotImplementedError # must be implemented by subclasses

    def get_unscaled_weights_and_covariances(self):
        """ Return a paired list of weights and corresponding covariance matrices. NOT yet scaled by self.independent_components """
        raise NotImplementedError

    def compute_prior(self):
        """ For now, we'll use our proposal probability. """
        return sum([x.compute_prior() for x in self.independent_components])

    def compute_posterior(self, data):
        try:
            self.posterior_score = self.compute_prior() + self.compute_likelihood(data)

        except numpy.linalg.linalg.LinAlgError:
            self.posterior_score = float("-inf")

        return self.posterior_score

    def propose(self):
        """
            Return a proposal, forward-back logp !
            To propose to a hypothesis, propose to one of its components and keep the rest fixed
         """

        # copy myself
        mynew = deepcopy(self)

        # choose something to propose to
        i = random.randint(0, len(self.independent_components)-1)

        # For now, we'll propose from everything's prior
        mynew.independent_components[i], fb = self.independent_components[i].propose() # propose

        # Return the new guy and my fb
        return mynew, fb


    def compute_likelihood(self, data):
        ps, covs = zip(*self.get_weights_and_covariances())

        # get the log prob under each covariance matrix
        lps = map(lambda c: multivariate_normal.logpdf(data, mean=origin, cov=c), covs)

        ## TODO: DOUBLE CHECK THIS:

        return sum(logsumexp([lp + log(p) for p, lp in zip(ps, lps)], axis=0))

    def sample(self, N=100):

        ps, covs = zip(*self.get_weights_and_covariances())
        # figure out how to divide our counts between p
        ns = numpy.random.multinomial(N, ps)

        s = numpy.random.multivariate_normal(origin, covs[0], size=ns[0])
        for i in xrange(1,len(covs)):
            s = numpy.append(s, numpy.random.multivariate_normal(origin, covs[i], size=ns[i]), axis=0)
        return s

    def get_weights_and_covariances(self):
        """
            Call get_unscaled_weights_and_convariances and then run the scaling
        """
        ps, covs = zip(*self.get_unscaled_weights_and_covariances())

        # extract the scale components, by design the last two elements
        sx, sy = self.independent_components[-2], self.independent_components[-1]

        return zipa(ps, [scaleW(w, sx.c, sy.c) for w in covs])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    Now we define a number of hypothesis types. Each uses the Distributions stochastics, stored in an array
#    The independent_components here come from Distributions and are used to define the weights_and_covariances
#    via rotation, etc.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Hmain(H):
    """
        Our primary model for data analysis. This lets us fit three components, and we can decide effectively how many we should have.
        NOTE: This includes a sparsity prior on the components.
    """
    def __init__(self):
        self.independent_components = [ DirichletSample(3, alpha=1.0), CircularNormal2D(),  FreeNormal2D(), FreeNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)  ]
    def get_unscaled_weights_and_covariances(self):
        p, c1, c2, c3, _, _ = self.independent_components
        return zipa(p.p, [c1.cov, c2.cov, c3.cov])



class H_circ(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(1), CircularNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD) ] # NOTE We need to use DirichletSample(1) so we get a logp
    def get_unscaled_weights_and_covariances(self):
        p, c, _, _ = self.independent_components
        return zipa(p.p, [c.cov])

class H_plus(H):
    def __init__(self):
        self.independent_components = [DirichletSample(2), AlignedNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD) ]
    def get_unscaled_weights_and_covariances(self):
        p, c, _, _ = self.independent_components
        return zipa(p.p, [c.cov, rot90(c.cov)])

class H_x(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), FreeNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)  ]
    def get_unscaled_weights_and_covariances(self):
        p, c, _, _ = self.independent_components
        return zipa(p.p, [c.cov, rot90(c.cov)])

class H_free(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(1), FreeNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)  ]
    def get_unscaled_weights_and_covariances(self):
        p, c, _, _ = self.independent_components
        return zipa(p.p, [c.cov])

class H_2free(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), FreeNormal2D(), FreeNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)   ]
    def get_unscaled_weights_and_covariances(self):
        p, c1, c2, _, _ = self.independent_components
        return zipa(p.p, [c1.cov, c2.cov])

class H_circ_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), CircularNormal2D(), CircularNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)   ]
    def get_unscaled_weights_and_covariances(self):
        p, c1, c2, _, _ = self.independent_components
        return zipa(p.p, [c1.cov, c2.cov])

class H_x_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(3), CircularNormal2D(),  FreeNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)  ]
    def get_unscaled_weights_and_covariances(self):
        p, c1, c2, _, _ = self.independent_components
        return zipa(p.p, [c1.cov, c2.cov, rot90(c2.cov)])

class H_plus_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(3), CircularNormal2D(),  AlignedNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)  ]
    def get_unscaled_weights_and_covariances(self):
        p, c1, c2, _, _ = self.independent_components
        return zipa(p.p, [c1.cov, c2.cov, rot90(c2.cov)])

class H_free_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), CircularNormal2D(),  FreeNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)  ]
    def get_unscaled_weights_and_covariances(self):
        p, c1, c2, _, _ = self.independent_components
        return zipa(p.p, [c1.cov, c2.cov])

class H_2free_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(3), CircularNormal2D(),  FreeNormal2D(), FreeNormal2D(), Normal1D(sd=WSD), Normal1D(sd=WSD)  ]
    def get_unscaled_weights_and_covariances(self):
        p, c1, c2, c3, _, _ = self.independent_components
        return zipa(p.p, [c1.cov, c2.cov, c3.cov])


# Define all possible types
ALL_TYPES = [H_circ, H_x, H_plus, H_free, H_2free, H_circ_o, H_x_o, H_plus_o, H_free_o, H_2free_o]
