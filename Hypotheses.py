
import numpy
import random
from copy import deepcopy
from numpy import log
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from Distributions import *

origin = numpy.zeros(2)

class H(object):
    """ A class to represent mixture hypotheses using a list of covariance matrices """

    def __init__(self):
        raise NotImplementedError # must be implemented by subclasses

    def get_weights_and_covariances(self):
        """ Return a paired list of weights and corresponding covariance matrices. """
        raise NotImplementedError

    def compute_prior(self):
        """ For now, we'll use our proposal probability. """
        return sum([x.lp for x in self.independent_components])

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
        x = self.independent_components[i].propose() # just call the constructor

        mynew.independent_components[i] = x # set it

        # Return the new guy and my fb
        return mynew, (mynew.independent_components[i].lp - self.independent_components[i].lp)


    def compute_likelihood(self, data):
        ps, covs = zip(*self.get_weights_and_covariances()) #

        # get the log prob under each covariance matrix
        lps = map(lambda c: multivariate_normal.logpdf(data, mean=origin, cov=c), covs)

        ## TODO: DOUBLE CHECK THIS:

        return sum(logsumexp([lp + log(p) for p, lp in zip(ps, lps)], axis=0))

    def sample(self, N=100):

        ps, covs = zip(*self.get_weights_and_covariances())

        # figure out how to divide our counts between p
        ns = numpy.random.multinomial(N, ps)

        s = numpy.random.multivariate_normal(numpy.zeros(2), covs[0], size=ns[0])
        for i in xrange(1,len(covs)):
            s = numpy.append(s, numpy.random.multivariate_normal(numpy.zeros(2), covs[i], size=ns[i]), axis=0)
        return s

"""
    Now we define a number of hypothesis types. Each uses the Distributions stochastics, stored in an array
"""

class H_circ(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(1), CircularNormal2D() ] # NOTE We need to use DirichletSample(1) so we get a logp
    def get_weights_and_covariances(self):
        p, c = self.independent_components
        return zip(p.p, [c.cov])

class H_plus(H):
    def __init__(self):
        self.independent_components = [DirichletSample(2), AlignedNormal2D()]
    def get_weights_and_covariances(self):
        p, c = self.independent_components
        return zip(p.p, [c.cov, rot90(c.cov)])

class H_x(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), FreeNormal2D() ]
    def get_weights_and_covariances(self):
        p, c = self.independent_components
        return zip(p.p, [c.cov, rot90(c.cov)])

class H_free(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(1), FreeNormal2D() ]
    def get_weights_and_covariances(self):
        p, c = self.independent_components
        return zip(p.p, [c.cov])

class H_2free(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), FreeNormal2D(), FreeNormal2D()  ]
    def get_weights_and_covariances(self):
        p, c1, c2 = self.independent_components
        return zip(p.p, [c1.cov, c2.cov])

class H_circ_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), CircularNormal2D(), CircularNormal2D()  ]
    def get_weights_and_covariances(self):
        p, c1, c2 = self.independent_components
        return zip(p.p, [c1.cov, c2.cov])

class H_x_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), CircularNormal2D(),  FreeNormal2D() ]
    def get_weights_and_covariances(self):
        p, c1, c2 = self.independent_components
        return zip(p.p, [c1.cov, c2.cov, rot90(c2.cov)])

class H_plus_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), CircularNormal2D(),  AlignedNormal2D() ]
    def get_weights_and_covariances(self):
        p, c1, c2 = self.independent_components
        return zip(p.p, [c1.cov, c2.cov, rot90(c2.cov)])

class H_free_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(2), CircularNormal2D(),  FreeNormal2D() ]

    def get_weights_and_covariances(self):
        p, c1, c2 = self.independent_components
        return zip(p.p, [c1.cov, c2.cov])

class H_2free_o(H):
    def __init__(self):
        self.independent_components = [ DirichletSample(3), CircularNormal2D(),  FreeNormal2D(), FreeNormal2D() ]
    def get_weights_and_covariances(self):
        p, c1, c2, c3 = self.independent_components
        return zip(p.p, [c1.cov, c2.cov, c3.cov])



