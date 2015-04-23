"""
    Set up our hypotheses, as subclasses of H
"""
import numpy
from numpy import log
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
from covariances import *

origin = numpy.zeros(2)

def rdirichlet(n, a):
    """ Returns a dirichlet sample and its lp """
    alpha = numpy.array([a]*n)
    x = numpy.random.dirichlet(alpha)
    lp = scipy.stats.dirichlet.logpdf(x, alpha)
    return x, lp

class H():

    ALPHA = 2.0 # the alpha parameter

    """ A class to represent mixture hypotheses using a list of covariance matrices """
    def __init__(self):
        raise NotImplementedError # must be implemented by subclasses

    def compute_prior(self):
        """ For now, we'll use our proposal probability, that had better be set by other initializers """
        return self.lp_proposal

    def compute_posterior(self, data):
        try:
            self.posterior_score = self.compute_prior() + self.compute_likelihood(data)
        except (numpy.linalg.linalg.LinAlgError, ValueError):
            self.posterior_score = float("-inf")

        return self.posterior_score

    def compute_likelihood(self, data):
        assert len(self.covs) == len(self.p), "*** Covariances and probabilities must be the same size!"

        # get the log prob under each covariance matrix
        lps = map(lambda cov: multivariate_normal.logpdf(data, mean=origin, cov=cov), self.covs)

        return sum(logsumexp([lp + log(mp) for mp, lp in zip(self.p, lps)], axis=0))

    def sample(self, N=100):

        # figure out how to divide our counts between p
        ns = numpy.random.multinomial(N, self.p)

        s = numpy.random.multivariate_normal(numpy.zeros(2), self.covs[0], size=ns[0])
        for i in xrange(1,len(self.covs)):
            s = numpy.append(s, numpy.random.multivariate_normal(numpy.zeros(2), self.covs[i], size=ns[i]), axis=0)
        return s


# Subclasses that implement various restrictions

class H_circ(H):
    def __init__(self):
        c, lp = random_invwishart(1) # let's draw from a 1-D wishart for this
        self.covs = [ numpy.diag([c,c]) ]
        self.lp_proposal = lp
        self.p = [1.0]


class H_x(H):
    def __init__(self):
        c, lp = random_invwishart(2)
        self.covs = [ c, rot90(c) ]
        self.p, lpd = rdirichlet(2, H.ALPHA)
        self.lp_proposal = lp + lpd

class H_plus(H):
    def __init__(self):
        cx, lpx = random_invwishart(1) # draw these separately and use a diagonal
        cy, lpy = random_invwishart(1)
        c = numpy.diag([cx,cy])
        self.p, lpd = rdirichlet(2, H.ALPHA)
        self.lp_proposal = lpx+lpy+lpd
        self.covs = [ c, rot90(c) ]

class H_free(H):
    def __init__(self):
        c, lp = random_invwishart(2)
        self.p = [1.0]
        self.lp_proposal = lp
        self.covs = [ c ]


class H_2free(H):
    def __init__(self):
        c1, lp1 = random_invwishart(2)
        c2, lp2 = random_invwishart(2)
        self.p, lpd = rdirichlet(2, H.ALPHA)
        self.lp_proposal = lp1+lp2+lpd
        self.covs = [ c1, c2 ]



class H_circ_o(H):
    def __init__(self):
        c, lp = random_invwishart(1) # let's draw from a 1-D wishart for this
        co, lpo = random_invwishart(1)
        self.p, lpd = rdirichlet(2, H.ALPHA)
        self.lp_proposal = lp+lpo+lpd
        self.covs = [ numpy.diag([c,c]), numpy.diag([co,co]) ]

class H_x_o(H):
    def __init__(self):
        c, lp = random_invwishart(2)
        co, lpo = random_invwishart(1)
        self.p, lpd = rdirichlet(3, H.ALPHA)
        self.lp_proposal = lp+lpo+lpd
        self.covs = [ c, rot90(c), numpy.diag([co,co]) ]

class H_plus_o(H):
    def __init__(self):
        cx, lpx = random_invwishart(1) # draw these separately and use a diagonal
        cy, lpy = random_invwishart(1)
        co, lpo = random_invwishart(1)
        self.p, lpd = rdirichlet(3, H.ALPHA)
        c = numpy.diag([cx,cy])
        self.lp_proposal = lpx+lpy+lpo+lpd
        self.covs = [ c, rot90(c), numpy.diag([co,co]) ]

class H_free_o(H):
    def __init__(self):
        c, lp = random_invwishart(2)
        co, lpo = random_invwishart(1)
        self.p, lpd = rdirichlet(2, H.ALPHA)
        self.lp_proposal = lp+lpo+lpd
        self.covs = [ c, numpy.diag([co,co]) ]

class H_2free_o(H):
    def __init__(self):
        c1, lp1 = random_invwishart(2)
        c2, lp2 = random_invwishart(2)
        co, lpo = random_invwishart(1)
        self.p, lpd = rdirichlet(3, H.ALPHA)
        self.lp_proposal = lp1+lp2+lpo+lpd
        self.covs = [ c1, c2, numpy.diag([co,co]) ]





if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot some if run as main
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    import matplotlib.pyplot as plt # for plotting

    def myscatter(plt, m, **kwargs):
        plt.scatter(m[:,0], m[:,1], **kwargs)

    for i in xrange(100):
        h = H_free_o()
        plt.clf()
        myscatter(plt, h.sample(N=400), c='red',  alpha=0.9 )
        print h.covs
        plt.show()
    