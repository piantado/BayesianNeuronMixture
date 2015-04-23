import numpy
from numpy import log

def MHSampler(h, propose, data, burn=0, steps=1000000):
    """
    Yield MH samples

    h0 -- a starting hypothesis
    propose -- a function that takes a hypothesis and proposes
    """

    h.compute_posterior(data)

    for i in xrange(steps):
        p, fb = propose(h)
        p.compute_posterior(data)

        if log(numpy.random.random()) < p.posterior_score - h.posterior_score - fb:
            h = p

        # yield the current sample
        if i>burn:
            yield h
