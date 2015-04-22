"""
    The big hypothesis, doing inference over little Hes
"""

import numpy

class MetaHypothesis(object):
    """
        Now define a meta-hypothesis which does inference over each of the Hes
    """
    def __init__(self, hyps):
        """ Take in a list of hypotheses (object types) whose constructors get called to propose. This proposes from the prior. """
        self.hyps = hyps
        self.i = numpy.random.randint(len(hyps)) # which hypothesis are we using?
        self.h = hyps[self.i]() # call this constructor

    def compute_posterior(self, data):
        self.posterior_score = self.h.compute_posterior(data)
        return self.posterior_score

    def lp_proposal(self):
        return self.h.lp_proposal

    def sample(self, N=100):
        return self.h.sample(N=N)

        
