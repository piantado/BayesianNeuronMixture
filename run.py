"""
    - Check out scipy GMM 
    - NOTE: You may need to install new versions of scipy (and for that, cython)

    TODO: Check rotation -- make sure its doing it right!
    TODO: Fancier proposals -- we don't have to just propose from the prior, but can keep some!
"""

# Import relevant modules
import numpy
from copy import deepcopy

from H import *
from MetaHypothesis import MetaHypothesis
from MetropolisHastings import MHSampler

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run Metropolis-hastings

import matplotlib.pyplot as plt # for plotting

# generate the data
genh = H_x_o()
data = genh.sample(N=50)
print "# Generating: ", genh.p

PLOT_EVERY = 150000

# which hypotheses will we consider?
thetypes = [H_circ, H_x, H_plus, H_free, H_2free, H_circ_o, H_x_o, H_plus_o, H_free_o, H_2free_o]

h = MetaHypothesis(thetypes)
h.compute_posterior(data)

## Store some information here
typemax           = [ None ] * len(thetypes) # find the best we've seen so far of each type
typemax_posterior = [ float("-inf") ] * len(thetypes) # find the score of the best for each type
model_marginals   = [ 0 ] * len(thetypes) # Posterior marginal on each model type

def propose(h):
    # For now, use a boring proposla distribution
    # return forward-backward
    p = MetaHypothesis(thetypes)
    p.compute_posterior(data)

    fb = p.lp_proposal() - h.lp_proposal()

    return p, fb

##
for i, h in enumerate(MHSampler(h, propose, data)):

    if h.posterior_score > typemax_posterior[h.i]: # if this is better than the best we've seen for this
        typemax[h.i] = h
        typemax_posterior[h.i] = h.posterior_score

    # Actually draw a sample
    model_marginals[h.i] += 1 # increment our count of how many models use i

    # make a fancier plot with one for each submodel
    if i%PLOT_EVERY == 0 and i>0:

        # plot each model
        f, plts = plt.subplots(1,11)
        for mi in range(len(thetypes)):

            if typemax[mi] is None:
                continue

            ## Plot the data
            plts[mi].scatter(data[:,0], data[:,1], c='black', marker="+")

            # TODO: Should be ellipses
            m = typemax[mi].sample(N=5000)
            plts[mi].scatter(m[:,0], m[:,1], c='blue', alpha=0.01)
            plts[mi].set_title(str(mi) +": "+ thetypes[mi].__name__ + "\n"+\
                               " MAP: " + str(round(typemax_posterior[mi],1)) +"\n"+\
                               " p: " + str(map(lambda x: round(x,3), typemax[mi].h.p)) )

        plts[-1].bar(range(len(thetypes)), model_marginals)
        plt.show()






