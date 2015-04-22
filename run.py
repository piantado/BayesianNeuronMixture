"""
    - Check out scipy GMM 
    - NOTE: You may need to install new versions of scipy (and for that, cython)

    TODO: Check rotation -- make sure its doing it right!

"""

# Import relevant modules
import numpy
from copy import deepcopy
from numpy import log

from H import *
from MetaHypothesis import MetaHypothesis


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run Metropolis-hastings

import matplotlib.pyplot as plt # for plotting

# generate the data
data = H_plus().sample(N=100)

BURN = 100 # how many samples do we burn?
SAMPLES = 100000
PLOT_EVERY = 5000

# which hypotheses will we consider?
thetypes = [H_circ, H_x, H_plus, H_free, H_2free, H_circ_o, H_x_o, H_plus_o, H_free_o, H_2free_o]

h = MetaHypothesis(thetypes)
h.compute_posterior(data)

## Store some information here
typemax           = [ None ] * len(thetypes) # find the best we've seen so far of each type
typemax_posterior = [ float("-inf") ] * len(thetypes) # find the score of the best for each type
model_marginals   = [ 0 ] * len(thetypes) # Posterior marginal on each model type

## Run the sampler
for i in xrange(SAMPLES):

    p = MetaHypothesis(thetypes)
    p.compute_posterior(data)

    if log(numpy.random.random()) < p.posterior_score - h.posterior_score - (p.lp_proposal()-h.lp_proposal()):
        h = p

    # track the best we've found. Here we use p to cover everything, even rejected samples
    if p.posterior_score > typemax_posterior[p.i]: # if this is better than the best we've seen for this
        typemax[p.i] = p
        typemax_posterior[p.i] = p.posterior_score

    # Actually draw a sample
    if i > BURN:

        model_marginals[h.i] += 1 # increment our count of how many models use i
        # print h.posterior_score, h.i

    # make a fancier plot with one for each submodel
    if i%PLOT_EVERY == 0 and i>BURN:

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
            plts[mi].set_title(str(mi) +": "+ thetypes[mi].__name__ + "\n MAP: " + str(round(typemax_posterior[mi],1)) )

        plts[-1].bar(range(len(thetypes)), model_marginals)
        plt.show()






