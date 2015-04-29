import numpy
import itertools
from numpy import exp
from Hypotheses import *
from MetropolisHastings import MHSampler
import matplotlib.pyplot as plt # for plotting

PLOT_EVERY = 15000
BURN = 5000
NDATA = 150

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  generate the data
genh = H_2free_o()
data = genh.sample(N=NDATA)
print "# Generating: ", NDATA, genh, genh.independent_components[0].p

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# which hypotheses will we consider?

thetypes = [H_circ, H_x, H_plus, H_free, H_2free, H_circ_o, H_x_o, H_plus_o, H_free_o, H_2free_o]

# Make one chain for each type of thing
model_samplers = [ MHSampler(t(), data, burn=BURN) for t in thetypes]

# we'll make an index variable which switches between chains
idx = random.randint(0, len(model_samplers))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize some stats here

best_sample       = [ None ] * len(thetypes) # find the best we've seen so far of each type
best_posterior    = [ float("-inf") ] * len(thetypes) # find the score of the best for each type
model_marginals   = numpy.array([ 0 ] * len(thetypes)) # Posterior marginal on each model type

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Actually run MCMC
for i in itertools.count():

    # Update each sampler
    samples = [] # a list of model samples this iteration
    for m, ms in enumerate(model_samplers):
        s = ms.next()
        samples.append(s)
        # Keep track of the best found for each
        if s.posterior_score > best_posterior[m]:
            best_posterior[m] = s.posterior_score
            best_sample[m] = s
    assert len(samples) == len(thetypes)

    ## propose a change to idx, and update
    idxprop = random.randint(0,len(model_samplers)-1)
    if random.random() < exp(samples[idxprop].posterior_score - samples[idx].posterior_score):
        idx=idxprop

    model_marginals[idx] += 1 # update the marginal model count

    # make a fancier plot with one for each submodel
    if i%PLOT_EVERY == 0 and i>0:

        # plot each model
        f, plts = plt.subplots(1,len(model_samplers)+1)
        for mi in range(len(thetypes)):

            ## Plot the data
            plts[mi].scatter(data[:,0], data[:,1], c='black', marker="+")

            # TODO: Should draw some ellipses!
            m = best_sample[mi].sample(N=5000)
            plts[mi].scatter(m[:,0], m[:,1], c='blue', alpha=0.02)
            plts[mi].set_title(str(mi) +": "+ thetypes[mi].__name__ + "\n"+\
                               " MAP: " + str(round(best_posterior[mi],1)) +"\n"+\
                               " p: " + str(map(lambda x: round(x,3), best_sample[mi].independent_components[0].p)) )

        plts[-1].bar(range(len(thetypes)), model_marginals)
        plt.show()



