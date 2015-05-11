"""
Run the actual inference

"""
import itertools
from numpy import log
from Hypotheses import *
from MetropolisHastings import MHSampler

def run_inference(thetypes, data, burn=1000, yield_every=1000):

    # Make one chain for each type of thing
    model_samplers = [ MHSampler(t(), data, burn=burn) for t in thetypes]

    # we'll make an index variable which switches between chains
    idx = random.randint(0, len(model_samplers)-1)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize some stats here. This is what will be yielded

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
        if log(random.random()) < samples[idxprop].posterior_score - samples[idx].posterior_score:
            idx=idxprop

        model_marginals[idx] += 1 # update the marginal model count

        # make a fancier plot with one for each submodel
        if i%yield_every == 0 and i>0:
            yield (best_sample, best_posterior, model_marginals)

