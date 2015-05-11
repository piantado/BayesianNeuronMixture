from Hypotheses import *
from Infer import run_inference
from Visualization import plot

PLOT_EVERY = 1000
BURN = 100
NDATA = 150

thetypes = ALL_TYPES

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  generate the data
genh = H_2free_o()
data = genh.sample(N=NDATA)
print "# Generating: ", NDATA, genh, genh.independent_components[0].p

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# which hypotheses will we consider?

for best_sample, best_posterior, model_marginals in run_inference(thetypes, data, burn=BURN, yield_every=PLOT_EVERY):
    plot(data, thetypes, best_posterior, best_sample, model_marginals)


