"""

See this: http://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/

"""

import numpy
from numpy import abs

from MetropolisHastings import MHSampler
from Hypotheses import Hmain
from scipy.linalg import eig

PLOT_EVERY = 10000
BURN = 1000
SKIP = 100
PLOT_EVERY = 10000

origin = numpy.zeros(2)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Load the data

import pandas

datatmp = pandas.read_csv('data/OFCinfwater.csv', header=None)

data = datatmp.values # the numpy array


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run the MCMC
import matplotlib.pyplot as plt # for plotting

print "p1 p2 p3 e1 f1 e2 f2 e3 f3"

for i, h in enumerate(MHSampler(Hmain(), data, burn=BURN)):

    if i%(SKIP+1)==0:
        ps, covs = zip(*h.get_unscaled_weights_and_covariances())

        ## And do a plot
        f, plts = plt.subplots(1,3)

        print ps[0], ps[1], ps[2],
        for c in covs:
            ev, em = eig(c)
            ev = numpy.real(ev) ## TODO: ADD ASSERT NO IMAGINARY
            print ev[0], ev[1],
        print ""


    # if i%(PLOT_EVERY+1) == 0:
    #
    #     N=10000 # How many to plot?
    #     ALPHA=0.02
    #     ns = numpy.random.multinomial(N, ps) # distribute among the parts
    #     print ps
    #     for k, cov in enumerate(covs):
    #         plts[k].scatter(data[:,0], data[:,1], c='black', marker="+") # put the dat ain each plot
    #         print cov
    #         s = numpy.random.multivariate_normal(origin, cov, size=ns[k])
    #         plts[k].scatter(s[:,0], s[:,1], c='blue', alpha=ALPHA)
    #
    #     plt.show()