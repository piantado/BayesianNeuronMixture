"""
    Assess performance through some simulations.

    python SimulatePerformance.py > simulation.txt

    Then run analyze-simulation.py in R

    This primarily tells us that some of these models are easier to detect than others, even when the true model is known

    We can run 5 jobs in parallel (using GNU parallel) via
        $ seq 1 5 | parallel 'python SimulatePerformance.py' > simulation.txt

"""

import sys
from Hypotheses import ALL_TYPES
from Infer import run_inference

for it in xrange(100):

    for ndata in [500, 200, 100, 50, 20]:

        for ti, t in enumerate(ALL_TYPES):

            genh = t()
            data = genh.sample(N=ndata)

            myiter = run_inference(ALL_TYPES, data, burn=1000, yield_every=1000)

            _, _, model_marginals = myiter.next() # just yield once, via yield_every

            Zmm = float(sum(model_marginals))

            print it, ndata, ti, t.__name__, model_marginals[ti]/Zmm,
            for mm in model_marginals:
                print mm/float(sum(model_marginals)),
            print "\n",

            sys.stdout.flush()