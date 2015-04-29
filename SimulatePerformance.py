"""
    Assess performance through some simulations!
"""
from Hypotheses import ALL_TYPES
from Infer import run_inference

for it in xrange(100):

    for ndata in [10, 50, 100, 200]:

        for ti, t in enumerate(ALL_TYPES):

            genh = t()
            data = genh.sample(N=ndata)

            myiter = run_inference(ALL_TYPES, data, burn=1000, yield_every=1000)

            _, _, model_marginals = myiter.next() # just yield once, via yield_every

            print it, ndata, ti, t.__name__, model_marginals[ti]/float(sum(model_marginals))
