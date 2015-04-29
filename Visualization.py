
import matplotlib.pyplot as plt # for plotting

def plot(data, thetypes, best_posterior, best_sample, model_marginals):
    """ Convenient plots of our summary stats from Infer """

    # plot each model
    f, plts = plt.subplots(1,len(thetypes)+1)
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

