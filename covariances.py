import numpy

import scipy.stats
from scipy.stats import wishart ## NOTE: Requires scipy 0.16 development version
from numpy.linalg import inv

def random_invwishart(n):
    """
        Returns the sample and its log probability under the prior
    """

    df = n # the dimension

    m = wishart.rvs(df, numpy.eye(n))
    lp = wishart.logpdf(m, df, numpy.eye(n))

    if n == 1:
        return 1.0/m, lp
    else:
        return inv(m), lp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Rotation operations

m90 = numpy.matrix([[0,1],[-1,0]]) # the rotation matrix -- 90 degrees
def rotate(m, R=m90):
    """ Rotate a matrix (m) by another (R)"""
    return (R*m)*(R.transpose())

def rotate_theta(m, theta):
    """ Rotate by an angle theta"""
    R = numpy.matrix([[numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]])
    return rotate(m, R=R)

def rot90(m1):
    """ 90 degree rotation of m1 """
    return rotate(m1, R=m90)


if __name__ == "__main__":

    # Plot some wisharts

    import matplotlib.pyplot as plt

    x=numpy.linspace(1e-5,10,100)
    plt.plot(x, wishart.pdf(x, df=2,scale=1))

    plt.show()
	