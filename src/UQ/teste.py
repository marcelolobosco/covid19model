import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from tqdm import tqdm
import pandas as pd
import sys

from scipy.stats import lognorm, norm, uniform
import chaospy as cp

'''
dist = cp.LogNormal(np.log(0.26697788755836127), 0.046378061446869505, -0.08710186740229636)
ns=10000
samples = dist.sample(ns)
samp = samples.T

r = lognorm.rvs(0.046378061446869505, -0.08710186740229636, 0.26697788755836127, size=1000)


dist = cp.Normal(0.018791138450681123, 0.0018960285902594259)
ns=10000
samples = dist.sample(ns)
samp = samples.T

r = norm.rvs(0.018791138450681123, 0.0018960285902594259, size=1000)
'''


dist = cp.Uniform(48.839824314323145, 48.839824314323145+24.376228335718743)
ns=10000
samples = dist.sample(ns)
samp = samples.T

r = uniform.rvs(48.839824314323145, 24.376228335718743, size=1000)

(48.839824314323145, 24.376228335718743)

plt.hist(samp, bins=50, density=True)
plt.hist(r, bins=50, density=True)
plt.show();
