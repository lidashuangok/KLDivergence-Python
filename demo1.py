# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-

import numpy as np
import scipy
from scipy.stats import norm
from matplotlib import pyplot as plt

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def kl_divergence_1(p, q):
    return scipy.stats.entropy(p,q)

x = np.arange(-10, 10, 0.001)
p = norm.pdf(x, 0, 2)
q = norm.pdf(x, 2, 2)
print(kl_divergence(p,q))
print(kl_divergence_1(p,q))
# plt.title('KL(P||Q) = %1.3f' % kl_divergence(p, q))
# plt.plot(x, p)
# plt.plot(x, q, c='red')