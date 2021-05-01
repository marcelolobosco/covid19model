import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import factorial as fat
from scipy.optimize import differential_evolution
import math

# Bibliotecas proprias
from NovoModelo import *


from scipy.stats import lognorm, norm, uniform
import chaospy as cp


#pathway to the data 
path = '../../data/'

# global data
# ajustar pela viremia e anticorpos
dadosViremiaLog10 = pd.read_csv(path+'Viral_load_paper.csv',',')
dadosAnticorposLog2 = pd.read_csv(path+'IgG_IgM.csv',',')
dadosIL6 = pd.read_csv(path+'IL6_ajuste.csv',',')


#Viremia data
dadosViremia = pd.read_csv(path+'dataset_viremia.csv',',')
dadosViremia = dadosViremia[dadosViremia['Num_Samples'] > 5]
virus_mean=np.log10(dadosViremia['Mean']+1)
virus_max=np.log10(dadosViremia['Max']+1)-np.log10(dadosViremia['Mean']+1)
virus_min=np.log10(dadosViremia['Mean']+1)-np.log10(dadosViremia['Min']+1)
virus_std = dadosViremia['Log10_STD']
virus_num = dadosViremia['Num_Samples']
print(dadosViremia)

mask_virus2     =[0,0,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
print(mask_virus2)
for i in range(len(mask_virus2)):
    if i not in dadosViremia['Day']:
        mask_virus2[i] = 0
    else:
        mask_virus2[i] = 1
print(mask_virus2)
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
'''
