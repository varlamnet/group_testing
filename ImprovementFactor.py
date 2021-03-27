# This script generates improvement factors for each algorithm. E.g. an 
# improvement factor of 5 indicates that 1 test is on average sufficient to 
# test 5 people.

import os
os.chdir(os.path.dirname(__file__))
import numpy as np
from warnings import filterwarnings
filterwarnings('ignore')

n=100 # Number of individuals
k=2 # Number of infected

# This will be used later for calculating the improvement factor
sensdic = {}
specdic = {}
spec95 = {}
sens95 = {}

for name in ['COMP','DD','SCOMP','CBP', 'SR']:
    sensdic[name] = np.genfromtxt(f'data_output/sensitivity_{name}.csv',
    delimiter=',')
    specdic[name] = np.genfromtxt(f'data_output/specificity_{name}.csv',
    delimiter=',')

    # Number of tests for 95% specificity
    sens95[name] = np.where(sensdic[name]>.95)[0][0]
    spec95[name] = np.where(specdic[name]>.95)[0][0]
    # Linear Interpolation to get a slightly more precise number
    sens95[name] = sens95[name]-1 + (.95 - sensdic[name][sens95[name]-1])\
    /(sensdic[name][sens95[name]] - sensdic[name][sens95[name]-1])
    spec95[name] = spec95[name]-1 + (.95 - specdic[name][spec95[name]-1])\
    /(specdic[name][spec95[name]] - specdic[name][spec95[name]-1])

# Number of tests for 95% sensitivity AND 95% specificity
ImproveDorfman = n/(n/5 + (1-(1-k/n)**5)*n)
ImproveSR = n/max(sens95['SR'], spec95['SR'])
ImproveCOMP = n/max(sens95['COMP'], spec95['COMP'])
ImproveDD = n/max(sens95['DD'], spec95['DD'])
ImproveSCOMP = n/max(sens95['SCOMP'], spec95['SCOMP'])
ImproveCBP = n/max(sens95['CBP'], spec95['CBP'])

print(ImproveDorfman, ImproveCOMP, ImproveDD, ImproveCBP, ImproveSCOMP, 
ImproveSR)