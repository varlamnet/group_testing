# This script generates RMSE errors averaged over Monte Carlo experiments for a 
# range of group tests, m = 1,..., mmax.

import numpy as np
from MonteCarlo import runMonteCarlo
from Functions import rmse
import plotly.graph_objects as go
from Algorithms import *
from warnings import filterwarnings
filterwarnings('ignore')

n = 100 # Number of individuals
k = 2 # Number of infected
mmax = 40 # Maximum number of group tests
Monte = 1000 # Number of Monte Carlo experiments

loss = rmse # Loss function (can change to any in Functions.py)
z = np.arange(mmax-1)

rmsedic = {}

rmsedic['COMP'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
    x,n,k,m: Algorithm(x,n,k,m).COMP()).reshape(-1,)
rmsedic['DD'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
    x,n,k,m: Algorithm(x,n,k,m).DD()).reshape(-1,)
rmsedic['SCOMP'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
    x,n,k,m: Algorithm(x,n,k,m).SCOMP()).reshape(-1,)
rmsedic['CBP'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
    x,n,k,m: Algorithm(x,n,k,m).CBP()).reshape(-1,)
rmsedic['SR'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
    x,n,k,m: SR(x,n,k,m).xhat()).reshape(-1,)

fig = go.Figure()
for key, value in rmsedic.items():
    if key == 'SR':
        fig.add_trace(go.Scatter(x=z, y=value, mode='lines', name=key, 
                         line=dict(color='black', width=4, dash='dot')))
    else:
        fig.add_trace(go.Scatter(x=z, y=value, mode='lines', name=key))

fig.update_layout(autosize=False,
                  font=dict(size=13),
                  title = f"RMSE as a function of group tests N={n}, k={k}",
                  title_x=0.5,
                  xaxis_title='Tests',
                  xaxis_range = [0.01,mmax-2],
                  yaxis_range = [-0.02,6],
                  yaxis_title='RMSE',
                  legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.75, 
                  bgcolor='rgba(0,0,0,0)'),
    width=550, height=400, margin=dict(l=50, r=50, b=20, t=40))
fig.show()

# for names in ['COMP','DD','SCOMP','CBP', 'SR']:
#     np.savetxt(f"{loss.__name__}"+f"_{names}.csv", series, delimiter=",")
