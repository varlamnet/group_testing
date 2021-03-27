# This script generates sensitivity averaged over Monte Carlo experiments for a 
# range of group tests, m = 1,..., mmax.

import numpy as np
from MonteCarlo import runMonteCarlo
from Functions import sensitivity
import plotly.graph_objects as go
from Algorithms import *
from warnings import filterwarnings
filterwarnings('ignore')

n = 100 # Number of individuals
k = 2 # Number of infected
mmax = 40 # Maximum number of group tests
Monte = 1000 # Number of Monte Carlo experiments

loss = sensitivity # Loss function (can change to any in Functions.py)
z = np.arange(mmax-1)

sensdic = {}

sensdic['COMP'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
     x,n,k,m: Algorithm(x,n,k,m).COMP()).reshape(-1,)
sensdic['DD'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
     x,n,k,m: Algorithm(x,n,k,m).DD()).reshape(-1,)
sensdic['SCOMP'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
     x,n,k,m: Algorithm(x,n,k,m).SCOMP()).reshape(-1,)
sensdic['CBP'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
     x,n,k,m: Algorithm(x,n,k,m).CBP()).reshape(-1,)
sensdic['SR'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
     x,n,k,m: SR(x,n,k,m).xhat()).reshape(-1,)

fig = go.Figure()
for key, value in sensdic.items():
    if key == 'SR':
        fig.add_trace(go.Scatter(x=z, y=value, mode='lines', name=key, 
                         line=dict(color='black', width=4, dash='dot')))
    else:
        fig.add_trace(go.Scatter(x=z, y=value, mode='lines', name=key))

fig.update_layout(autosize=False,
                  title = f"Sensitivity as a function of group tests",
                  font=dict(size=13),
                  title_x=0.5,
                  xaxis_title='Tests',
                  xaxis_range = [0.01,mmax-2],
                  yaxis_range = [0,1.01],
                  yaxis_title='Sensitivity',
                  legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.05, 
                  bgcolor='rgba(0,0,0,0)'),
    width=550, height=400, margin=dict(l=50, r=50, b=20, t=40))
fig.show()

# for key, value in sensdic.items():
#     np.savetxt(f"{loss.__name__}"+f"_{key}.csv", value, delimiter=",")
