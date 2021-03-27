# This script produces RMSE comparison between the proposed SR algorithm and 
# Tapestry (Ghosh et al. 2020).

import numpy as np
from MonteCarlo import runMonteCarlo
from Functions import rmse
import plotly.graph_objects as go
from Algorithms import *
from warnings import filterwarnings
filterwarnings('ignore')

n = 60 # Number of individuals
k = 2 # Number of infected
mmax = 30 # Maximum number of group tests
Monte = 1000 # Number of Monte Carlo experiments

loss = rmse # Change this to any (see the loss functions above)
z = np.arange(mmax-1)

rmsedic = {}

rmsedic['SR'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
    x,n,k,m: SR(x,n,k,m).xhat()).reshape(-1,)
rmsedic['Tap'] = runMonteCarlo(loss,n, k, mmax, Monte).run(lambda \
    x,n,k,m: Tap(x,n,k,m).xhat()).reshape(-1,)

fig = go.Figure()
fig.add_trace(go.Scatter(x=z, y=rmsedic['SR'], mode='lines', name='SR', 
                        line=dict(color='black', width=4, dash='dot')))
fig.add_trace(go.Scatter(x=z, y=rmsedic['Tap'], mode='lines', name='Tap'))

fig.update_layout(autosize=False,
                font=dict(
                    size=13
                    ),
                title = f"k = {k}, N = {n}",
                title_x=0.5,
                xaxis_title='Tests',
                xaxis_range = [0.01,mmax-2],
                yaxis_range = [-0.02,3],
                yaxis_title='RMSE',
                legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.75, 
                bgcolor='rgba(0,0,0,0)'),
    width=550, height=400, margin=dict(l=50, r=50, b=20, t=40))
fig.show()