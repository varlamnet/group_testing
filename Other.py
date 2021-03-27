# This script generates a noisy (n x 1) vector x and applies algorithms from
#  the original manuscript to recover it through m measurements. Note that this 
# is a single generation and results may vary substantially. Monte Carlo 
# scripts resolve this by averaging over many generations.

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Algorithms import Algorithm, SR

n = 100 # Number of individuals
k = 2   # Number of infected
m = 20  # Number of group tests

xpure = np.zeros([n,1])
xpure[0:k] = 1
np.random.shuffle(xpure)
x = xpure + 0.05 * np.random.randn(n).reshape(-1,1)


xhat1 = Algorithm(x,n,k,m).COMP()
xhat2 = Algorithm(x,n,k,m).DD()
xhat3 = Algorithm(x,n,k,m).CBP()
xhat4 = Algorithm(x,n,k,m).SCOMP()
xhat5 = SR(x,n,k,m).xhat()

fig = make_subplots(
    rows=1, cols=6, 
    subplot_titles=("True", "COMP", "DD", "CBP", "SCOMP", "SR"), 
    shared_yaxes=True
    )

for num,pred in enumerate([x,xhat1,xhat2,xhat3,xhat4,xhat5]):
    fig.add_trace(go.Heatmap(z=pred), row=1, col=num+1)
    fig.update_traces(dict(showscale=False))


fig.update_layout(autosize=False,
    width=800, height=400, margin=dict(l=50, r=50, b=20, t=30))
fig.update_xaxes(showticklabels=False)
fig.show()
