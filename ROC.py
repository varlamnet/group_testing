# This generates ROC curve and AUC value. The thresholding parameter tau is 
# varied from 0 to 1, false positive and true positive (sensitivity) rates are 
# calculated. Number of group tests 'm' is fixed to be 20, this can be changed 
# in MonteCarlo.py.

import numpy as np
from MonteCarlo import runMonteCarloROC
from Functions import rmse
import plotly.graph_objects as go
from Algorithms import *
from warnings import filterwarnings
filterwarnings('ignore')

n = 100 # Number of individuals
k = 2 # Number of infected
Monte = 10 # Number of Monte Carlo experiments
thresholds = np.linspace(0,1,101) # Range of threshold value (tau)

# Compute temporary false positive and true positive (sensitivity) rates
temp_fp,temp_tp = runMonteCarloROC(n,k,thresholds, Monte).run()

# The following is not conceptually important, this is for smoothing the 
# estimates
tprate=temp_tp.copy()
for i,j in enumerate(tprate):
  if i>0 and tprate[i]>tprate[i-1]:
    tprate[i] = tprate[i-1]/3 + tprate[i-2]/3 + tprate[i-3]/3
for _ in range(5):
  for i,j in enumerate(tprate):
    if i>0 and i<99:
      tprate[i] = tprate[i-1]/2 + tprate[i+1]/2

fprate=temp_fp.copy()
for i,j in enumerate(fprate):
  if i>0 and fprate[i]>fprate[i-1]:
    fprate[i] = fprate[i-1]/3 + fprate[i-2]/3 + fprate[i-3]/3
for _ in range(5):
  for i,j in enumerate(fprate):
    if i>0 and i<99:
      fprate[i] = fprate[i-1]/2 + fprate[i+1]/2

from sklearn.metrics import auc
fig = go.Figure()
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

fig.add_trace(go.Scatter(x=fprate.flatten(), y=tprate.flatten(), name="First", 
mode='lines', line=dict(color='black', width=4)))
fig.update_layout(title = \
  f"ROC curve (AUC={auc(fprate.flatten(), tprate.flatten()):.4f})",
                  title_x=0.5,
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    yaxis=dict(range=(0, 1)),
    autosize=False,
    width=550, height=400, margin=dict(l=50, r=50, b=20, t=40))
fig.show()