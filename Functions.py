import numpy as np

def thresholder(x,tau):
  """ Thresholding function

  Args:
      x (float or ndarray): a scalar
      tau (float or ndarray): value b/w 0 and 1

  Returns:
      [float or ndarray]: 1 if x > tau, 0 if x <= tau
  """
  assert tau>=0 and tau<=1, "tau not in [0,1]"
  return np.round((np.sign(x-tau)+1)/2) 

def rmse(x, xhat):
  """ Root Mean Square Error

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [float or ndarray]: Root Mean Square Error
  """
  if np.all(x == 0):
    if np.all(xhat == 0):
      return 0.
    else:
      return 1.
  return np.linalg.norm(x - xhat) / np.linalg.norm(x)

def tp(x,xhat):
  """ True Positive

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [int]: True Positive
  """
  return np.sum(np.logical_and(x>0,xhat>0))

def fp(x, xhat):
  """ False Positive

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [int]: False Positive
  """
  return np.sum(np.logical_and(x==0,xhat>0))

def tn(x,xhat):
  """ True negative

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [int]: True negative
  """
  return np.sum(np.logical_and(x==0,xhat==0))

def fn(x, xhat):
  """ False Negative

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [int]: False Negative
  """
  return np.sum(np.logical_and(x>0,xhat==0))

def precision(x, xhat): 
  """ Precision, or positive predictive value

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [float]: Precision
  """
  tp_ = tp(x,xhat)
  fp_ = fp(x,xhat)
  if tp_ + fp_ != 0:
    precision = tp_ / (tp_ + fp_)
  else:
    precision = 1
  return precision
  
def sensitivity(x, xhat):
  """ Sensitivity, Recall, or true positive rate

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [float]: sensitivity
  """
  tp_ = tp(x,xhat)
  fn_ = fn(x,xhat)
  if tp_ + fn_ != 0:
    sensitivity = tp_ / (tp_ + fn_)
  else:
    sensitivity = 1
  return sensitivity

def specificity(x, xhat):
  """ Specificity, or True Negative rate

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [float]: Specificity
  """
  tn_ = tn(x,xhat)
  fp_ = fp(x,xhat)
  if tn_ + fp_ != 0:
    specificity = tn_ / (tn_ + fp_)
  else:
    specificity = 1
  return specificity

def fpr(x,xhat):
  """ False Positive rate

  Args:
      x (array_like): True value
      xhat (array_like): Prediction

  Returns:
      [float]: False Positive rate
  """
  return 1-specificity(x, xhat)
