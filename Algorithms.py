import numpy as np
from Functions import thresholder
from sklearn.linear_model import Lasso

class Algorithm():
  def __init__(self,x,n,k,m,*args):
    """Runs known algorithms. Takes >=4 arguments and generates an (m x n) 
    sensing matrix A and an (m x 1) vector y of observed test results.

    Args:
        x (ndarray): (n x 1) sparse vector of individual viral loads
        n (int): Number of individuals
        k (int): Number of infected
        m (int): Number of group tests
    """
    self.x, self.n, self.k, self.m = x, n, k, m
    self.A = np.random.binomial(1, p=min(.5,1/k), size=[m,n])
    self.y = np.sign(np.dot(self.A,x))

  def COMP(self):
    """Combinatorial Orthogonal Matching Pursuit (produces no false negatives)

    Returns:
        [ndarray]: (n x 1) prediction vector
    """
    hat = np.zeros([self.n,1])
    for i in range(self.n):
        hat[i] = 1-np.sign(np.sum(self.A[:,i].reshape(-1,1)>self.y))
    return hat

  def DD(self):
    """Definite Defectives (produces no false positives)

    Returns:
        [ndarray]: (n x 1) prediction vector
    """
    hat = np.zeros([self.n,1])
    for i in range(self.n):
        hat[i] = 1-np.sign(np.sum(self.A[:,i].reshape(-1,1)>self.y))
    ind = []
    Atemp = self.A.copy()
    Atemp[np.where(self.y==0)[0],:] = 0
    Atemp[:,np.where(hat==0)[0]] = 0
    for i in np.where(self.y>=1)[0]:
        if np.sum(self.A[i,np.where(hat>0)[0]]) == 1:
            ind.append(list(np.where(Atemp[i,:] == 1)[0])[0])
    hat = np.zeros([self.n,1])
    if len(ind) != 0:
        hat[ind] = 1
    return hat

  def SCOMP(self):
    """Sequential Combinatorial Orthogonal Matching Pursuit (produces no false 
    positives)

    Returns:
        [ndarray]: (n x 1) prediction vector
    """
    hat = np.zeros([self.n,1])
    for i in range(self.n):
        hat[i] = 1-np.sign(np.sum(self.A[:,i].reshape(-1,1)>self.y))
    ind = []
    Atemp = self.A.copy()
    Atemp[np.where(self.y==0)[0],:] = 0
    Atemp[:,np.where(hat==0)[0]] = 0
    for i in np.where(self.y>=1)[0]:
        if np.sum(self.A[i,np.where(hat>0)[0]]) == 1:
            ind.append(list(np.where(Atemp[i,:] == 1)[0])[0])
    hat = np.zeros([self.n,1])
    if len(ind) != 0:
        hat[ind] = 1
    # Additional step: need to explain all positive tests
    while any(np.sign(np.dot(self.A, hat)) != np.sign(self.y)):
        # Find an individual that, if sick, would explain largest # of cases 
        # (break ties randomly)
        pos = np.random.choice(np.argwhere(np.sum(Atemp, axis = 0) 
                        == max(np.sum(Atemp, axis = 0))).flatten()) 
        ind.append(pos)
        hat[ind] = 1
        Atemp[:,pos] = 0
    return hat

  def CBP(self):
    """Combinatorial Basis Pursuit 

    Returns:
        [ndarray]: (n x 1) prediction vector
    """
    indices = set(list(np.arange(self.n)))
    for i in np.where(self.y==0)[0]:
        indices -= set(list(np.where(self.A[i,:]==1)[0]))
    hat = np.zeros([self.n,1])
    hat[list(indices)]=1
    return hat


class SR(Algorithm):
  def __init__(self,x,n,k,m,alpha=.001, tau=.5):
    super().__init__(x,n,k,m)
    """Runs the proposed Sparse Recovery (SR) algorithm. Takes 6 arguments and 
    generates an (m x n) sensing matrix A and an (m x 1) vector y of observed 
    test results.

    Args:
        x (ndarray): (n x 1) sparse vector of individual viral loads
        n (int): Number of individuals
        k (int): Number of infected
        m (int): Number of group tests
        alpha (float): Lasso regulatization parameter
        tau (float): Threshold value (for producing a binary prediction)
    """
    self.alpha, self.tau = alpha, tau
    # Constant column weight (m by n)
    l = int(np.round(m/2+5e-16,0)) # number of ones
    A = np.zeros([m,n])
    A[0:l,:]=1
    for j in range(n):
        A[:,j] = np.random.permutation(A[:,j])
    self.A = A
    self.y = np.dot(self.A,x) 

  def xhat(self):
    """Decoding step: produces an (n x 1) binary prediction vector

    Returns:
        [ndarray]: (n x 1) prediction vector
    """
    rgr_lasso = Lasso(alpha=self.alpha, positive = True)
    rgr_lasso.fit(self.A, self.y)
    hat = rgr_lasso.coef_.reshape(-1,1)
    hat = np.minimum(np.maximum(hat,5e-16),1-5e-16) # map to (0,1)
    return thresholder(hat,self.tau)


class Tap(Algorithm):
  def __init__(self,x,n,k,m,alpha=.001):
    super().__init__(x,n,k,m)
    """Runs Tapestry (Ghosh et al 2020) algorithm. Takes 6 arguments and 
    generates an (m x n) sensing matrix A and an (m x 1) vector y of observed 
    test results.

    Args:
        x (ndarray): (n x 1) sparse vector of individual viral loads
        n (int): Number of individuals
        k (int): Number of infected
        m (int): Number of group tests
        alpha (float): Lasso regulatization parameter
    """
    self.alpha = alpha
    Tapestry = np.zeros([24,60])
    for i,j in enumerate([[10,15], [3,17,21], [6,12], [0,1,20], [9,10,21],
                        [10,16], [2,9], [4,19], [14,18,23], [10,13,22],
                        [1,16], [4,8], [3,8,12], [9,12], [3,5],
                        [12,17,18], [2,14,20], [8,10,14], [1,6,8], [4,5,14],
                        [6,11,15], [5,7,20], [11,12,13], [2,10,17], [2,19,21],
                        [7,8,9], [8,11,22], [7,23], [13,15,20], [2,4,22],
                        [0,2,12], [0,19], [11,18,20], [1,12,21], [0,7,15],
                        [0,18,22], [14,16,21], [4,23], [3,6,13], [15,18,19],
                        [5,15,22], [2,16], [5,8,19], [7,11], [6,16,17],
                        [0,11,23], [6,20,21], [9,17], [6,14,22], [1,4,17],
                        [0,5,17], [1,3,9], [7,18], [22,23], [15,16,23],
                        [13,21], [3,10,20], [13,14], [3,11,19], [1,5,18]]):
      for jj in j:
        Tapestry[jj,i] = 1 
    self.A = Tapestry[:self.m,:]
    self.y = np.dot(self.A,x)

  def xhat(self):
    """Decoding step: produces an (n x 1) binary prediction vector

    Returns:
        [ndarray]: (n x 1) prediction vector
    """
    rgr_lasso = Lasso(alpha=self.alpha, positive = True)
    rgr_lasso.fit(self.A, self.y)
    hat = rgr_lasso.coef_.reshape(-1,1)
    hat = np.minimum(np.maximum(hat,5e-16),1-5e-16) # map to (0,1)
    return thresholder(hat,.5)


