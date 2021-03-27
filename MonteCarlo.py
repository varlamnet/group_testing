import numpy as np
from Algorithms import Algorithm, SR
from Functions import fpr, sensitivity

class runMonteCarlo():
    def __init__(self,loss,n,k,mmax, Monte, alpha=.001):
        """Runs Monte Carlo experiments: for each number of group tests, from 1 
        to mmax, the loss function is calculated and averaged over 'Monte' 
        iterations. 

        Args:
            loss (function): Loss function
            n (int): Number of individuals
            k (int): Number of infected
            mmax (int): Max number of group tests to iterate over
            Monte (int): Number of Monte Carlo experiments
            alpha (float, optional): Lasso regulatization parameter. Defaults 
            to .001.
        """
        self.loss, self.n, self.k, self.mmax, self.Monte, self.alpha = loss, \
        n, k, mmax, Monte, alpha

    def run(self, alg):
        rmsearray = np.array([])
        for m in np.arange(1,self.mmax):
            err = []
            for _ in range(self.Monte):
                xpure = np.zeros([self.n,1])
                xpure[0:self.k] = 1
                np.random.shuffle(xpure)
                x = xpure 

                # Prediction
                hat = alg(x,self.n,self.k,m)

                # Error
                err.append(self.loss(x, np.sign(np.maximum(0,np.round(hat)))))
            rmsearray = np.append(rmsearray, sum(err) / len(err))
        return rmsearray.reshape(-1,1)

class runMonteCarloROC():
    def __init__(self,n,k, thresholds, Monte):
        """Runs Monte Carlo experiments the proposed Sparse Recovery (SR) 
        algorithm for generating ROC/AUC: for each threshold value (tau) in 
        range 'thresholds', false positive rate and sensitivity are calculated, 
        and averaged over 'Monte' iterations. The number of group tests m is 
        set to 20 and the regularization parameter alpha to .001.

        Args:
            n (int): Number of individuals
            k (int): Number of infected
            thresholds (ndarray): Array of possible threshold values (tau)
            Monte (int): Number of Monte Carlo experiments

        """
        self.n, self.k, self.thresholds, self.Monte = n, k, thresholds, Monte\

    def run(self):
        fprarray = np.array([])
        tprarray = np.array([])
        for tau in self.thresholds:
            fpr_ = []
            tpr_ = []
            for _ in range(self.Monte):
                xpure = np.zeros([self.n,1])
                xpure[0:self.k] = 1
                np.random.shuffle(xpure)
                x = xpure 

                # Prediction
                xhat = SR(x,self.n,self.k,20,.001,tau = tau).xhat()

                # Error
                fpr_.append(fpr(x, xhat))
                tpr_.append(sensitivity(x, xhat))
            fprarray = np.append(fprarray, sum(fpr_) / len(fpr_))
            tprarray = np.append(tprarray, sum(tpr_) / len(tpr_))
        return fprarray.reshape(-1,1), tprarray.reshape(-1,1)
