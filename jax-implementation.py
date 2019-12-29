import jax.numpy as np
from jax.config import config
from jax import grad, jit, vmap

import numpy as npn
import numpy.matlib as nm

from functools import partial

def gram(kernel, xs):
  return vmap(lambda x: vmap(lambda y: kernel(x, y))(xs))(xs)

def rbf(x, y):
  return np.exp(-np.sum((x - y)**2))

class SVGD():
    def __init__(self):
        pass
    
    def svgd_kernel(self, theta):
        Kxy = gram(rbf, theta)
        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1, keepdims=True)
        dxkxy = dxkxy + theta * sumkxy  
        return (Kxy, dxkxy)
    

    def update(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')
        
        theta = npn.copy(x0) 
        
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):           
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(theta)  
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]  
            
            # adagrad 
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = npn.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad 
            
        return theta

class MVN:
    def __init__(self, mu, A):
        self.mu = mu
        self.A = A
    
    def dlnprob(self, theta):
        return -1*np.matmul(theta-nm.repmat(self.mu, theta.shape[0], 1), self.A)
    
if __name__ == '__main__':
    A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
    mu = np.array([-0.6871,0.8010])
    
    model = MVN(mu, A)
    
    x0 = npn.random.normal(0,1, [10,2]);
    theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=0.01)
    
    print ("ground truth: ", mu)
    print ("svgd: ", np.mean(theta,axis=0))
    
