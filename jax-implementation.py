import jax.numpy as np
from jax.config import config
from jax.experimental import optimizers
from jax import grad, jit, vmap

import numpy as npn
import numpy.matlib as nm

from functools import partial

def lnprob(theta):
  return -1*np.matmul(theta-nm.repmat(mu, theta.shape[0], 1), A)

def gram(kernel, xs):
  return vmap(lambda x: vmap(lambda y: kernel(x, y))(xs))(xs)

def rbf(x, y):
  return np.exp(-np.sum((x - y)**2))

def svgd_kernel(theta):
    Kxy = gram(rbf, theta)
    k_grad = grad(rbf)
    return Kxy, k_grad(theta, theta)

@jit
def step(i, opt_state, theta):
    lnpgrad = lnprob(theta)
    kxy, dxkxy = svgd_kernel(theta)
    grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0] 
    
    return opt_update(i, -grad_theta, opt_state)

### Parameters 
A = np.array([[0.2260,0.1652],[0.1652,0.6779]])
mu = np.array([-0.21,0.9010])
x0 = npn.random.normal(0,1, [10,2]);

print(mu, 'ground truth')

step_size = 1e-2
alpha = 0.9

### Jax

theta = npn.copy(x0)
opt_init, opt_update, get_params = optimizers.adagrad(step_size=step_size, momentum=alpha)
opt_state = opt_init(theta)

for iter in range(1000):
    theta = get_params(opt_state)
    opt_state = step(iter, opt_state, theta)

print(np.mean(get_params(opt_state), axis=0), 'jax')

### Manual Adagrad 

theta = npn.copy(x0)
fudge_factor = 1e-6
historical_grad = 0

for iter in range(1000):     
    lnpgrad = lnprob(theta)
    # calculating the kernel matrix
    kxy, dxkxy = svgd_kernel(theta)  
    grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

    # if iter % 100 == 0: print(np.mean(theta, axis=0))
    
    # adagrad 
    if iter == 0:
        historical_grad = historical_grad + grad_theta ** 2
    else:
        historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
    adj_grad = npn.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
    theta = theta + step_size * adj_grad 

print(np.mean(theta, axis=0), 'manual')
