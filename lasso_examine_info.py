import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC


def gen_population(p = 200,
             N = 1000,
             design_err = 1,
             seed = 0):
    """
    seed controls the parameter and design matrix generation.
    the random number generator is reset for the error generation.
    """
    np.random.seed(seed)
    s = np.random.binomial(1, .1, p) #select which para elements are nonzero

    s0 = np.sum(s)
    s0    #this is the number of nonzero entries in the parameter vector
    
    _beta = np.random.normal(0, 5, p)   #simulate some betas
    beta = _beta * s                    #true parameter vector 

    s_cond = s != 0                     #bool sparsity structure
    beta_short = _beta[s_cond]          #build a reduced set
    beta_short    

    X = np.random.normal(0, design_err, (N, p))
    Xbeta = np.dot(X, beta)

    output = {'X' : X,
              'beta' : beta,
              'beta_short' : beta_short,
              's' : s,
              's0' : s0,
              's_cond' : s_cond,
              'p' : p,
              'N' : N,
              'Xbeta' : Xbeta}

    return output


def gen_response(sigma_err, Xbeta, **kwargs):
    N = kwargs.get('N')
    
    np.random.seed()
    err = np.random.normal(0, sigma_err, (N,))
    #Y = np.dot(X, beta) + err
    Y = Xbeta + err

    # clf = LassoLarsIC(criterion = 'aic')  #fit the lasso
    # clf.fit(X,Y)             
    # point_estimate = clf.coef_  #obtain the coefficient estimates

    return Y


def boot_coef(X, Y):
    """
    Takes original data and new sampling index.
    
    Returns coefficient vector. 
    """
    np.random.seed()
    N = X.shape[0]
    ind = np.random.choice(N, N)  #resample step
    clf = LassoLarsIC(criterion = 'aic')
    clf.fit(X[ind,],Y[ind])
    point_estimate = clf.coef_
    return point_estimate


#def boot_ci(M, response, X, s0, s_cond, beta_short, **kwargs, *,alpha = .05):
def boot_ci(M, response, alpha = .05, **kwargs):
    """
    M - Number of bootstrap samples
    ** Just pass the **gen_population output after M, as in
    boot_ci(100, **gen_population)
    """
    X = kwargs.get('X')
    s0 = kwargs.get('s0')
    s_cond = kwargs.get('s_cond')
    beta_short = kwargs.get('beta_short')
    
    Y = response
    boot = pd.DataFrame(index = range(M), data = np.zeros((M, s0)))
    for i in range(M):
        boot.iloc[i,] = boot_coef(X, Y)[s_cond]
    ci = boot.quantile(q = (alpha / 2, 1 - alpha / 2), axis = 0) #make the CIs
    
    ci_contain = np.zeros(s0) #preallocate vector which is 1 if CI
                            #contains true parameter value
    for i in range(s0):
        if ci.iloc[0, i] < beta_short[i] and beta_short[i] < ci.iloc[1, i]:
            ci_contain[i] = 1
    return ci, ci_contain



p = 200
N = 1000
design_err = 5
sigma_err = 2
H = 100 #number of simulations of the error vector and subsequent recovery of parameter estimates
M = 100 #number of bootstrap samples


#generate the population; this remains fixed for the rest of the script
population = gen_population(p = p,
                            N = N,
                            design_err = design_err)

contains = np.zeros((H, population['s0']))
for i in range(H):
    print(i,H)
    response = gen_response(sigma_err = sigma_err, **population)
    *_, contains[i,] = boot_ci(M = M, response = response, alpha = .05, **population)

observed_coverage = np.mean(contains, axis = 1)

##make wrapper function
def wrapper(M = 100, alpha = .05, **kwargs):
    sigma_err = kwargs.get('sigma_err')
    population = kwargs.get('population')
    
    reponse = gen_response(sigma_err = sigma_err)
    *_, contains[i,] = boot_ci(M = M, response = response, alpha = alpha, **population)

wrapper(M = 10, alpha = .05, population = population)
