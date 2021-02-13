import numpy as np

def IC_sum_rate(H, p, var_noise):
    H = np.square(H)
    fr = np.diag(H)*p
    ag = np.dot(H,p) + var_noise - fr
    y = np.sum(np.log(1+fr/ag) )
    return y
def np_sum_rate(X,Y):
    avg = 0
    n = X.shape[0]
    for i in range(n):
        avg += IC_sum_rate(X[i,:,:],Y[i,:],1)/n
    return avg