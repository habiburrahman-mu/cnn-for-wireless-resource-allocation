import numpy as np

def feature_extraction(H,num_train,N):
    H_ii = np.zeros((num_train,N))
    H_ij = np.zeros((num_train,N,N))
    H_ij_T = np.zeros((num_train,N,N))
    D = np.zeros((num_train,N,N))
    for ii in range(num_train):
        diag_H = np.diag(H[ii,:,:])
        for jj in range(N):
            H_ii[ii,jj] = H[ii,jj,jj]
            H_ij[ii,jj,:] = H[ii,:,jj].T
            H_ij[ii,jj,jj] = 0
            H_ij_T[ii,jj,:] = H[ii,jj,:]
            H_ij_T[ii,jj,jj] = 0
            D[ii,jj,:] = diag_H
            D[ii,jj,jj] = 0
    return H_ii, H_ij, H_ij_T, D

