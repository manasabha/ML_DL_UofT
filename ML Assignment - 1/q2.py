# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017
#h
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((N,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

# Randomize the data using generated indices
x  = x[idx]
y  = y[idx]

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        #Adjust y_Test dimensions to predictions dimension, to do subtraction correctly
        y_test = np.reshape(y_test, (predictions.shape[0],predictions.shape[1]))
        losses[j] = ((predictions - y_test)**2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    #Calculating terms needed for constructing the diagonal matrix A
    diff_array = l2(x_train, test_datum).reshape(x_train.shape[0],1)
    new_diff = np.divide(diff_array , 2 * (tau ** 2))
    sum_tot_exp = np.exp(logsumexp(-new_diff))
    a=[]
    for distance in new_diff:
        a.append(np.divide(np.exp(-distance),sum_tot_exp))
    np.set_printoptions(suppress=True)
    A = np.diag(np.reshape(a, (len(a),)))

    # Using x_train, A, y_train to calculate Weights w
    x_transpose    = np.transpose(x_train)
    regularization = lam*np.identity(x_train.shape[1],dtype=int)
    term_a_1 = x_transpose@A@x_train
    term_a = term_a_1+regularization
    term_b = x_transpose@A@y_train
    w = np.linalg.solve(term_a,term_b)

    # Predict the test value
    predict = np.dot(test_datum,w)

    return (predict)


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO
    partition_size = int(x.shape[0] / k)
    losses_list = []
    for i in range (k):
        low_index = i*partition_size
        high_index = low_index+partition_size
        print("fold: {} start index of test {}".format(i,low_index))
        x_train_1, x_test , x_train_2 = np.array_split(x, (low_index,high_index), axis=0)
        y_train_1, y_test, y_train_2 = np.array_split(y, (low_index, high_index), axis=0)
        x_train = np.concatenate((x_train_1, x_train_2), axis=0)
        y_train = np.concatenate((y_train_1, y_train_2), axis=0)
        losses_list.append(run_on_fold(x_test,y_test,x_train,y_train,taus))

    losses = np.array(losses_list)
    return (losses.mean(axis=0))
    ## TODO


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.xlabel("tau")
    plt.ylabel("avg loss")
    plt.plot(taus,losses)
    plt.tight_layout()
    plt.savefig("q2_avg_loss_vs_tau.png")
    plt.show()
    print("min loss = {}".format(losses.min()))

