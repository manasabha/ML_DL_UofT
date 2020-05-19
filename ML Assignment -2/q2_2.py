'''
Question 2.2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import det
from scipy.special import logsumexp

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for k in range(10):
        means[k] = np.mean(train_data[np.nonzero(train_labels==k)[0]],axis=0)
    # Compute means
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class 
    '''
    covariances = np.zeros((10, 64, 64))
    count_class = np.zeros(10).astype(int).reshape(10,1)
    mean_diff = np.zeros((700,64))
    means = compute_mean_mles(train_data,train_labels)
    for k in range(10):
        count_class[k] = np.count_nonzero(train_labels==k)
        mean_diff = train_data[np.nonzero(train_labels==k)[0]]-means[k]
        prod = np.zeros((64, 64))
        for i in range(mean_diff.shape[0]):
            mean_ind = mean_diff[i].reshape(64,1)
            mean_ind_t = np.transpose(mean_ind)
            prod = prod + np.dot(mean_ind,mean_ind_t)
        covariances[k] = prod/mean_diff.shape[0]
    return (covariances+0.01*np.identity(64))


def plot_cov_diagonal(covariances):
    # Plot the diagonal of each covariance matrix side by side
    cov_list = []
    for i in range(10):
        cov_diag = np.diag(covariances[i])
        # ...
        cov_list.append(cov_diag.reshape(8, 8))
    all_concat = np.concatenate(cov_list, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    covariance is 10*64*64
    mean is 10*64
    digits = train 7000*64
    for each class
    '''
    gen_prob = np.zeros((digits.shape[0],10))
    for k in range(10):
        sig_inv_k = inv(covariances[k])
        det_k = det(covariances[k])
        Z = -32*(np.log(2*np.pi))-(0.5*np.log(det_k))
        for i in range(digits.shape[0]):
            new_mat = (digits[i]-means[k])
            gen_prob[i][k]= (Z-(0.5*(np.dot(np.transpose(new_mat),np.dot(sig_inv_k,new_mat)))))
    return gen_prob

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    returns only likelyhood*prior. there is no denominator.
    I'll calculate this value in avg cond likelyhood, cuz that is only place where
    we need actual values.
    '''

    gen_likely = generative_likelihood(digits,means,covariances)
    return (gen_likely+np.log(0.1))

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    to_remove = logsumexp(cond_likelihood,axis=1).reshape(cond_likelihood.shape[0],1)
#    to_remove = np.log(np.sum(np.exp(cond_likelihood), axis=1)).reshape(cond_likelihood.shape[0], 1)
    actual_cond_likelihood = cond_likelihood-to_remove
    avg_true_label = 0

    for k in range(10):
        avg_true_label = avg_true_label+np.sum(actual_cond_likelihood[np.nonzero(labels==k)[0]][:,k])
    return (avg_true_label/labels.shape[0])


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    predictions = np.argmax(cond_likelihood, axis=1)
    # Compute and return the most likely class
    return predictions

def classification_accuracy(predicted_labels, actual_labels):
    '''
    Calculate error rate using 0-1 loss
    '''

    predicted_labels = predicted_labels.reshape(predicted_labels.shape[0],1).astype(float)
    actual_labels = actual_labels.reshape(actual_labels.shape[0], 1).astype(float)

    return ((np.count_nonzero(predicted_labels-actual_labels)/predicted_labels.shape[0])*100)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    np.set_printoptions(suppress=True)
    plot_cov_diagonal(covariances)
    # Evaluation
    train_pred = classify_data(train_data,means,covariances)
    test_pred = classify_data(test_data,means,covariances)

    train_acc = classification_accuracy(train_pred,train_labels)
    test_acc = classification_accuracy(test_pred,test_labels)

    print("train acc {}% test acc {}%".format(100-train_acc,100-test_acc))
    np.set_printoptions(suppress=True)
    print("avg cond likelihoold for train ",avg_conditional_likelihood(train_data,train_labels,means,covariances))
    print("avg cond likelihoold for test ",avg_conditional_likelihood(test_data,test_labels,means,covariances))

if __name__ == '__main__':
    main()