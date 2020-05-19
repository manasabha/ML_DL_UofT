'''
Question 2.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point

        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
#        train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        digit = None
        orig_indices = np.argsort(self.l2_distance(test_point))[:k]
        find_dups = np.bincount(np.take(self.train_labels,orig_indices).astype(int))

        return float(digit)
        '''

def all_distance_calculation(train_data, train_labels, test_data):
    knn_for_distance = KNearestNeighbor(train_data,train_labels)
    '''
    All_test_distance is a 4000*15 array where for each test point
    distance are calculated and stored for future use.
    Store only labels for top 15 distances, as that is the max number of neighbors
    used through out the code.
    Even for breaking ties, we will reduce k, not increase.
    '''
    all_test_labels = np.zeros((4000,15))

    for test_index in range(test_data.shape[0]):
        test_indices = np.argsort(knn_for_distance.l2_distance(test_data[test_index]))[:15]
        all_test_labels[test_index] = np.take(train_labels,test_indices)

    '''
    Calculate and keep distances for all training points, to calculate train accuracy.
    We can't take just top 15, as in k-fold validation, there is a chance that, top 5 might be in validation fold.
    In that case, for every index which can fold in the validation fold,
    i.e indices of range (n-1)*700 to n*700, if there are any indices 
    all_train_label is dict of list with 2 arrays. one is for non kfold, second is used in k fold.
    '''

    all_train_distances = np.zeros((7000,7000))

    for train_index in range(train_data.shape[0]):
        all_train_distances[train_index] = knn_for_distance.l2_distance(train_data[train_index]) # has all indices based on dist sorted

    return all_test_labels,all_train_distances

def classification_accuracy(predicted_labels, actual_labels):
    '''
    Calculate error rate using 0-1 loss
    '''

    predicted_labels = predicted_labels.reshape(predicted_labels.shape[0],1).astype(float)
    actual_labels = actual_labels.reshape(actual_labels.shape[0], 1).astype(float)

    return ((np.count_nonzero(predicted_labels-actual_labels)/predicted_labels.shape[0])*100)

def majority_label(all_labels,k):
    neigh = k
    final_labels = []
#    print ("ml", all_labels.shape,k)
    for ind in range(all_labels.shape[0]):
        count = np.bincount(all_labels[ind,:neigh].astype(int))
#        print (ind,neigh,count)
        while np.count_nonzero(count == count[np.argmax(count)]) > 1:
#            print ("here")
            neigh = neigh-1
            count = np.bincount(all_labels[ind,:neigh].astype(int))
        neigh = k
        final_labels.append(np.argmax(count))
#        print("neigh",neigh)
    return np.array(final_labels)



def cross_validation(train_data,train_labels,all_train_distances):
    k_range = np.arange(1,16)
    test_error_for_k = np.zeros(15)

    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        test_error_for_fold = 0
        kf = KFold(n_splits=10)
        for train, test in kf.split(train_data):
            new_train_labels = train_labels[train]
            new_test_labels = train_labels[test]
            labels_for_test = np.take(new_train_labels,np.argsort(np.take(np.take(all_train_distances,test, axis=0),train,axis=1),axis=1))[:,:k]
            final_labels_for_test = majority_label(labels_for_test,k)
            cv_test_error = classification_accuracy(final_labels_for_test,new_test_labels)
            test_error_for_fold = test_error_for_fold+cv_test_error
        test_error_for_k[k-1] = test_error_for_fold/10
    min_k = np.argsort(test_error_for_k)[0]
    min_test_error = test_error_for_k[min_k]
    print(test_error_for_k)
    print(100-test_error_for_k)
    train_error_for_fold = 0
    '''
    Since we don't use training error to pick k, we can calculate training error for only best k.
    '''
    kf = KFold(n_splits=10)
    for train, test in kf.split(train_data):
        new_train_labels = train_labels[train]
        labels_for_train = np.take(new_train_labels,
                                   np.argsort(np.take(np.take(all_train_distances, train, axis=0), train, axis=1),
                                              axis=1))[:, :min_k+1]
        final_labels_for_train = majority_label(labels_for_train, min_k+1)
        train_error = classification_accuracy(final_labels_for_train, new_train_labels)
        train_error_for_fold = train_error_for_fold + train_error

    min_train_error = train_error_for_fold / 10

    return (min_k,min_test_error,min_train_error)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_labels = train_labels.reshape(train_labels.shape[0],1)
    test_labels  = test_labels.reshape(test_labels.shape[0],1)
    knn = KNearestNeighbor(train_data, train_labels)

    all_test_labels, all_train_distances = all_distance_calculation(train_data, train_labels, test_data)
    for k in (1,15):
        test_majority_labels  = majority_label(all_test_labels,k)
        test_error = classification_accuracy(test_majority_labels,test_labels)
        indices_for_train = np.argsort(all_train_distances, axis=1)[:, :k]
        final_labels_for_train = majority_label(np.take(train_labels, indices_for_train), k)
        train_error = classification_accuracy(final_labels_for_train,train_labels)
        print("For k {} Train error is {}% and Test Error is {}%".format(k,100-train_error,100-test_error))

    best_k, cv_error_best_k, train_error_best_k = cross_validation(train_data, train_labels, all_train_distances)
    best_k = best_k+1
    test_majority_labels_best_k = majority_label(all_test_labels, best_k)
    test_error_best_k = classification_accuracy(test_majority_labels_best_k, test_labels)

    print("best k {} train error {}% cv error {}% test error {}%".format(best_k,100-train_error_best_k,100-cv_error_best_k,100-test_error_best_k))
if __name__ == '__main__':
    main()