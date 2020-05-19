import numpy as np
from sklearn.datasets import load_boston
import  matplotlib.pyplot as plt
from scipy.spatial import distance
BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    

def variance(vec):
    mean_vec = np.mean(vec)
    return (np.sum(np.divide(((vec-mean_vec)**2),len(vec))-1))

def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    x_transpose = np.transpose(X)
    cost        = (y-np.dot(X,w))
    gradient_full = np.divide(np.dot(x_transpose,cost),X.shape[0])
    return (-gradient_full)

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    np.set_printoptions(suppress=True)

    batch_sampler = BatchSampler(X, y, BATCHES)
    # Example usage
    K = 500
    mega_batch_avg = np.zeros(X.shape[1])

    # Calculating avg gradient over K iterations for batch size 50
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b, y_b, w)
        mega_batch_avg = mega_batch_avg+batch_grad

    mega_batch_avg = np.divide(mega_batch_avg,K)
    full_grad = lin_reg_gradient(X, y, w)


    square_distance = np.sum(((full_grad-mega_batch_avg)**2))

    print("Square Distance {} ".format(square_distance))
    print(full_grad.shape)
    cosine_similarity_for_full_and_batch = cosine_similarity(full_grad,mega_batch_avg)
    print ("Cosine Similiarity {}".format(cosine_similarity_for_full_and_batch))
    index_of_interest = np.random.choice(X.shape[1])
    log_m = []
    actual_variance=[]

    #Calculating variance over m [1-400], K =500
    for m in range (1,401):
        batch_sampler = BatchSampler(X, y, m)
        number_list_for_variance = []
        for i in range(K):
            X_b, y_b = batch_sampler.get_batch()
            batch_grad = lin_reg_gradient(X_b, y_b, w)
            number_list_for_variance.append(batch_grad[index_of_interest])
        log_m.append(np.log(m))
        actual_variance.append(variance(number_list_for_variance))
    log_actual_variance = np.log(actual_variance)
    plt.xlabel("log(m)")
    plt.ylabel(("log(var)"))
    plt.plot(log_m,log_actual_variance)
    plt.savefig("q3_logm_logvariance.png")
    plt.show()


if __name__ == '__main__':
    main()
