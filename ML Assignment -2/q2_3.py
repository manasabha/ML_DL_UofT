'''
Question 2.3 Skeleton Code

Here you should implement and evaluate the Naive Bayes classifier.
'''
import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.special import logsumexp


def binarize_data(pixel_values):
    '''
    Binarize the data by thresholding around 0.5
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    count_class = np.zeros(10).astype(int).reshape(10,1)
    count_label_per_class= np.zeros((10,64)).astype(int)
    for k in range(10):
       count_class[k] = np.count_nonzero(train_labels==k)
       count_label_per_class[k] = np.count_nonzero(train_data[np.nonzero(train_labels==k)[0]],axis=0)
    eta = (count_label_per_class+1)/(count_class+2)
    return (eta)

def plot_images(class_images):
    '''
    Plot each of the images corresponding to each class side by side in grayscale
    '''
    im_list = []
    for i in range(10):
        img_i = class_images[i]
        im_list.append(img_i.reshape(8, 8))
    all_concat = np.concatenate(im_list, 1)
    plt.imshow(all_concat, cmap='gray')
    plt.show()

def generate_new_data(eta):
    '''
    Sample a new data point from your generative distribution p(x|y,theta) for
    each value of y in the range 0...10

    Plot these values
    '''
    generated_data = np.zeros((10, 64))
    for k in range(eta.shape[0]):
        generated_data[k] = np.select([eta[k]>0.4],[eta[k]])
    plot_images(generated_data)

    generated_data = np.zeros((10, 64))
    for k in range(eta.shape[0]):
        generated_data[k] = np.select([eta[k]>0.5],[eta[k]])
    plot_images(generated_data)

    new_data = np.zeros((10,64))
    for k in range(eta.shape[0]):
        for i in range(eta.shape[1]):
            new_data[k][i] = np.random.binomial(1,eta[k][i],1)
    plot_images(new_data)

def generative_likelihood(bin_digits, eta):
    '''
    Compute the generative log-likelihood:
        log p(x|y, eta)

    Should return an n x 10 numpy array
    bin_digits will be train/ test data
    eta is 10*64, change to 64*10
    '''
    eta_t = np.transpose(eta)
    gen_likely_1 = np.dot(bin_digits,np.log(eta_t))
    gen_likely_2 = np.dot((1-bin_digits),np.log(1-eta_t))
    gen_likely = gen_likely_1+gen_likely_2
    return gen_likely

def conditional_likelihood(bin_digits, eta):
    '''
    Compute the conditional likelihood:

        log p(y|x, eta)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    bin_digits = new x on which we need to predict class.
    '''
    gen_likely = generative_likelihood(bin_digits,eta)
    return gen_likely+np.log(0.1)

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, eta) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    to_remove = logsumexp(cond_likelihood, axis=1).reshape(cond_likelihood.shape[0], 1)
    #    to_remove = np.log(np.sum(np.exp(cond_likelihood), axis=1)).reshape(cond_likelihood.shape[0], 1)
    actual_cond_likelihood = cond_likelihood - to_remove
    avg_true_label = 0

    for k in range(10):
        avg_true_label = avg_true_label + np.sum(actual_cond_likelihood[np.nonzero(labels == k)[0]][:, k])
    return (avg_true_label / labels.shape[0])


def classify_data(bin_digits, eta):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
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
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)
    train_labels = train_labels.reshape(train_labels.shape[0],1)
    test_labels = test_labels.reshape(test_labels.shape[0],1)

    # Fit the model
    eta = compute_parameters(train_data, train_labels)
    # Evaluation
    plot_images(eta)

    generate_new_data(eta)
    preds_train = classify_data(train_data,eta)
    preds_test = classify_data(test_data,eta)

    train_error = classification_accuracy(preds_train,train_labels)
    test_error = classification_accuracy(preds_test,test_labels)

    print("train error {} test error {}".format(100-train_error,100-test_error))
    np.set_printoptions(suppress=True)
    print("avg cond likelihoold for train ", avg_conditional_likelihood(train_data,train_labels,eta))
    print("avg cond likelihoold for test ",avg_conditional_likelihood(test_data,test_labels,eta))

if __name__ == '__main__':
    main()
