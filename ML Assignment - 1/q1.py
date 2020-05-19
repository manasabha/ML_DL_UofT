from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error

np.random.seed(0)

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        plt.xlabel(features[i])
        plt.ylabel('Price')
        plt.scatter(X[:,i],y)
    plt.tight_layout()
    plt.savefig('dependant_vs_independant.png')
    plt.show()

def fit_regression(X,Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    '''
    :param X: Design Matrix
    :param Y: Target Vector
    :return: Weight Matrix

    Solving for weight matrix using ((X^T.X)^-1)X^TY
    '''
    X_transpose = np.transpose(X)
    X_trans_mult_X = np.dot(X_transpose,X)
    X_mult_y = np.dot(X_transpose,Y)
    return np.linalg.solve(X_trans_mult_X,X_mult_y)

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}\n".format(features))
    np.set_printoptions(suppress=True)
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    no_of_data_entries   = X.shape[0]
    bias_to_append       = np.ones(no_of_data_entries)
    X_with_bias          = np.c_[bias_to_append,X]
    no_of_training_entries = int (0.8*no_of_data_entries)

    #Stores randomly generated indices to pick training and test set.
    index_list = np.random.choice(no_of_data_entries, no_of_training_entries, replace=False)

    # Use set operations, to get array of test indices
    set_index = set(index_list)
    set_all = set(np.arange(506))
    test_index = np.array(list(set_all-set_index))

    #Construct X_train, Y_train, X_test, Y_test
    x_train = np.take(X_with_bias, index_list, axis=0)
    y_train = y[index_list].reshape(no_of_training_entries,1)
    x_test = np.take(X_with_bias,test_index, axis=0)
    y_test = y[test_index].reshape((no_of_data_entries-no_of_training_entries),1)

    #Standardizing

    X_scale = X

    mean_array = (X_scale.mean(axis=0)).reshape(1,X_scale.shape[1])
    std_array = (np.std(X_scale,axis=0)).reshape(1,X_scale.shape[1])
    for i in range(X_scale.shape[0]):
        X_scale[i] = X_scale[i]-mean_array
        X_scale[i] = np.divide(X_scale[i],std_array)

    X_scale = np.c_[np.ones(X_scale.shape[0]),X_scale]
    x_train_scale = np.take(X_scale, index_list, axis=0)
    x_test_scale = np.take(X_scale,test_index, axis=0)


    #Fit Regression
    w = fit_regression(x_train, y_train)

    #Fit Scaled
    w_scale = fit_regression(x_train_scale,y_train)
    #Tabulate Features with Weights
    print ("features with weights for data\n {}\n".format(np.c_[features,w[1:]]))
    print ("features with weights for normalized data \n {}\n".format(np.c_[features,w_scale[1:]]))

    # Compute fitted values, MSE, etc.
    predictions = np.dot(x_test,w)
    predictions_scaled = np.dot(x_test_scale,w_scale)

    MSE = ((y_test-predictions)**2).mean()
    MSE_scale = ((y_test-predictions_scaled)**2).mean()
    R2_score = r2_score(y_test,predictions)
    R2_score_scale = r2_score(y_test,predictions_scaled)
    median_abs_error = median_absolute_error(y_test,predictions)
    median_abs_error_scale = median_absolute_error(y_test,predictions_scaled)

    print("Errors for Actual dataset\n\n")
    print ("MSE: {}".format(MSE))
    print ("RMSE: {}".format(MSE**0.5))
    print("R2_Score: {}".format(R2_score))
    print("median_abs_error: {}\n\n".format(median_abs_error))

    print("Errors for Standardized dataset\n\n")

    print ("MSE: {}".format(MSE_scale))
    print ("RMSE: {}".format(MSE_scale**0.5))
    print("R2_Score: {}".format(R2_score_scale))
    print("median_abs_error: {}".format(median_abs_error_scale))

if __name__ == "__main__":
    main()

