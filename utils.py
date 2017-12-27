import _pickle as cPickle
import gzip
import numpy as np
from matplotlib import pyplot as plt

def get_relevant_data(data):
    X, Y = [], []
    for idx, num in enumerate(data[1]):
        if num==0 or num==1:
            X.append(data[0][idx])
            Y.append(num)
    X = np.asarray(X)
    Y = np.asarray(Y)
    X = X.T
    Y = Y.reshape(1, Y.shape[0])
    
    # X.shape = (784, m)
    # Y.shape = (1, m)
    # m is number of images
    
    return (X, Y)

def load_data():
    '''
    MNIST dataset contains 28*28 pixel images of 0-9 digits
    Loads the .pkl.gz file, returns train and validation set for 0 and 1 images only.
    '''
    # Download dataset from https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    train = get_relevant_data(training_data)
    val = get_relevant_data(validation_data)
    return (train, val)

def plot_training(costs_train, costs_val):
    epochs = range(len(costs_train))
    plt.plot(epochs, costs_train)
    plt.plot(epochs, costs_val)
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Cost')
    plt.show()

if __name__ == "main":
    pass