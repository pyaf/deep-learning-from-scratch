import _pickle as cPickle
import gzip
import numpy as np
from matplotlib import pyplot as plt

def load_data():
    '''
    MNIST dataset contains 28*28 pixel images of 0-9 digits
    Loads the .pkl.gz file, returns train and validation set for 0 and 1 images only.
    '''
    # Download dataset from https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    X_train, Y_train = training_data
    X_val, Y_val = validation_data
    X_train = X_train.T
    X_val = X_val.T
    Y_train = np.eye(10)[Y_train]    
    Y_val = np.eye(10)[Y_val]
    Y_train = Y_train.T
    Y_val = Y_val.T
    return ((X_train, Y_train), (X_val, Y_val))

def plot_training(costs_train, costs_val):
    epochs = range(len(costs_train))
    plt.plot(epochs, costs_train)
    plt.plot(epochs, costs_val)
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Cost')
    plt.show()

if __name__ == "main":
    pass