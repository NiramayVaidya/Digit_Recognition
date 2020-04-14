import numpy as np
import json

class Network_trained(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        '''
        self.biases = np.load("trained_biases.npy")
        self.weights = np.load("trained_weights.npy")
        '''
        fw = open('trained_weights.json', 'r')
        fb = open('trained_biases.json', 'r')
        '''
        since weights and biases have been stored in the form of lists,
        converting them to numpy arrays/ndarrays
        '''
        self.weights = json.load(fw)
        self.weights = [np.array(w) for w in self.weights]
        self.biases = json.load(fb)
        self.biases = [np.array(b) for b in self.biases]
        fw.close()
        fb.close()

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
