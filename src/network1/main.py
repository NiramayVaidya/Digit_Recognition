import mnist_loader
import network
import numpy as np
import json

net = network.Network([784, 30, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#net.SGD(training_data, 30, 10, 3, test_data = None)
net.SGD(training_data, 30, 10, 3, test_data = test_data)
'''
np.save("trained_weights.npy", net.weights)
np.save("trained_biases.npy", net.biases)
'''
fw = open('trained_weights.json', 'w')
fb = open('trained_biases.json', 'w')
'''
Since a numpy array/ndarray in not json serializable,
converting it to a list
'''
json.dump([w.tolist() for w in net.weights], fw)
json.dump([b.tolist()for b in net.biases], fb)
fw.close()
fb.close()









