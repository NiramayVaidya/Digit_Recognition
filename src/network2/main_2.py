import mnist_loader
import network2
import overfitting
import json
import random
import numpy as np

#random.seed(12345678)
#np.random.seed(12345678)
training_data, validation_data, test_data = \
    mnist_loader.load_data_wrapper()
net = network2.Network([784, 100, 10])
#net.large_weight_initializer()
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, \
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, \
        monitor_training_cost=True, monitor_training_accuracy=True, early_stopping_n=10)

'''
evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = \
        net.SGD(list(training_data)[:1000], 10, 10, 0.5, lmbda=0.1, evaluation_data=test_data, \
        monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, \
        monitor_training_cost=True, monitor_training_accuracy=True, early_stopping_n=10)
'''

#len(list(test_data)) doesn't work
print('Evaluation accuracy: {}'.format(evaluation_accuracy[-1] / 10000  * 100))

net.save('network2_params.json')

f = open('network2_overfitting.json', 'w')
json.dump([evaluation_cost, evaluation_accuracy, training_cost, \
        training_accuracy], f)
f.close()
