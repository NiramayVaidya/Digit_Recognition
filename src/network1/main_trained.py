import network_trained
import cv2
import numpy as np

#filename = '../data/mnistasjpg/trainingSample/img_6.jpg'
filename = '../data/6-1.jpg'
print(filename)
image = cv2.imread(filename, 0)
image = cv2.resize(image, (28, 28))
#cv2.imwrite('../data/7-resized.png', image)
image = image.reshape(784, 1)
net_trained = network_trained.Network_trained([784, 100, 10])
output = net_trained.feedforward(image)
print(output)
output = np.argmax(output)
print(output)
