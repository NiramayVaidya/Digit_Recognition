import json
import random
import sys
sys.path.append('../src/')
import mnist_loader
import network2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as \
        FigureCanvas
import numpy as np
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

def make_plots(filename, num_epochs, vbox, 
               training_cost_xmin=0, 
               test_accuracy_xmin=0, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0,
               training_set_size=1000):
    f = open(filename, "r")
    test_cost, test_accuracy, training_cost, training_accuracy \
        = json.load(f)
    f.close()

    epochs_post_early_stopping = len(training_cost)
    '''
    while num_epochs - len(training_cost) > 0:
        training_cost.append(0.0)
        training_accuracy.append(0.0)
        test_cost.append(0.0)
        test_accuracy.append(0.0)
    '''
    plot_training_cost(training_cost, epochs_post_early_stopping, training_cost_xmin, vbox)
    plot_test_accuracy(test_accuracy, epochs_post_early_stopping, test_accuracy_xmin, vbox)
    plot_test_cost(test_cost, epochs_post_early_stopping, test_cost_xmin, vbox)
    plot_training_accuracy(training_accuracy, epochs_post_early_stopping, 
                           training_accuracy_xmin, training_set_size, vbox)
    plot_overlay(test_accuracy, training_accuracy, epochs_post_early_stopping,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size, vbox)

def plot_training_cost(training_cost, num_epochs, training_cost_xmin, vbox):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
            training_cost[training_cost_xmin:num_epochs], \
                    color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    canvas = FigureCanvas(fig)
    canvas.set_size_request(1400, 600)
    vbox.pack_start(canvas, True, True, 0)

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin, vbox):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
            [accuracy/100.0 
             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    canvas = FigureCanvas(fig)
    canvas.set_size_request(1400, 600)
    vbox.pack_start(canvas, True, True, 0)

def plot_test_cost(test_cost, num_epochs, test_cost_xmin, vbox):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs), 
            test_cost[test_cost_xmin:num_epochs],
            color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    canvas = FigureCanvas(fig)
    canvas.set_size_request(1400, 600)
    vbox.pack_start(canvas, True, True, 0)

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size, vbox):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
            color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    canvas = FigureCanvas(fig)
    canvas.set_size_request(1400, 600)
    vbox.pack_start(canvas, True, True, 0)

def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size, vbox):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy/100.0 for accuracy in test_accuracy], 
            color='#2A6EA6',
            label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs), 
            [accuracy*100.0/training_set_size 
             for accuracy in training_accuracy], 
            color='#FFA933',
            label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([90, 100])
    plt.legend(loc="lower right")
    #plt.show()
    canvas = FigureCanvas(fig)
    canvas.set_size_request(1400, 600)
    vbox.pack_start(canvas, True, True, 0)
