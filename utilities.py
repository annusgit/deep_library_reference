

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as pl
import numpy as np
import os


"""
    this file contains some useful utility functions
"""


def return_postorder_traversal_order(this_op, this_class):
    """
        Performs a post-order traversal, returning a list of nodes
        in the order in which they have to be computed

        Args:
           operation: The operation to start traversal at
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, this_class):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(this_op)
    return nodes_postorder



class Data(object):
    """
        will return some dummy dataset for testing the network
    """

    def __init__(self):
        pass

    def create_data_set(self, **kwargs):
        generate_data = True
        if 'load_saved_data' in kwargs.keys():
            if kwargs['load_saved_data']:
                if os.path.exists(kwargs['filename']):
                    print('log: loading saved filename {}'.format(kwargs['filename']))
                    X, y, discriminator = np.load(kwargs['filename'])
                    generate_data = False
                else:
                    print('log: file does not exist, creating new data...')
        if generate_data:
            x_vals = 2*kwargs['max_val']*np.random.rand(kwargs['num_of_examples'])-kwargs['max_val']
            x_t = np.arange(-kwargs['num_of_examples']/2,kwargs['num_of_examples']/2)/(kwargs['num_of_examples']/10)
            # print(x_t)
            X = np.asarray((x_t, x_vals))
            discriminator = np.asarray([kwargs['discriminator'](t) for t in sorted(x_t)]).transpose()
            # print(discriminator.shape)
            # np.random.shuffle(discriminator) # just do this to get a better plot
            y = np.zeros(shape=(kwargs['num_of_examples']))
            y[x_vals > discriminator] = 1
            y = y.astype(np.int32)
        if 'save_dataset' in kwargs.keys():
            np.save(kwargs['filename'], (X, y, discriminator))
            print('log: dataset saved as {}'.format(kwargs['filename']))

        if kwargs['plot_data']:
            self.__plot(X.transpose(), discriminator)

        # must do this shuffling to make a better dataset
        full_deck = np.concatenate((X.transpose(), y.reshape((y.shape[0], 1))), axis=1)
        np.random.shuffle(full_deck)
        y = full_deck[:, 2].astype(np.int32)
        X = full_deck[:, 0:2]
        # print(full_deck.shape)

        if kwargs['one_hot']:
            y = self.__one_hot(y, 1)

        return X, y

    def __one_hot(self, arr, max_val):
        new_arr = np.zeros(shape=(arr.shape[0], max_val+1))
        new_arr[range(arr.shape[0]), arr] = 1
        return new_arr

    def __plot(self, X, disc):
        # will plot only 100 values from the dataset to show some distributions
        pl.figure('Data Distribution')
        # pl.title()
        objective_function = disc
        red = X[X[:,1] > objective_function]
        green = X[X[:,1] < objective_function]
        pl.scatter(red[:,0], red[:,1], color='r')
        pl.scatter(green[:,0], green[:,1], color='g')
        pl.scatter(X[:,0], disc, color='b', label='discriminator')
        pl.legend()
        pl.show()
        pass















