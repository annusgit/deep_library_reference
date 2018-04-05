

from __future__ import print_function
from __future__ import division

from utilities import Data
import graph_ops as go
import train
import numpy as np

def main():
    manager = Data()
    num_examples = 10**2
    max_val = 1
    train_batch_size = 32
    train_size = int(num_examples / 2)
    eval_size = int(num_examples / 2)
    examples, labels = manager.create_data_set(num_of_examples=num_examples, max_val=max_val,
                                   discriminator=lambda x: max_val * (1/(1+np.exp(-x))+1/(1+np.exp(x**2)))-max_val/2,
                                   one_hot=True, plot_data=False, load_saved_data=False, filename='dataset.npy')

    # this creates the input value
    graph = go.GRAPH()
    graph.get_global_graph()

    X = go.inputs()
    y = go.inputs()

    W_1 = go.Matrix(initial_val=np.random.uniform(low=-0.1,high=0.1,size=[128,2]))
    b_1 = go.Matrix(initial_val=np.ones(shape=[train_batch_size,128]))

    W_2 = go.Matrix(initial_val=np.random.uniform(low=-0.1,high=0.1,size=[256,128]))
    b_2 = go.Matrix(initial_val=np.ones(shape=[train_batch_size,256]))

    W_3 = go.Matrix(initial_val=np.random.uniform(low=-0.1,high=0.1,size=[128,256]))
    b_3 = go.Matrix(initial_val=np.ones(shape=[train_batch_size,128]))


    W_4 = go.Matrix(initial_val=np.random.uniform(low=-0.1,high=0.1,size=[2,128]))
    b_4 = go.Matrix(initial_val=np.ones(shape=[train_batch_size,2]))

    features = go.dot(X, W_1)
    features = go.add(features, b_1)
    first = go.sigmoid(features)
    print('done')

    features = go.add(go.dot(first, W_2), b_2)
    features = go.sigmoid(features)
    print('done')

    features = go.add(go.dot(features, W_3), b_3)
    features = go.sigmoid(features)
    # features = go.add(features, first)
    print('done')

    features = go.add(go.dot(features, W_4), b_4)
    features = go.sigmoid(features)
    print('done')

    loss = go.multiply(features, y)
    # print(loss.__class__)
    minimization_op = train.GradientDescentOptimizer(learning_rate=0.03).minimize(loss)
    graph.compile(this_operation=minimization_op, verbose=True)

    for _ in range(1):
        graph.run(minimization_op, feed_dict={X:examples[0:train_batch_size,:], y:labels[0:train_batch_size]})

        # print(features.output.shape)
        print(np.sum(np.sum(loss.output))/train_batch_size)


if __name__ == '__main__':
    main()



