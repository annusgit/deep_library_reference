

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
from utilities import return_postorder_traversal_order # for forward propagation


class GRAPH(object):
    """
        This is the graph that will save all of the connections between the different layers
        It will be resposible for doing all the forward and backward props
        and it will be undirected because we want forward "and" backward props
        This graph will have a global graph object, that is itself, let's call it a global_graph
    """

    def __init__(self):
        # this is the most important thing in this graph, the adjacent nodes must be known

        self.inputs = []
        self.variables = []
        self.ops = []
        self.forward_traversal_list = [] # this will be used for forward pass
        self.backward_traversal_list = [] # this will be used for backward pass

        pass


    def get_global_graph(self):
        """
            this is the global graph that will be used
        """
        global global_graph
        global_graph = self


    def compile(self, this_operation, verbose=False):
        " this function will compile all of the graph, with respect to some function at the end of the network"
        self.forward_traversal_list = return_postorder_traversal_order(this_op=this_operation, this_class=ops)
        self.backward_traversal_list = reversed(self.forward_traversal_list)
        if verbose:
            for node in self.backward_traversal_list:
                if hasattr(node, "input_nodes"):
                    print(node.__class__, node.input_nodes)


    def run(self, this_operation, feed_dict):
        # will be used for performing forward feed
        # print(self.forward_traversal_list)
        for node in self.forward_traversal_list:
            if type(node) == inputs:
                # Set the node value to the placeholder value from feed_dict
                node.output = feed_dict[node]
            elif type(node) == Matrix:
                # Set the node value to the variable's value attribute
                node.output = node.value
            else:  # Operation
                # Get the input values for this operation from node_values
                node.inputs = [input_node.output for input_node in node.input_nodes]

                # Compute the output of this operation
                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        return this_operation.output


    def backprop(self):
        """
            this method will be responsible for performing the backward propagation through the network
        """

        # we shall work with a dictionary of gradients
        grads = {}



class inputs(object):
    """
        this class will be used as input nodes to the graph of the network
    """

    def __init__(self):

        # will have some consumers to itself
        self.consumers = []
        global_graph.inputs.append(self)

        pass


class Matrix(object):

    """
        used as a matrix variable
    """

    def __init__(self, initial_val=None):

        self.matrix = initial_val

        # obviously some ops will use its value
        self.consumers = []

        # just add this variable to the global graph
        global_graph.variables.append(self)
        pass



class ops(object):
    """
        This class contains all the ops possible to do with this library
    """
    def __init__(self, input_nodes=[]): # the input nodes will also be objects
        # each operation will have some inputs to it and will have some consumers that will use its output value
        self.input_nodes = input_nodes
        self.consumers = [] # the consumers will themselves fill up this list

        # now let's fill up the input_nodes with their consumer, that is this object itself!!!
        for node in self.input_nodes:
            node.consumers.append(self)

        global_graph.ops.append(self)

        pass


    # computes the output of this operation
    def compute(self):
        " method must be overriden in the child classes"

        pass


"""
    the following classes will inherit from the ops class and implement some real ops
"""

class add(ops):
    """Returns x + y element-wise.
    """

    def __init__(self, x, y):
        """Construct add

        Args:
          x: First summand node
          y: Second summand node
        """
        super(add, self).__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the add operation

        Args:
          x_value: First summand value
          y_value: Second summand value
        """
        self.inputs = [x_value, y_value]
        return x_value + y_value


class dot(ops):
    """Multiplies matrix a by matrix b, producing a * b.
    """

    def __init__(self, a, b):
        """Construct matmul

        Args:
          a: First matrix
          b: Second matrix
        """
        super(dot, self).__init__([a, b])

    def compute(self, a_value, b_value):
        """Compute the output of the matmul operation

        Args:
          a_value: First matrix value
          b_value: Second matrix value
        """
        self.inputs = [a_value, b_value]
        return a_value.dot(b_value)


class sigmoid(ops):
    """Returns the sigmoid of x element-wise.
    """

    def __init__(self, a):
        """Construct sigmoid

        Args:
          a: Input node
        """
        super(sigmoid, self).__init__([a])

    def compute(self, a_value):
        """Compute the output of the sigmoid operation

        Args:
          a_value: Input value
        """
        return 1 / (1 + np.exp(-a_value))


class softmax(ops):
    """Returns the softmax of a.
    """

    def __init__(self, a):
        """Construct softmax

        Args:
          a: Input node
        """
        super(softmax, self).__init__([a])

    def compute(self, a_value):
        """Compute the output of the softmax operation

        Args:
          a_value: Input value
        """
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]


class log(ops):
    """Computes the natural logarithm of x element-wise.
    """

    def __init__(self, x):
        """Construct log

        Args:
          x: Input node
        """
        super(log, self).__init__([x])

    def compute(self, x_value):
        """Compute the output of the log operation

        Args:
          x_value: Input value
        """
        return np.log(x_value)


class multiply(ops):
    """Returns x * y element-wise.
    """

    def __init__(self, x, y):
        """Construct multiply

        Args:
          x: First multiplicand node
          y: Second multiplicand node
        """
        super(multiply, self).__init__([x, y])

    def compute(self, x_value, y_value):
        """Compute the output of the multiply operation

        Args:
          x_value: First multiplicand value
          y_value: Second multiplicand value
        """
        self.inputs = [x_value, y_value]
        return x_value * y_value


class reduce_sum(ops):
    """Computes the sum of elements across dimensions of a tensor.
    """

    def __init__(self, A, axis=None):
        """Construct reduce_sum

        Args:
          A: The tensor to reduce.
          axis: The dimensions to reduce. If `None` (the default), reduces all dimensions.
        """
        super(reduce_sum, self).__init__([A])
        self.axis = axis

    def compute(self, A_value):
        """Compute the output of the reduce_sum operation

        Args:
          A_value: Input tensor value
        """
        return np.sum(A_value, self.axis)


class negative(ops):
    """Computes the negative of x element-wise.
    """

    def __init__(self, x):
        """Construct negative

        Args:
          x: Input node
        """
        super(negative, self).__init__([x])

    def compute(self, x_value):
        """Compute the output of the negative operation

        Args:
          x_value: Input value
        """
        return -x_value















