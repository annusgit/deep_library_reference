

from __future__ import print_function
from __future__ import division
from collections import OrderedDict

graph = {
    'A': [],
    'B': [],
    'C': ['A'],
    'D': ['C'],
    'E': ['B'],
    'F': ['E', 'D'],
    'G': ['F', 'A'],
    'H': ['G']
}

class orderedset(OrderedDict):

    """
        A small implementation of an ordered set needed

    """
    def __init__(self):
        super(orderedset, self).__init__()

    def add(self, val):
        # print('added {}'.format(val))
        self[val] = None

    def get_list(self):
        return self.keys()



def get_order(graph, operation):
    """

    :param graph: graph object
    :param operation: operation for which we want to determine the feedforward order
    :return: the post-order list of operations to perform
    """
    postordered = orderedset() # it will be a set

    def postorder(graph, operation):

        "also input the final operation that you want to compute!!!"

        for op in graph[operation]:
            postorder(graph, op)
        postordered.add(operation)

    postorder(graph, operation)
    return postordered.get_list()


def main():

    # print(type(graph))
    # print(graph['G'])
    postordered = get_order(graph=graph, operation='H')
    print(postordered)
    # print([e for e in reversed(postordered)])




if __name__ == "__main__":

    main()