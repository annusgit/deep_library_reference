

def bad():
    import numpy as np
    # arr = np.array([[-1, 2, 4],[1, -8, 7],[0, 0, -1]])
    # arr[arr < 0] = 0
    # print(arr)

    def dot(x,y):
        result = np.zeros(shape=(x.shape[0], y.shape[1]))
        for i in range(int(x.shape[0])):
            for j in range(int(y.shape[1])):
               result[i][j] = sum(x[i, :] * y[:, j])
        return result

    def new_mydot(x, y, result):
    #     y = np.transpose(np.asarray(y))
    #     x = np.transpose(np.asarray(x))
    #     x = x.transpose()
    #     result = np.zeros(shape=(x.shape[0], y.shape[1]))
        for i in range(int(x.shape[0])):
            for j in range(int(y.shape[1])):
                vector = x[i, :] * y[:, j]
                for k in range(len(vector)):
                    # print(vector[k])
                    result[i][j] += vector[k]

    arr1 = np.array([[-1, 2, 4],[1, -8, 7],[0, 0, -1]])
    arr2 = np.array([[-1, 5, 4],[3, 8, 17],[0, 0, -1]])

    print(np.dot(arr1,arr2))
    result = np.zeros(shape=(3,3))
    new_mydot(arr1,arr2,result)
    print(result)


import time as t
def timing_elapsed(function, *args):
    def wrapper(*args):
        start = t.clock()
        function(args[0],args[1])
        return t.clock() - start
    return wrapper


def test_jit():
    import numpy as np
    from numba import jit, float64, vectorize, guvectorize, int64

    # # @vectorize([float64(float64,float64)], target='cpu')
    # # def numba_adder(x, y):
    # #     result = np.zeros(shape=np.shape(x))
    # #     for i in range(x.shape[0]):
    # #         for j in range(x.shape[1]):
    # #             result[i][j] = x[i][j]*y[i][j]
    # #     return result
    #     # return x + y
    #
    # @guvectorize([float64[:,:](float64[:,:],float64[:,:])], '(m,n),(m,n)->(m,n)', target='cpu')
    # def my_add(x, y, result):
    #     result = x*y
    #
    # x = np.random.uniform(low=0,high=10,size=(256,256)).astype(dtype=np.float64)
    # y = np.random.uniform(low=1,high=22,size=(256,256)).astype(dtype=np.float64)
    #
    # @timing_elapsed
    # def numpy_add(x,y):
    #     np.multiply(x,y)
    #
    # # @timing_elapsed
    # # def numba_add(x,y):
    # #     numba_adder(x,y)
    #
    # @timing_elapsed
    # def adder(x,y,result):
    #     my_add(x,y,result)
    #
    #

    x = np.random.uniform(low=0,high=10,size=(256,256)).astype(dtype=np.float64)
    y = np.random.uniform(low=1,high=22,size=(256,256)).astype(dtype=np.float64)

    @guvectorize([int64[:,:],int64[:,:],int64[:,:]], '(m,n),(m,n)->(m,n)')
    def my_mult(x, y, result):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                result[i][j] = x[i][j]*y[i][j]

    for i in range(10):

        start = t.clock()
        result1 = np.multiply(x,y)
        # print(result1.shape)
        print('numpy took {}'.format(t.clock()-start))

        start = t.clock()
        result2 = np.zeros(shape=x.shape)
        start = t.clock()
        my_mult(x,y,result2)
        print('your\'s took {}'.format(t.clock()-start))
    #

test_jit()









