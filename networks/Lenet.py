from network import Network
from utils.utils import im2col

class Lenet(Network):
    """
    implementation of LeNet-5 which is an architecture in the field of CNN(convolutional neural networks)
    (http://yann.lecun.com/exdb/lenet/)
    (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
    (https://en.wikipedia.org/wiki/LeNet)

    architecture:
    INPUT -> [CONV -> POOL] * 2 -> CONV -> FC -> OUTPUT

    details:
    INPUT: 2DArray[float]
       - shape: (28, 28)

    CONV_1:
       - kernel: (5, 5, 1)
       - kernel size: 6
       - bias: 6
       - padding: 2
       - stride: 1
       - output: (28, 28, 6)

    POOL_1:
       - kernel: (2, 2)
       - padding: 0
       - stride: 2
       - output: (14, 14, 6)

    CONV_2:
       - kernel: (5, 5, 6)
       - kernel size: 16
       - bias: 16
       - padding: 0
       - stride: 1
       - output: (10, 10, 16)

    POOL_2:
       - kernel: (2, 2)
       - padding: 0
       - stride: 2
       - output: (5, 5, 16)

    CONV_3:
       - kernel: (5, 5, 16)
       - kernel size: 120
       - bias: 120
       - padding: 0
       - stride: 1
       - output: (1, 1, 120)

    FC:
       - input: (120, 1)
       - weights: (84, 120)
       - bias: 84
       - output: (84, 1)

    OUTPUT:
       - input: (84, 1)
       - weights: (10, 84)
       - bias: 10
       - output: (10, 1)

    annotation:
    some part of this implementation is not same as original LeNet-5 implementation,
    because of its educational purpose.
    ex. I used Maxpooling instead of subsumpling layer in LeNet-5.
        I used RELU instead of sigmoid as activation function.
    """
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        arguments:
        x: 2DArray[float]
           - shape: (28, 28)

        returns:
        y: 2DArray[float]
           - shape: (10, 1)
        """
        x_col = im2col(x, self.k1_shape)

    def backward(self):
        pass

