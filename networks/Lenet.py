from networks.network import Network
from layers.layers import Conv, MaxPool, FC
from losses.losses import CrossEntropy
import numpy as np

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

    MAXPOOL_1:
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

    MAXPOOL_2:
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

    FC_1:
       - input: (120, 1)
3       - weights: (84, 120)
       - bias: 84
       - output: (84, 1)

    FC_2:
       - input: (84, 1)
       - weights: (10, 84)
       - bias: 10
       - output: (10, 1)

    LOSS:
       - input: (10, 1)
       - output: float

    annotation:
    some part of this implementation is not same as original LeNet-5 implementation,
    because of its educational purpose.
    ex. I used Maxpooling instead of subsumpling layer in LeNet-5.
        I used RELU instead of sigmoid as activation function.
    """
    def __init__(self):
        self.conv1 = Conv(6, 5, padding=2)
        self.maxpool1 = MaxPool(2)
        self.conv2 = Conv(16, 5)
        self.maxpool2 = MaxPool(2)
        self.conv3 = Conv(120, 5)
        self.fc1 = FC(84)
        self.fc2 = FC(10)
        self.layers = [
            self.conv1,
            self.maxpool1,
            self.conv2,
            self.maxpool2,
            self.conv3,
            self.fc1,
            self.fc2,
        ]
        self.crossentropy = CrossEntropy()

    def forward(self, x, y):
        """
        arguments:
        x: 3DArray[float]
           - shape: (28, 28, 1)

        returns:
        y: 2DArray[float]
           - shape: (10, 1)
        """
        #print("forward")
        for layer in self.layers:
            x = layer.forward(x)
        #print("Lenet output:", x)
        return x

    def backward(self, dLdy):
        #print("backward")
        for layer in reversed(self.layers):
            dLdy = layer.backward(dLdy)

    def train(self, x_train, y_train):
        y = self.forward(x_train, y_train)
        loss = self.crossentropy.forward(y, y_train)
        print("loss:", loss)
        dLdy = self.crossentropy.backward(y, y_train)
        self.backward(dLdy)

    def save(self, filename="model.npz"):
        np.savez(
            filename,
            conv1=self.conv1.w,
            conv2=self.conv2.w,
            conv3=self.conv3.w,
            fc1=self.fc1.w,
            fc2=self.fc2.w
        )

    def load(self, filename="model.npz"):
        model = np.load(filename)
        self.conv1.w = model["conv1"]
        self.conv2.w = model["conv2"]
        self.conv3.w = model["conv3"]
        self.fc1.w = model["fc1"]
        self.fc2.w = model["fc2"]
