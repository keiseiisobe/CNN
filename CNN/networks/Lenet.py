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
    INPUT -> [CONV -> POOL] * 2 -> CONV -> FC -> FC -> SOFTMAX -> CROSSENTROPY

    details:
    INPUT: 3DArray[float]
       - shape: (28, 28, 1)

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
       - weights: (84, 120)
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

    def forward(self, x):
        """
        arguments:
        x: 3DArray[float]
           - shape: (28, 28, 1)

        returns:
        y: 2DArray[float]
           - shape: (10, 1)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dLdy):
        for layer in reversed(self.layers):
            dLdy = layer.backward(dLdy)

    def train(self, x_train, y_train):
        y = self.forward(x_train)
        loss = self.crossentropy.forward(y, y_train)
        dLdy = self.crossentropy.backward(y, y_train)
        self.backward(dLdy)
        return loss

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def test(self, x_train, y_train):
        y = self.forward(x_train)
        print("predict:", self.softmax(y))
        print("label:", y_train)

    def save(self, filename="model.npz"):
        np.savez(
            filename,
            conv1=self.conv1.weights,
            conv2=self.conv2.weights,
            conv3=self.conv3.weights,
            fc1=self.fc1.weights,
            fc2=self.fc2.weights
        )

    def load(self, filename="model.npz"):
        model = np.load(filename)
        self.conv1.weights = model["conv1"]
        self.conv2.weights = model["conv2"]
        self.conv3.weights = model["conv3"]
        self.fc1.weights = model["fc1"]
        self.fc2.weights = model["fc2"]
