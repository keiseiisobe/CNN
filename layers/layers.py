from abc import ABC, abstractmethod

# base layer
class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, delta):
        pass

# normal layers
class Conv(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, delta):
        pass

class MaxPool(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, delta):
        pass

class FC(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, delta):
        pass

# loss layers
class CrossEntropy(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, delta):
        pass

# activation layers
class Relu(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        x[x < 0] = 0
        return x

    def backward(self, delta):
        pass
