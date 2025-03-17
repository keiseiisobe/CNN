from abc import ABC, abstractmethod

class Activation(ABC):
    def __init__(self):
        pass

    def fn(self, x):
        pass

    def grad(self, x):
        pass

class ReLU:
    def __init__(self):
        pass

    def fn(self, x):
        #print("ReLU")
        x[x < 0] = 0
        return x

    def grad(self, x):
        x[x < 0] = 0
        return x
