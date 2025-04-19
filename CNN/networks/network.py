from abc import ABC, abstractmethod


class Network(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def backward(self, dLdy):
        pass

    @abstractmethod
    def train(self, x_train, y_train):
        pass
