from abc import ABC, abstractmethod


class Network(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x, y):
        pass

    @abstractmethod
    def backward(self, delta):
        pass

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
        pass
