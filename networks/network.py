from abc import ABC, abstractmethod


class Network(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
