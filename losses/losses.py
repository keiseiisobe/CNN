import numpy as np

class CrossEntropy:
    def __init__(self):
        self.probs = None

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, predict, label):
        #print("crossentropy")
        self.probs = self.softmax(predict)
        loss = -np.sum(np.log(self.probs) * label)
        return loss

    def backward(self, predict, label):
        return predict - label
