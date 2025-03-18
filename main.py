from mnist import mnist
import matplotlib.pyplot as plt
from networks.Lenet import Lenet
import numpy as np

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = mnist.load_data()
    # normalization
    x_train = x_train / 255
    x_test = x_test / 255
    # create one hot vector of labels
    one_hot_y_train = np.zeros((y_train.shape[0], 10))
    for i, v in enumerate(y_train):
        one_hot_y_train[i][v] = 1
    one_hot_y_test = np.zeros((y_test.shape[0], 10))
    for i, v in enumerate(y_test):
        one_hot_y_test[i][v] = 1

    # hyperparameters
    epoch = 100
    batch_size = 1  # 1 (stochastic gradient decent)

    lenet = Lenet()
    for i in range(epoch):
        for j in range(batch_size):
            loss = lenet.train(x_train[j].reshape(28, 28, 1), one_hot_y_train[j].reshape(-1, 1))
            print("loss:", loss)
    lenet.test(x_train[j].reshape(28, 28, 1), one_hot_y_train[j].reshape(-1, 1))

# plt.subplots()
# plt.imshow(x_train[0].reshape(28, 28))
# plt.show()
# print("x_train: ", x_train.shape)
# print("y_train: ", y_train.shape)
# print("x_test: ", x_test.shape)
# print("y_test: ", y_test.shape)
