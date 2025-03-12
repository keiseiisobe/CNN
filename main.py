from mnist import mnist
import matplotlib.pyplot as plt
from networks.Lenet import Lenet

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = mnist.load_data()
    # normalization
    x_train = x_train / 255
    x_test = x_test / 255

    # hyperparameters
    epoch = 1
    # batch_size = 1 -> default

    lenet = Lenet()
    for i in range(epoch):
        for j in range(1): # len(x_train)):
            lenet.forward(x_train[j].reshape(28, 28), y_train[j])

# plt.subplots()
# plt.imshow(x_train[0].reshape(28, 28))
# plt.show()
# print("x_train: ", x_train.shape)
# print("y_train: ", y_train.shape)
# print("x_test: ", x_test.shape)
# print("y_test: ", y_test.shape)
