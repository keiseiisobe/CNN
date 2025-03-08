from mnist import mnist
import matplotlib.pyplot as plt
from networks.Lenet import Lenet

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = mnist.load_data()
    # normalization
    x_train /= 255
    x_test /= 255

    # hyperparameters
    epoch = 10

    lenet = Lenet()
    y = lenet.forward(x_train[0].reshape(28, 28))
    print("y.shape:", y.shape)
    print("y:", y)

# plt.subplots()
# plt.imshow(x_train[0].reshape(28, 28))
# plt.show()
# print("x_train: ", x_train.shape)
# print("y_train: ", y_train.shape)
# print("x_test: ", x_test.shape)
# print("y_test: ", y_test.shape)
