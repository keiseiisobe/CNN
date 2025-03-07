from mnist import mnist
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test = mnist.load_data()
fig, axes = plt.subplots(3, 3)
for i in range(9):
    plt.imshow(x_train[0].reshape(28, 28))
plt.show()
print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)
