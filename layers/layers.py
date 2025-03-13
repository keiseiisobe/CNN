from abc import ABC, abstractmethod
import numpy as np

# for debug
import sys
np.set_printoptions(threshold=sys.maxsize)

# random number generator
rng = np.random.default_rng()

# hyperparameters
learning_rate = 0.0001

# base layer
class Layer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, delta):
        pass

# normal layers
class Conv(Layer):
    def __init__(self, field_size, field_depth, num_of_filters, padding=0, stride=1):
        self.field_size = field_size
        self.field_depth = field_depth
        self.num_of_filters = num_of_filters
        self.padding = padding
        self.stride = stride
        self.weights = rng.standard_normal((field_size, field_size, field_depth, num_of_filters)) * \
            np.sqrt(2.0 / (field_size**2 * field_depth))
        self.x = None
        self.out_shape = None

    def _get_out_shape(self, x, *args):
        """
        output shape derivation
        out_height = (in_height - w_height + 2 * padding) / stride + 1
        out_width = (in_width - w_width + 2 * padding) / stride + 1
        out_depth = in_depth
        """
        field_size = args[0][0] if len(args) == 1 else self.field_size
        in_h, in_w, in_d = x.shape
        # padding has already been handled in the top of forward function.
        out_h = out_w = int((in_h - field_size) / self.stride + 1)
        out_d = self.num_of_filters
        return out_h, out_w, out_d
                
    def _im2col(self, x, out_shape, *args):
        """
        this function stretchs out input images.
       (based on https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/utils/utils.py#L486)
        """
        field_size = args[0][0] if len(args) == 1 else self.field_size
        field_depth = args[0][2] if len(args) == 1 else self.field_depth
        x_col = np.zeros((field_size**2 * field_depth, out_shape[0]**2))
        print("x_col: ", x_col.shape)
        print("out shape: ", out_shape)
        if len(args) == 1:
            print("w shape: ", args[0])
        for h in range(0, out_shape[0], self.stride):
            for w in range(0, out_shape[1], self.stride):
                col = x[h:h + field_size, w:w + field_size, :].ravel()
                x_col[:, h + w] += col[:]
        print("x shape: ", x.shape)
        print("out shape: ", out_shape)
        return x_col

    def forward(self, x):
        print("forwarding conv layer")
        if len(x.shape) == 2:
            x = np.expand_dims(x, 2)
        if self.padding > 0:
            x = np.pad(x, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), constant_values=0)
        self.x = x
        self.out_shape = self._get_out_shape(x)
        x_col = self._im2col(x, self.out_shape)
        w_raw = self.weights.reshape(self.num_of_filters, self.field_size**2 * self.field_depth)
        y = np.dot(w_raw, x_col).reshape(self.out_shape[0], self.out_shape[1], self.out_shape[2])
        print("x_col shape:", x_col.shape)
        print("w_raw shape:", w_raw.shape)
        print("y shape:", y.shape)
        return y

    def backward(self, delta):
        """
        dL/dw = conv(x, delta)
        dL/dx = conv(delta, 180 degree rotated w)
        """
        print("backwarding conv layer")
        delta = delta.reshape(self.out_shape)
        print("x shape:", self.x.shape)
        print("delta shape:", delta.shape)
        out_shape = self._get_out_shape(self.x, delta.shape)
        print("out shape:", out_shape)


class MaxPool(Layer):
    def __init__(self, field_size, field_depth, padding=0, stride=2):
        self.field_size = field_size
        self.field_depth = field_depth
        self.padding = padding
        self.stride = stride

    def _get_out_shape(self, x):
        in_h, in_w, in_d = x.shape
        out_h = out_w = int((in_h - self.field_size) / self.stride + 1)
        out_d = in_d
        return out_h, out_w, out_d
        
    def forward(self, x):
        print("forwarding maxpooling layer")
        out_shape = self._get_out_shape(x)
        y = np.zeros(out_shape)
        for h in range(0, out_shape[0], self.stride):
            for w in range(0, out_shape[1], self.stride):
                y[h, w, :] = np.max(x[h:h + self.field_size, w:w + self.field_size, :], (0, 1))
        print("x shape:", x.shape)
        print("out shape:", out_shape)
        print("y shape:", y.shape)
        return y

    def backward(self, delta):
        print("backwarding maxpooling layer")

class FC(Layer):
    def __init__(self, weights_shape):
        self.x = None
        self.weights_shape = weights_shape
        self.weights = rng.standard_normal(self.weights_shape) * np.sqrt(2.0 / weights_shape[1])

    def forward(self, x):
        print("forwarding fully connected layer")
        if len(x.shape) > 1:
            x = x.flatten()
        x = np.expand_dims(x, 1)
        y = np.dot(self.weights, x)
        print("x shape:", x.shape)
        print("w shape:", self.weights.shape)
        print("y shape:", y.shape)
        self.x = x
        return y

    def backward(self, delta):
        print("backwarding fully connected layer")
        print("delta shape:", delta.shape)
        print("x shape:", self.x.shape)
        dw = np.dot(delta, self.x.T)
        print("dw shape:", dw.shape)
        self.weights -= dw * learning_rate
        dx = np.dot(self.weights.T, delta)
        print("dx shape:", dx.shape)
        return dx

# loss layers
class CrossEntropy(Layer):
    def __init__(self):
        self._label = None
        self.probs = None

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value.copy()

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, x):
        print("forwarding loss layer")
        print("x shape:", x.shape)
        print("x:", x)
        self.probs = self.softmax(x)
        print("probs shape:", self.probs.shape)
        print("probs:", self.probs)
        y = -np.sum(np.log(self.probs) * self._label)
        print("loss:", y)
        return y

    def backward(self, delta):
        print("backwarding loss layer")
        return self.probs - self._label

# activation layers
class Relu(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        print("forwarding relu layer")
        x[x < 0] = 0
        return x

    def backward(self, delta):
        print("backwarding relu layer")
        delta[delta < 0] = 0
        return delta
