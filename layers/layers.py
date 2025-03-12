from abc import ABC, abstractmethod
import numpy as np

# for debug
import sys
np.set_printoptions(threshold=sys.maxsize)

# random number generator
rng = np.random.default_rng()

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
        self.weights = rng.standard_normal((field_size, field_size, field_depth, num_of_filters))

    def _get_out_shape(self, x):
        """
        output shape derivation
        out_height = (in_height - w_height + 2 * padding) / stride + 1
        out_width = (in_width - w_width + 2 * padding) / stride + 1
        out_depth = in_depth
        """
        in_h, in_w, in_d = x.shape
        # padding has already been handled in the top of forward function.
        out_h = out_w = int((in_h - self.field_size) / self.stride + 1)
        out_d = self.num_of_filters
        return out_h, out_w, out_d
                
    def _im2col(self, x, out_shape):
        """
        this function stretchs out input images.
       (based on https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/utils/utils.py#L486)
        """
        x_col = np.zeros((self.field_size**2 * self.field_depth, out_shape[0]**2))
        for h in range(0, out_shape[0], self.stride):
            for w in range(0, out_shape[1], self.stride):
                col = x[h:h + self.field_size, w:w + self.field_size, :].ravel()
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
        out_shape = self._get_out_shape(x)
        x_col = self._im2col(x, out_shape)
        w_raw = self.weights.reshape(self.num_of_filters, self.field_size**2 * self.field_depth)
        y = np.dot(w_raw, x_col).reshape(out_shape[0], out_shape[1], out_shape[2])
        print("x_col shape:", x_col.shape)
        print("w_raw shape:", w_raw.shape)
        print("y shape:", y.shape)
        return y

    def backward(self, delta):
        pass

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
        pass

class FC(Layer):
    def __init__(self, weights_shape):
        self.weights_shape = weights_shape
        self.weights = rng.standard_normal(self.weights_shape)

    def forward(self, x):
        print("forwarding fully connected layer")
        # flatten
        if len(x.shape) > 1:
            x.ravel()
        x = np.expand_dims(x, 1)
        print("x shape:", x.shape)
        print("w shape:", self.weights.shape)

    def backward(self, delta):
        pass

# loss layers
class CrossEntropy(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        print("forwarding cross entropy layer")

    def backward(self, delta):
        pass

# activation layers
class Relu(Layer):
    def __init__(self):
        pass

    def forward(self, x):
        print("forwarding relu layer")
        x[x < 0] = 0
        return x

    def backward(self, delta):
        pass
