from abc import ABC, abstractmethod
import numpy as np
from activations.activations import ReLU

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
    def __init__(self, out_ch, kernel_shape: tuple, padding=0, stride=1, activation="ReLU"):
        self.out_ch = out_ch
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.stride = stride
        self.act_fn = ReLU() if activation == "ReLU" else None
        self.in_ch = None
        self.weights = None
        self.bias = None
        self.x = None
        self.is_initialized = False

    def _init_params(self, in_ch):
        self.in_ch = in_ch
        self.weights = rng.standard_normal(
            (
                self.kernel_shape,
                self.kernel_shape,
                self.in_ch,
                self.out_ch
            )
        ) * np.sqrt(2.0 / (self.kernel_shape**2 * self.in_ch))
        self.is_initialized = True

    def _im2col(self, x, weights_shape):
        """
        this function stretchs out input images.
       (based on https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/utils/utils.py#L486)
        """
        k = self.kernel_shape
        s = self.stride
        in_h, in_w, in_ch = x.shape
        out_h = in_h - k + 1
        out_w = in_w - k + 1
        x_col = np.zeros((k**2 * self.in_ch, out_h * out_w))
        for h in range(0, out_h, self.stride):
            for w in range(0, out_w, self.stride):
                patch = x[h:h+k, w:w+k, :].ravel()
                x_col[:, h + w] += patch[:]
        return x_col, out_h, out_w

    def _col2im(self, x_col):
        k = self.kernel_shape
        s = self.stride
        x_col_h, x_col_w = x_col.shape
        x_h, x_w, in_ch = self.x.shape
        w_h, w_w, in_ch, out_ch = self.weights.shape
        out_h = x_h - w_h + 1
        out_w = x_w - w_w + 1
        dx = np.zeros(self.x.shape)
        for h in range(out_h):
            for w in range(out_w):
                x_col_patch = x_col[:, h + w].reshape((k, k, in_ch))
                dx[h:h*s+k, w:w*s+k, :] += x_col_patch
        return dx

    def _inner_forward(self, x):
        p = self.padding
        if p > 0:
            # zero padding for 0 and 1 axis (x and y), not for 2 axis (self.in_ch)
            self.x = np.pad(x, ((p, p), (p, p), (0, 0)), constant_values=0)
        x_col, out_h, out_w = self._im2col(self.x, self.weights.shape)
        w_raw = self.weights.reshape(self.out_ch, -1)
        return (w_raw @ x_col).reshape(out_h, out_w, self.out_ch)
        
    def forward(self, x):
        #print("conv")
        #print("x:", x.shape)
        # store input to obtain grad later
        self.x = x
        if not self.is_initialized:
            self._init_params(self.x.shape[2])
        z = self._inner_forward(self.x)
        a = self.act_fn.fn(z)
        #print("y:", a.shape)
        return a

    def backward(self, dLdy):
        """
        dL/dw = conv(x, delta)
        dL/dx = conv(delta, 180 degree rotated w)
        """
        #print("conv")
        #print("dLdy:", dLdy.shape)
        dLdy = dLdy.reshape((-1, self.out_ch))
        dz = self.act_fn.grad(dLdy)
        #print("dz:", dz.shape)
        x_col, out_h, out_w = self._im2col(self.x, self.weights.shape)
        #print("x:", self.x.shape)
        #print("w:", self.weights.shape)
        #print("out_h:", out_h)
        #print("x_col:", x_col.shape)
        dw = (x_col @ dz).reshape(self.weights.shape)
        self.weights -= dw * learning_rate
        self.weights = np.rot90(self.weights, k=2)
        #print("w:", self.weights.shape)
        dx_col = dw.reshape(-1, self.out_ch) @ dz.T
        #print("dx_col:", dx_col.shape)
        dx = self._col2im(dx_col)
        #print("dx:", dx.shape)
        return dx
        


class MaxPool(Layer):
    def __init__(self, kernel_shape, padding=0, stride=2):
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.stride = stride
        self.x = None

    def forward(self, x):
        #print("maxpool")
        #print("x:", x.shape)
        in_h, in_w, in_ch = x.shape
        k = self.kernel_shape
        s = self.stride
        out_h = int((in_h - k) / s + 1)
        out_w = int((in_w - k) / s + 1)
        out_ch = in_ch
        y = np.zeros((out_h, out_w, out_ch))
        for h in range(0, out_h, s):
            for w in range(0, out_w, s):
                y[h, w, :] = np.max(x[h:h+k, w:w+k, :], (0, 1))
        # store input to obtain grad later
        self.x = x
        #print("y:", y.shape)
        return y

    def backward(self, dLdy):
        #print("maxpool")
        #print("dLdy:", dLdy.shape)
        in_h, in_w, in_ch = self.x.shape
        k = self.kernel_shape
        s = self.stride
        out_h = int((in_h - k) / s + 1)
        out_w = int((in_w - k) / s + 1)
        out_ch = in_ch
        dLdy = dLdy.reshape((out_h, out_w, out_ch))
        dx = np.zeros(self.x.shape)
        for h in range(0, out_h):
            for w in range(0, out_w):
                for c in range(out_ch):
                    x_patch = self.x[h:h*s+k, w:w*s+k, c]
                    x, y = np.argwhere(x_patch == np.max(x_patch))[0]
                    mask = np.zeros(x_patch.shape)
                    mask[x, y] = 1
                    dx[h:h*s+k, w:w*s+k, c] += dLdy[h, w, c] * mask
        #print("dx:", dx.shape)
        return dx

class FC(Layer):
    def __init__(self, n_out, activation="ReLU"):
        self.n_out = n_out
        self.act_fn = ReLU() if activation == "ReLU" else None
        self.n_in = None
        self.weights = None
        self.x = None
        self.is_initialized = False

    def _init_params(self, n_in):
        self.n_in = n_in
        self.weights = rng.standard_normal(
            (
                self.n_out,
                self.n_in
             )
        ) * np.sqrt(2.0 / self.n_in)
        self.is_initialized = True

    def forward(self, x):
        #print("fc")
        #print("x:", x.shape)
        # insert Flatten layer if needed
        if len(x.shape) > 1:
            x = x.flatten()
        x = np.expand_dims(x, 1)
        if not self.is_initialized:
            self._init_params(x.shape[0])
        z = np.dot(self.weights, x)
        a = self.act_fn.fn(z)
        self.x = x
        #print("y:", a.shape)
        return a

    def backward(self, dLdy):
        #print("fc")
        #print("dLdy:", dLdy.shape)
        dz = self.act_fn.grad(dLdy)
        dw = np.dot(dz, self.x.T)
        self.weights -= dw * learning_rate
        dx = np.dot(self.weights.T, dz)
        #print("dx shape:", dx.shape)
        return dx
