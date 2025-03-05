import numpy as np
import numpy.typing as npt
import _pickle as pickle
from typing import Annotated
from networks import Lenet

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict

# ConvNet Architecture
# INPUT -> CONV_1(32, (3, 3)) -> RELU_1 -> POOL_1(2, 2)
#       -> CONV_2(64, (3, 3)) -> RELU_2 -> POOL_2(2, 2)
#       -> CONV_3(64, (3, 3)) -> Flatten(3 * 3 * 64)
#       -> (W1) -> FC(64) -> (W2)
#       -> OUTPUT(10)

# parameter shapes
# CONV_1 = (3, 3, 32)
# CONV_2 = (3, 3, 64)
# CONV_3 = (3, 3, 64)
# W1 = (64, 3 * 3 * 64)
# W2 = (10, 64)

NDArrayFloat = npt.NDArray[np.float64]

class Model:
    def __init__(self):
        rng = np.random.default_rng()
        self.model = {}
        self.model["CONV_1"] = rng.standard_normal((3, 3, 32)) # missing initialization
        self.model["CONV_2"] = rng.standard_normal((3, 3, 64))
        self.model["CONV_3"] = rng.standard_normal((3, 3, 64))
        self.model["W1"] = rng.standard_normal((64, 3*3*64))
        self.model["W2"] = rng.standard_normal((10, 64))

    def im2col(self,
               x: Annotated[NDArrayFloat, (32*32*3)]
               ) -> Annotated[NDArrayFloat, (27, 900)]:
        pass

    def fi2raw(self):
        pass
    
    def sigmoid(self, y):
        return 1.0 / (1.0 + np.exp(-y))

    def cross_entropy_loss(self):
        pass
    
    def forward(self,
                x: Annotated[NDArrayFloat, (32*32*3,)]
                ) -> Annotated[NDArrayFloat, (10)]:
        x_col = self.im2col(x)
        w_raw = self.fi2raw(self.model["CONV_1"])
        out = np.dot(w_raw, x_col) # CONV_1
        out[out < 0] = 0 # RELU_1
        out = max_pooling(out)
        pass

    def backward(self, loss: np.float64):
        pass


if __name__ == "__main__":
    filename_head = "cifar-10/data_batch_"
    net = Lenet()
    # cifar-10/ data_batch_1 ~ data_batch_5
    for i in range(1, 6):
        dict = unpickle(filename_head + str(i))
        # 10000 images per a data_batch_*
        for j in range(10000):
            data = dict[b"data"][i]
            # data.shape = [32 * 32 * 3, ]
            # normalize data value to be between 0 and 1
            data /= 255
            net.train(data)
