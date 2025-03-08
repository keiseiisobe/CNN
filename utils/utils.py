def im2col(x, k_shape, padding, stride):
    """
    arguments:
    x: np.3DArray[float]
       - shape: (width, height, channel)
    k_shape: tuple
       - "kernel shape"
       - shape: (width, height, channel, size)
    padding: int
    stride: int

    returns:
    x_col: np.3DArray[float]
       - shape: (k_width * k_height, y_width * y_height, channel)
    """
    pass
