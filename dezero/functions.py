import numpy as np
from dezero.core import Function, as_variable

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y #Variable 인스턴스임에 주의

    def backward(self, gy): ##모든 변수는 Variable의 인스턴스
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


# class Sum(Function):
#     # def __init__(self, axis, keepdims):
#     #     self.axis = axis
#     #     self.keepdims = keepdims

#     def forward(self, x):
#         self.x_shape = x.shape
#         y = x.sum()
#         return y

#     def backward(self, gy):
#         # gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis,
#         #                                 self.keepdims)
#         gx = broadcast_to(gy, self.x_shape)
#         return gx


# def sum(x):
#     return Sum()(x)
