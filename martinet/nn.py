from abc import ABC, abstractmethod

import numpy as np


class FuncCtx:
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *args: np.ndarray) -> None:
        self.saved_tensors = args


class MartiNet(ABC):
    def __init__(self):
        self.ctx = FuncCtx()

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(MartiNet):
    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__()
        self.w = np.random.randn(in_size, out_size)
        self.w_grad = None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.w)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.w_grad = np.dot(self.x.T, grad)
        return np.dot(grad, self.w.T)


class Sequential(MartiNet):
    """
    Sequential model for MartiNet
    """

    def __init__(self, *args: MartiNet) -> None:
        self.layers = args

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
