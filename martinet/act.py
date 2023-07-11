import numpy as np
from .nn import MartiNet


class Sigmoid(MartiNet):
    def forward(self, x: np.ndarray) -> np.ndarray:
        result = 1.0 / (1.0 + np.exp(-x))
        self.ctx.save_for_backward(result)
        return result

    def backward(self, grad: np.ndarray) -> np.ndarray:
        (result,) = self.ctx.saved_tensors
        return grad * result * (1.0 - result)


class ReLU(MartiNet):
    def forward(self, x: np.ndarray) -> np.ndarray:
        result = np.maximum(0.0, x)
        self.ctx.save_for_backward(result)
        return result

    def backward(self, grad: np.ndarray) -> np.ndarray:
        (result,) = self.ctx.saved_tensors
        return grad * (result > 0).astype(float)


class Act(MartiNet):
    func_map = {
        "relu": ReLU(),
        "sigmoid": Sigmoid(),
    }

    def __init__(self, kind: str = "relu"):
        super().__init__()
        self.activation = self.func_map[kind]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.activation.backward(grad)
