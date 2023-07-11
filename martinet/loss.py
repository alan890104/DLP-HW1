import numpy as np


class MSE:
    def __init__(self):
        pass

    def __call__(self, pred_y: np.ndarray, y: np.ndarray):
        return self.forward(pred_y, y)

    def forward(self, pred_y: np.ndarray, y: np.ndarray):
        return np.mean((pred_y - y) ** 2)

    def backward(self, pred_y: np.ndarray, y: np.ndarray):
        return 2 * (pred_y - y) / y.size


class Loss:
    func_map = {
        "mse": MSE,
    }

    def __init__(self, kind: str = "mse") -> None:
        self.loss = self.func_map[kind]()

    def __call__(self, pred_y: np.ndarray, y: np.ndarray):
        return self.loss.forward(pred_y, y)

    def forward(self, pred_y: np.ndarray, y: np.ndarray):
        return self.loss.forward(pred_y, y)

    def backward(self, pred_y: np.ndarray, y: np.ndarray):
        return self.loss.backward(pred_y, y)
