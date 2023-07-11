from .nn import Sequential, Linear
from typing import Literal


class GD:
    def __init__(self, nn: Sequential, lr: float = 0.1, **kwargs) -> None:
        self.nn: Sequential = nn
        self.lr: float = lr

    def step(self) -> None:
        for layer in self.nn.layers:
            if isinstance(layer, Linear):
                layer.w -= self.lr * layer.w_grad / layer.w.shape[0]

    def zero_grad(self) -> None:
        for layer in self.nn.layers:
            if isinstance(layer, Linear):
                layer.w_grad = None


class Adam:
    def __init__(
        self,
        nn: Sequential,
        lr: float = 0.1,
        alpha: float = 0.9,
        beta: float = 0.999,
        epsilon: float = 1e-8,
        **kwargs,
    ) -> None:
        self.nn = nn
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.lr = lr

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for layer in self.nn.layers:
            if isinstance(layer, Linear):
                layer.w_grad = None


class Optimizer:
    func_map = {
        "gd": GD,
    }

    def __init__(
        self,
        nn: Sequential,
        lr: float = 0.1,
        kind: Literal["gd"] = "gd",
        **kwargs,
    ) -> None:
        self.opt = self.func_map[kind](nn, lr, **kwargs)

    def step(self) -> None:
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()
