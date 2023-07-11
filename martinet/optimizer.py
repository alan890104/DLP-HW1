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
