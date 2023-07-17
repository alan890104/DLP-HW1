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


class Momentum:
    def __init__(
        self,
        nn: Sequential,
        lr: float = 0.1,
        beta: float = 0.9,
        **kwargs,
    ) -> None:
        self.nn: Sequential = nn
        self.lr: float = lr
        self.beta: float = beta

    def step(self) -> None:
        for layer in self.nn.layers:
            if isinstance(layer, Linear):
                if not hasattr(layer, "v"):
                    setattr(layer, "v", 0)
                layer.v = self.beta * layer.v - (
                    self.lr * layer.w_grad / layer.w.shape[0]
                )
                layer.w += layer.v

    def zero_grad(self) -> None:
        for layer in self.nn.layers:
            if isinstance(layer, Linear):
                layer.w_grad = None


class Optimizer:
    func_map = {
        "gd": GD,
        "momentum": Momentum,
    }

    def __init__(
        self,
        nn: Sequential,
        lr: float = 0.1,
        kind: Literal["momentum", "gd"] = "momentum",
        **kwargs,
    ) -> None:
        self.opt = self.func_map[kind](nn, lr, **kwargs)

    def step(self) -> None:
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()
