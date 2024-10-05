from typing import Any, List
from mytorch import Tensor
from mytorch.layer import Layer, Conv2d, Linear

class Model:
    def __init__(self) -> None:
        self.training = True

    def __call__(self, inp: Tensor) -> Tensor:
        return self.forward(inp)

    "Override this method when defining your own model."
    def forward(self, x: Tensor) -> Tensor:
        print("forward method not implemented.")
        return None

    def parameters(self) -> List[Layer]:
        params = []
        for _, attribValue in self.__dict__.items():
            if isinstance(attribValue, (Conv2d, Linear)):
                params.extend(attribValue.parameters())
        return params

    def summary(self):
        for attribName, attribValue in self.__dict__.items():
            if isinstance(attribValue, Layer):
                print(attribName + ': ', attribValue)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
