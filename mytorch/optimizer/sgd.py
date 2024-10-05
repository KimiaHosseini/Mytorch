from typing import List
from mytorch.layer import Layer
from mytorch.optimizer import Optimizer
from mytorch import Tensor


class SGD(Optimizer):
    def __init__(self, layers: List[Layer], learning_rate=0.1):
        super().__init__(layers)
        self.learning_rate = learning_rate

    def step(self):
        "Implement SGD algorithm"
        for l in self.layers:
            l.weight.data -= l.weight.grad.data * self.learning_rate
            if l.need_bias:
                l.bias.data -= l.bias.grad.data * self.learning_rate