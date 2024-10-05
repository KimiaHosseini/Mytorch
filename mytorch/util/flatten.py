import numpy as np
from mytorch import Tensor
from mytorch import Dependency

def flatten(x: Tensor) -> Tensor:
    """
    TODO: implement flatten. 
    this methods transforms a n dimensional array into a flat array
    hint: use numpy flatten
    """
    # data = x.data.flatten()
    # req_grad = x.requires_grad
    # depends_on = []
    # if req_grad:
    #     def grad_fn(grad: np.ndarray) -> np.ndarray:
    #         return grad.reshape(x.shape)
    #
    #     depends_on = [Dependency(x, grad_fn)]
    #
    # return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
    return x.reshape((x.shape[0], -1))
