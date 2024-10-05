from mytorch import Tensor
import numpy as np
from mytorch import Dependency



def MeanSquaredError(preds: Tensor, actual: Tensor):
    "TODO: implement Mean Squared Error loss"
    error = preds - actual
    error2 = error ** 2
    mse = error2.sum()
    size = Tensor(np.array([error2.data.size], dtype=np.float64))
    size_inv = size ** -1
    mse = mse * size_inv

    if mse.requires_grad:
        def grad_fn(grad: np.ndarray):
            return grad * (2 * error.data / error2.data.size)

        mse.depends_on.append(Dependency(error, grad_fn))

    return mse
