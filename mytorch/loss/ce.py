from mytorch import Tensor, Dependency
import numpy as np

def CategoricalCrossEntropy(predicted: Tensor, actual: Tensor) -> Tensor:
    exp_predictions = np.exp(predicted.data)
    sum_exp_predictions = np.sum(exp_predictions, axis=1, keepdims=True)
    softmax_output = exp_predictions / sum_exp_predictions

    epsilon = 1e-9
    log_softmax_output = np.log(softmax_output + epsilon)

    batch_size = predicted.shape[0]
    true_label_indices = []
    for i in range(batch_size):
        true_label_indices.append(int(actual.data[i].data.item()))
    selected_log_probs = log_softmax_output[np.arange(batch_size), true_label_indices]

    mean_loss = -np.mean(selected_log_probs)

    gradient_output = softmax_output.copy()
    gradient_output[np.arange(batch_size), true_label_indices] -= 1
    gradient_output /= batch_size

    loss_tensor = Tensor(np.array([mean_loss]), requires_grad=True)

    if predicted.requires_grad:
        def backward_pass(grad: np.ndarray) -> np.ndarray:
            return grad * gradient_output

        loss_tensor.depends_on.append(Dependency(predicted, backward_pass))

    return loss_tensor
