# Mytorch

MyTorch is a lightweight and educational deep learning library designed to help users understand the fundamental components of neural networks. Unlike established libraries like PyTorch or TensorFlow, MyTorch encourages you to build and interact with key concepts from scratch. This project is perfect for students or developers looking to get a deeper understanding of neural network operations, such as backpropagation, optimizers, loss functions, and layers, without relying on external black-box libraries.

The project is modular, with distinct directories for layers, activation functions, optimizers, and utilities, making it easy to extend and modify components to explore various neural network architectures and learning algorithms.
Core Components of MyTorch

## Activation Functions (activation/):
        These functions are used to introduce non-linearity into the network. Non-linear functions allow the model to learn from the error and update weights in complex ways that cannot be achieved using only linear transformations.
        MyTorch includes:
            ReLU: Rectified Linear Unit, a popular activation function for its simplicity and efficiency.
            Leaky ReLU: A variant of ReLU that allows a small gradient when the input is negative.
            Sigmoid: A classical function that maps values between 0 and 1, often used for binary classification.
            Tanh: Hyperbolic tangent, which maps values between -1 and 1.
            Softmax: Primarily used in the final layer for multi-class classification.
            Step: A step function used mainly for binary decisions.

## Layers (layer/):
        Layers form the core building blocks of any neural network. MyTorch provides implementations of essential layers like fully connected layers and convolutional layers, along with pooling layers for dimensionality reduction.
        MyTorch includes:
            Linear Layer: Fully connected layer where every neuron from the previous layer is connected to every neuron in the next layer.
            Conv2D: A 2D convolutional layer to capture spatial relationships in data (e.g., images).
            MaxPool2D and AvgPool2D: Pooling layers used for down-sampling, either by taking the maximum or the average of a local region.

## Loss Functions (loss/):
        Loss functions calculate the difference between the model's prediction and the actual label during training. The goal of training is to minimize this loss.
        MyTorch includes:
            Cross Entropy (CE): Used in classification tasks where the output is categorical.
            Mean Squared Error (MSE): Used in regression tasks to measure the difference between actual and predicted values.

## Optimizers (optimizer/):
        Optimizers are used to update the parameters (weights) of the model based on the calculated gradients during backpropagation. They minimize the loss by tweaking the parameters.
        MyTorch includes:
            SGD: Stochastic Gradient Descent, the simplest optimizer.
            Adam: Adaptive Moment Estimation, a popular optimizer that adjusts the learning rate for each parameter.
            RMSProp: An optimizer that divides the learning rate by an exponentially decaying average of squared gradients.
            Momentum: A variation of SGD that uses a moving average of gradients to accelerate learning in relevant directions.

## Utilities (util/):
        Helper functions to support various aspects of neural network training.
        MyTorch includes:
            Data Loader: A function to batch data and make it easier to feed into the model during training.
            Flatten: A utility to flatten a multi-dimensional input, usually before feeding it to fully connected layers.
            Initializer: Functions to initialize the weights and biases of the model.

## Tensor (tensor.py):
        Provides custom tensor operations, mimicking functionalities of tensors from popular deep learning frameworks like PyTorch or NumPy. It allows for element-wise operations, broadcasting, and more.

## Model (model.py):
        Likely serves as the main file where models are defined, and training loops are managed. It’s where the forward pass (predictions) and backward pass (gradients) are computed.
