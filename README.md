### Deep Rewiring in Neural Networks

Uses the Deepr algorithm from the paper https://arxiv.org/abs/1711.05136.

SNN_MNIST.ipynb - spiking feedforward neural network, spike rate for encoding,decoding,loss, with ATAN surrogate gradient.\\
torch_MNIST.ipynb - Feedforward neural network with ReLU neurons & cross entropy loss, for comparison with SNN's.\\
torch_MNIST_deepr.ipynb - dynamic rewiring works with good accuracy.\\ 
SNN_deepr_MNIST.ipynb - good dynamic rewiring, possibly lower connectivity than classical.\\

At the moment, sparsity is not good enough to justify use of sparse matricies (<1%).
