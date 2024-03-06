### Deep Rewiring in Neural Networks

Please install snnTorch from source! there are useful performance fixes that have not been used yet

Uses the Deepr algorithm from the paper https://arxiv.org/abs/1711.05136.

SNN_MNIST.ipynb - spiking feedforward neural network, spike rate for encoding,decoding,loss, with ATAN surrogate gradient.\\
torch_MNIST.ipynb - Feedforward neural network with ReLU neurons & cross entropy loss, for comparison with SNN's.\\
torch_MNIST_deepr.ipynb - dynamic rewiring works with good accuracy.\\ 
SNN_deepr_MNIST.ipynb - good dynamic rewiring, possibly lower connectivity than classical.\\

SNN_DVS - reconstruct the DVS model from the IBM  "A Low Power, Fully Event-Based Gesture Recognition System" paper and apply sparsity algorithm to see if we can get comparable accuracy when training on DVS gesture. serves as an alternative to trinary weights?