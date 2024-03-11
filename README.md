### Deep Rewiring in Neural Networks

Please install snnTorch from source! there are useful performance fixes that have not made it to the current pip release.

deepr.py implements the Deepr algorithm from the paper https://arxiv.org/abs/1711.05136.

torch_MNIST.ipynb - Feedforward neural network with ReLU neurons & cross entropy loss, for comparison with SNN's.  
torch_MNIST_deepr.ipynb - dynamic rewiring works with good accuracy.  
SNN_MNIST.ipynb - spiking feedforward neural network, spike rate for encoding,decoding,loss, with ATAN surrogate gradient.  
SNN_deepr_MNIST.ipynb - good dynamic rewiring, possibly lower connectivity than classical.  

old.SNN_DVS.ipynb - Model for DVSGesture dataset (abandoned due to poor accuracy & high training time)  

SNN_DVS2.ipynb & SNN_DVS2_Deepr.ipynb - training models for DVSGesture dataset, with/without DeepR algorithm used.