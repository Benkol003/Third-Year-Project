import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import utils
from snntorch import spikegen
import snntorch.spikeplot as splt
from snntorch.surrogate import Sigmoid

import matplotlib.pyplot as plt

from IPython.display import HTML
import webbrowser


######### RATE ENCODING TUTORIAL #####################################################################

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

# Training Parameters
batch_size=128
data_path='./tmp/data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Torch Variables
dtype = torch.float

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)


data = iter(train_loader)


### SNN & spikingjelly normalises the timestep as 1
'''
# Temporal Dynamics
num_steps = 10

######## Spikegen #########
# create vector filled with 0.5
raw_vector = torch.ones(num_steps)*0.5
print("raw vector: ",raw_vector)

# pass each sample through a Bernoulli trial

rate_coded_vector = torch.bernoulli(raw_vector) #random assignment
print(f"The bernoulli output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")
#As num_steps approaches infinity, the proportion of spikes approaches the original raw value.

# Iterate through minibatches
data_it, targets_it = next(data)

# Spiking Data
spike_data = spikegen.rate(data_it, num_steps=num_steps)
print("spike_data size: ",spike_data.size()) # - torch.Size([10, 128, 1, 28, 28]) - [num_steps x batch_size x input dimensions]
# images are (1,28,28)
##############################

# print a spike-encoded image
spike_data_sample = spike_data[:, 0, 0] #torch.Size([10, 28, 28])
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample, fig, ax)
plt.rcParams['animation.ffmpeg_path'] = "D:\Program Files (x86)\\ffmpeg\\bin\\ffmpeg.exe"
HTML(anim.to_html5_video())


# Assuming anim.to_html5_video() returns the HTML content as a string
html_content = anim.to_html5_video()
'''
# Save the HTML content to a file
'''
with open("output.html", "w") as html_file:
    html_file.write(html_content)

# Open the file in the default web browser
webbrowser.open("output.html")

print(f"The corresponding target is: {targets_it[0]}")
'''

###############################################################################################




################ MNIST Model ##############################################################

# layer parameters
num_inputs = 784
num_hidden = 1000
num_outputs = 10
num_steps = 20
beta = 0.99

# initialize layers

#spike encoding at input layer

fc1 = nn.Linear(num_inputs, num_hidden)
lif1 = snn.Leaky(beta=beta,spike_grad=Sigmoid)
fc2 = nn.Linear(num_hidden, num_outputs)
lif2 = snn.Leaky(beta=beta,spike_grad=Sigmoid)

#pytorch creates the tensors to represent the network layout and weights for each layer; snntorch provides the model that operates on the entire tensor (at each layer).

# Initialize hidden states
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

# record outputs
mem2_rec = []
spk1_rec = []
spk2_rec = []

for data_it, targets_it in data: #iterate over mini-batches:
  
    #data_it torch.Size([128, 1, 28, 28])

    spike_data = spikegen.rate(data_it, num_steps=num_steps) # torch.Size([20, 128, 1, 28, 28])



###############################################################################################