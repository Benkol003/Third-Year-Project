##attempts to implement dropping neuron connections (by zeroing weight matrix elements) over snntorch_MNIST.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import utils
from snntorch import spikegen
import snntorch.spikeplot as splt
from snntorch import surrogate


import matplotlib.pyplot as plt
import numpy as np

dtype=torch.float
torch.manual_seed(734)
print("Feedforward SNN Trained on MNIST")

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Training Parameters
batch_size=256
data_path='./tmp/data/mnist'
num_classes = 10  # MNIST has 10 output classes

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# # temporary dataloader if MNIST service is unavailable
# !wget www.di.ens.fr/~lelarge/MNIST.tar.gz
# !tar -zxvf MNIST.tar.gz

# mnist_train = datasets.MNIST(root = './', train=True, download=True, transform=transform)
# mnist_test = datasets.MNIST(root = './', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True,drop_last=True)

################ MNIST Model ##############################################################

# layer parameters
num_inputs = 28*28
num_hidden1 = 200
num_hidden2 = 50
num_outputs = 10
num_steps = 26  # for spike encoding
beta = 0.95 #leak rate
lr=5e-5
#weight_decay=1e-6

spike_grad1 = surrogate.atan() 

class WeightDropConnect(nn.Module):
    def __init__(self,threshold = 1e-06):
        super().__init__()
        self.threshold=threshold

    #mask the weights on the forward pass
    def forward_pre_hook(self,module,args): # -> None or modified input:

        module.weight.data = torch.where(
            torch.abs(module.weight.data) < self.threshold, 
            torch.zeros_like(module.weight.data), 
            module.weight.data)
        #zero_weights = torch.eq(module.weight.data, 0).sum().item()
        #print("zero weights: ",zero_weights)

    #mask the gradients on the backward pass
    def back_hook(self,module,grad_input,grad_output): #-> tuple(nn.Tensor) or None: want a post hook for dropconnect, pre hook for dropout
        # DO NOT MODIFY INPUT IN-PLACE!
        return (grad_input) 
    

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        #self.dropout = nn.Dropout(0.05)


        # initialize layers
        self.dropper = WeightDropConnect()
        

        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.fc1.register_forward_pre_hook(self.dropper.forward_pre_hook)
        self.fc1.register_full_backward_hook(self.dropper.back_hook)
        self.lif1 = snn.Leaky(beta=beta,spike_grad=spike_grad1)

        self.fc2 = nn.Linear(num_hidden1,num_hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad1)

        self.fc3 = nn.Linear(num_hidden2,num_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad1)

        #pytorch creates the tensors to represent the network layout and weights for each layer; snntorch provides the model that operates on the entire tensor (at each layer).

  
    def forward(self,x): #x is input data

        #spike encoding at input layer
        x_spk = spikegen.rate(x,num_steps=num_steps) 

        # Initialize neuron hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif2.init_leaky()

        # record outputs
        mem3_rec = []
        spk3_rec = []

        for step in range(num_steps):
            #x = x_spk[step] #for encoded input

            cur1 = self.fc1(x)
            #cur1 = self.fc1(x) #for non-encoded input
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)
    
###################################################################################

########### STATS ##############

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    dev = np.std((targets == idx).detach().cpu().numpy())
    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
        print(f"Train set deviation for a single minibatch: {dev*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
        print(f"Test set deviation for a single minibatch: {dev*100:.2f}%")

def train_printer():
    print(f"Epoch {epoch}, Iteration {img_counter}")
    print(f"Train Set Loss: {loss_hist[batch_counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[img_counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")


##############################


# Load the network onto CUDA
net = Net().to(device)

loss = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(net.parameters(),lr=lr, momentum=0.9,) #weight_decay=weight_decay)


num_epochs = 3
loss_hist = []
test_loss_hist = []

img_counter = 0 #total no. of images iterated over

#training loop
for epoch in range(num_epochs):
    batch_counter=0 #image number within current batch

    train_batches = iter(train_loader)

    #mini-batch loop
    for data, targets in train_batches: #torch.Size([128, 1, 28, 28]), torch.Size([128])

        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train() #inform pytorch
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        #calculate loss
        loss_val = torch.zeros((1), dtype=dtype, device=device)

        for step in range(num_steps):
           loss_val += loss(mem_rec[step], targets)

        optimiser.zero_grad() #(reset for batch)
        loss_val.backward() #calculate backpropogation error gradient
        optimiser.step() #then update parameters

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad(): #tell pytorch to disable gradient calculation (save compute)

            net.eval()
            test_data, test_targets = next(iter(test_loader)) #different test set each time due to shuffle?
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss(test_mem[step], test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if img_counter % 50 == 0:
                train_printer()
            img_counter += 1
            batch_counter +=1


###############################################################################################

# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
  net.eval()
  for data, targets in test_loader:
    data = data.to(device)
    targets = targets.to(device)

    # forward pass
    test_spk, _ = net(data.view(data.size(0), -1))

    # calculate total accuracy
    _, predicted = test_spk.sum(dim=0).max(1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")

print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

###TODO: change loss functions, latency encoding,
# can we implement STDP?

#Notes:
#training time is almost double
