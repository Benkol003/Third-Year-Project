import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np

dtype=torch.float
torch.manual_seed(734)
print("Feedforward ANN Trained on MNIST")

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

############ MNIST Model ################################

# layer parameters
num_inputs = 28*28
num_hidden1 = 200
num_hidden2 = 50
num_outputs = 10

lr=1e-2 #TODO learning rate does not directly translate from the SNN model due to the iteration steps for producing spikes
#weight_decay=1e-6


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.n1 = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.n2 = nn.ReLU()
        self.fc3 = nn.Linear(num_hidden2, num_outputs)

    def forward(self,x):
        x = self.fc1(x)
        x = self.n1(x)
        x = self.fc2(x)
        x = self.n2(x)
        x = self.fc3(x)

        return x


    


########### STATS ##############

def print_batch_accuracy(data, targets, train=False):
    logits = net(data.view(batch_size, -1))
    pred_probab = nn.Softmax(dim=1)(logits)
    pred = pred_probab.argmax(1)
    acc = np.mean((targets == pred).detach().cpu().numpy())
    dev = np.std((targets == pred).detach().cpu().numpy())
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
    for data, targets in train_batches: #torch.Size([256, 1, 28, 28]), torch.Size([256])

        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train() #inform pytorch
        logits = net(data.view(batch_size, -1)) #torch.Size([256, 10])

        loss_val = loss(logits, targets)

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
            logits = net(test_data.view(batch_size, -1))
            

            # Test set loss
            test_loss = loss(logits, test_targets)
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
    logits = net(data.view(data.size(0), -1))

    # calculate total accuracy
    pred_probab = nn.Softmax(dim=1)(logits)
    pred = pred_probab.argmax(1)
    total += targets.size(0)
    correct += (pred == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")

print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

 