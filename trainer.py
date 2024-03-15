import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Optional

from tqdm.auto import tqdm

import os

import matplotlib.pyplot as plt

import numpy as np
import random

#set seeding

#set seeds
seed = 115115
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

gtrain = torch.Generator()
gtrain.manual_seed(seed) #need to call this later to reset the generator for reproducibility
gtest = torch.Generator()
gtest.manual_seed(seed)

def gen_reset():
    global gtrain, gtest
    gtrain.manual_seed(seed)
    gtest.manual_seed(seed)


def test_stats(net : nn.Module,test_loader,iterations = None,device=None):

    r"""
    Evaluate accuracy of model on test dataset.
    """

    if iterations == None:
        iterations = len(test_loader)

    with torch.no_grad():
        net.eval()
        accuracy = []
        iteration = 0
        tqdm_bar = tqdm(total=iterations,desc="Testing progress:")
        for (data,targets) in iter(test_loader):
            if iteration >= iterations:
                break
            tqdm_bar.update(1)
            iteration +=1
            data = data.to(device)
            targets = targets.to(device)

            logits = net(data)

            accuracy.append(net.accuracy_metric(logits,targets))
            
    tqdm_bar.close()
    accuracy = sum(accuracy)/len(accuracy)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

def trainer(net : nn.Module, 
            train_loader: DataLoader,
            valid_loader: DataLoader,
            optimiser : torch.optim.Optimizer,
            scheduler : Optional[torch.optim.lr_scheduler.LRScheduler] = None,
            epochs : int = 1, 
            iterations : Optional[int] = None,
            valid_after : int = 50,
            valid_iterations : int = 1,
            deepr : bool = False,
            model_path : str = None, 
            device: torch.device = None):
    
    r"""
    Function that trains a ML model with provided dataloaders & parameters.
    If both epochs & iteration args are provided, will run for whichever is shorter.
    In-training validation only calculates accuracy for a single batch - the precision of the validation accuracy value will be related to the batch size.
    """
    
    net  = net.to(device)

    if iterations == None:
        iterations = len(train_loader) * epochs

    loss_hist = []
    valid_loss_hist = []

    valid_loader = iter(valid_loader)

    iteration = 0

    tqdm_bar = tqdm(total=iterations,desc="Training progress:")
    for epoch in range(epochs):
        if iteration >= iterations:
            break
        for (data,targets) in iter(train_loader):
            if iteration >= iterations:
                break
            tqdm_bar.update(1)
            data = data.to(device)
            targets = targets.to(device)

            #forward pass
            net.train()
            logits = net(data)
            loss = net.loss(logits,targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if deepr:
                net.deepr.update(device=device)
            if scheduler is not None:
                scheduler.step()

            loss_v = loss.item()
            #store train loss history
            loss_hist.append(loss_v) #maybe use dict instead to avoid axis scaling

            #Test set evaluation
            if (iteration)%valid_after == 0:
                with torch.no_grad():
                    net.eval()

                    #TODO this shit is replicated
                    #TODO uh should i use one sample or multiple here
                    valid_loss_v = 0.0
                    valid_accuracy = 0.0
                    for i in range(valid_iterations):
                        valid_data, valid_targets = next(valid_loader)
                        valid_data=valid_data.to(device)
                        valid_targets=valid_targets.to(device)
                        valid_logits = net(valid_data)
                        valid_loss_v += net.loss(valid_logits,valid_targets).item()
                        valid_accuracy += net.accuracy_metric(valid_logits,valid_targets)

                    valid_loss_v/=valid_iterations
                    valid_accuracy/=valid_iterations

                    valid_loss_hist.append(valid_loss_v)
                    if scheduler is not None:
                        tqdm.write(f"LR: {scheduler.get_last_lr()}")
                    tqdm.write(f"Iteration: {iteration}")
                    tqdm.write(f"Training loss: {loss_v:.2f}")
                    tqdm.write(f"Validation loss: {valid_loss_v:.2f}")

                    
                    tqdm.write(f"Validation accuracy: {valid_accuracy*100:.2f}%")

                    acc = net.accuracy_metric(logits,targets)
                    tqdm.write(f"Training accuracy: {acc*100:.2f}%")
                    tqdm.write("----------------")

            iteration +=1

    tqdm_bar.close()
    if model_path is not None:
        if os.path.isfile(model_path):
            os.remove(model_path)
        torch.save(net.state_dict(),model_path)

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(range(0,iterations,valid_after),valid_loss_hist)
    if model_path is None:
        plt.title("Loss Curves")
    else:
        plt.title("Loss Curves - "+model_path)
    plt.legend(["Train Loss", "Validation Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    return net