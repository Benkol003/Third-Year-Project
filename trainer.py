import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Optional

from tqdm.auto import tqdm

import os

import matplotlib.pyplot as plt

def test_stats(net : nn.Module,test_loader,iterations = None,device=None):

    r"""
    Evaluate accuracy of model on test dataset.
    """

    if iterations == None:
        iterations = len(test_loader)

    test_loader = iter(test_loader)

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
            

    accuracy = sum(accuracy)/len(accuracy)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

def trainer(net : nn.Module, 
            train_loader: DataLoader,
            valid_loader: DataLoader,
            model_path : str, 
            lr : float, 
            epochs : int = 1, 
            iterations : Optional[int] = None,
            test_iterations : int = 50, 
            gradient_accumulation : int = 1, #TDO not implemented yet
            deepr : bool = False, 
            device: torch.device = None):
    
    r"""
    Function that trains a ML model with provided dataloaders & parameters.
    If both epochs & iteration args are provided, will run for whichever is shorter.
    In-training validation only calculates accuracy for a single batch - the precision of the validation accuracy value will be related to the batch size.
    """
    
    net  = net.to(device)
    optimiser = torch.optim.Adam(net.parameters(),lr)

    if iterations == None:
        iterations = len(train_loader) * epochs

    loss_hist = []
    valid_loss_hist = []
    valid_count = 0

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

            #backward pass TODO grad accumulation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if deepr:
                net.deepr.update(device=device)

            #store train loss history
            loss_hist.append(loss.item()) #maybe use dict instead to avoid axis scaling

            #Test set evaluation
            if (iteration)%test_iterations == 0:
                with torch.no_grad():
                    net.eval()

                    #TODO this shit is replicated
                    #TODO uh should i use one sample or multiple here
                    valid_data, valid_targets = next(valid_loader)#TODO may need iter
                    valid_data=valid_data.to(device)
                    valid_targets=valid_targets.to(device)
                    valid_logits = net(valid_data)
                    valid_loss = net.loss(valid_logits,valid_targets)

                    valid_loss_hist.append(valid_loss.item())

                    tqdm.write(f"Iteration: {iteration}")
                    tqdm.write(f"Training loss: {loss.item():.2f}")
                    tqdm.write(f"Validation loss: {valid_loss.item():.2f}")

                    valid_acc = net.accuracy_metric(valid_logits,valid_targets)
                    tqdm.write(f"Validation accuracy: {valid_acc*100:.2f}%")

                    acc = net.accuracy_metric(logits,targets)
                    tqdm.write(f"Training accuracy: {acc*100:.2f}%")
                    tqdm.write("----------------")

                    valid_count +=1

            iteration +=1

    if os.path.isfile(model_path):
        os.remove(model_path)
    torch.save(net.state_dict(),model_path)

    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(loss_hist)
    plt.plot(range(0,iterations,test_iterations),valid_loss_hist)
    plt.title("Loss Curves - "+model_path)
    plt.legend(["Train Loss", "Validation Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    return net