import torch
import torch.nn as nn
from typing import Optional
from numpy.random import multinomial

#TODO autodetection os layer indicies from the layer type - net.parameters(), requires_grad

class DeepR(nn.Module):
    r'''
    Implementation of the DeepR algorithm.
    #this version calculates and enables weights across the entire model rather than per-layer. This should allow layers to have diferent connection percentages.
    '''

    @torch.no_grad()
    def __init__(self, layersList : nn.ModuleList, learnRate: float, layersIndicies : Optional[list] = None, connectivity : float = 0.5, device : Optional[torch.device] = None):
        super().__init__()

        if layersIndicies == None: 
            layersIndicies = range(len(layersList))
        self.layersIndicies = layersIndicies
        
        self.learnRate = learnRate
        self.connectivity = connectivity
        self.weightSignMasks = dict() #element of 1/-1 means connection active. using dict as not all layers may have weights i.e. wont use all indicies
        self.layersList = layersList

    #initialise with weight connectivity %
        for i in self.layersIndicies:
            weightMask = torch.bernoulli(torch.full_like(self.layersList[i].weight,self.connectivity,dtype=torch.float32,device=device))
            self.layersList[i].weight.data = self.layersList[i].weight * weightMask #hadamard product
            self.weightSignMasks[i]=torch.sign(self.layersList[i].weight) * weightMask
    
    
    @torch.no_grad()       
    def update(self, device : Optional[torch.device] = None):

        #calculate connectivity globally
        weight_count = 0
        connection_count = 0
        inactive_weight_dist = []
        for i in self.layersIndicies:
            weights = self.layersList[i].weight
            weightSignMask = self.weightSignMasks[i]
            #remove connections below threshold
            remove_weights = ((weights * weightSignMask) >= 0)
            weightSignMask = weightSignMask * remove_weights
            #set disabled weights to zero
            weightMask = torch.abs(weightSignMask)
            weights = weights * weightMask

            #calculate connections to activate
            active_connections = weightMask.sum()
            connection_count += active_connections
            inactive_weight_dist.append( float(torch.numel(weightMask) - active_connections)) 
            weight_count += torch.numel(weightMask)

            #need to update changes vfor the next section
            self.layersList[i].weight.data = weights
            self.weightSignMasks[i] = weightSignMask

        inactive_weight_dist = [iw / sum(inactive_weight_dist) for iw in inactive_weight_dist] #normalise for multinomial dist    
        global_activate = int ( (weight_count * self.connectivity) - connection_count )
        if  global_activate > 0:
            #split connections to activate randomly over all layers using the multinomial distrobution
            conn_split = multinomial(global_activate,inactive_weight_dist)

            for (i,to_activate) in zip(self.layersIndicies,conn_split):
                if to_activate<=0:
                    continue
                weights = self.layersList[i].weight
                weightSignMask = self.weightSignMasks[i]
                weightMask = torch.abs(weightSignMask)

                zero_indexes = torch.nonzero(weightMask == 0) #shape (no of zero elements, no. of weightSignMask dimensions)

                #randomly select disabled weights

                zero_ind_ind = torch.randint(0, zero_indexes.size(0), (to_activate,), device=device) #this produces indexes of the zero indexes list
                

                selected_weights = zero_indexes[zero_ind_ind]

                #enable weights selected with a random sign
                new_signs = ((torch.rand(to_activate,device=device) < 0.5).float() - 0.5) * 2
                #TODO generalise this to weight matrix of n dimensions:
                if(weightSignMask.dim()==2):
                    weightSignMask[selected_weights[:,0],selected_weights[:,1]] = new_signs
                    #assign initial values to weights equal to the learning rate
                    weights[selected_weights[:,0],selected_weights[:,1]] = new_signs * self.learnRate          
            
                elif(weightSignMask.dim()==3): #conv1d
                    weightSignMask[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2]] = new_signs
                    #assign initial values to weights equal to the learning rate
                    weights[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2]] = new_signs * self.learnRate

                elif(weightSignMask.dim()==4): #conv2d
                    weightSignMask[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2],selected_weights[:,3]] = new_signs
                    weights[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2],selected_weights[:,3]] = new_signs * self.learnRate
                
                else:
                    raise NotImplementedError("Not generalised enabling weights to n dimensions.")
            
                self.layersList[i].weight.data = weights
                self.weightSignMasks[i] = weightSignMask


@torch.no_grad()
def layer_connection_hist(deepr : DeepR,connectivity : float):
    r"""
    plots a histogram of connectivity percentage per-layer in a model.
    """
    
    conn_hist = []
    for i in deepr.layersIndicies:
        layer = deepr.layersList[i]
        conn_hist.append(float( torch.count_nonzero(layer.weight) / torch.numel(layer.weight)) * 100)
    
    plt.title(f"Connectivity: {connectivity*100}%")
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy %")
    plt.bar(deepr.layersIndicies,conn_hist)
    plt.axhline(y=connectivity*100, color='r', linestyle='--')
    plt.show()