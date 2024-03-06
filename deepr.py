import torch
import torch.nn as nn
from typing import Optional

class DeepR(nn.Module):
    r'''
    Implementation of the DeepR algorithm.
    '''

    #TODO send tensors to CUDA

    def __init__(self, layersList : nn.ModuleList, learnRate: float, layersIndicies : Optional[list] = None,  temp : float = 1e-5, alpha : float = 1e-5, connectivity : float = 0.5, device : Optional[torch.device] = None):
        super().__init__()

        if layersIndicies == None: 
            layersIndicies = range(len(layersList))
        self.layersIndicies = layersIndicies
        
        self.learnRate = learnRate
        self.regulariser = alpha * learnRate
        self.noiseTemp = temp
        self.connectivity = connectivity
        self.weightSignMasks = dict() #element of 1/-1 means connection active. using dict as not all layers may have weights i.e. wont use all indicies
        self.layersList = layersList

        with torch.no_grad(): #is this needed?
        #initialise with weight connectivity %
            for i in self.layersIndicies:
                #print("initial zero weights:",torch.sum(self.layersList[i].weight==0).item())
                weightMask = torch.bernoulli(torch.full_like(self.layersList[i].weight,self.connectivity,dtype=torch.float32,device=device))

                self.layersList[i].weight.data = self.layersList[i].weight * weightMask #hadamard product
                self.weightSignMasks[i]=torch.sign(self.layersList[i].weight) * weightMask

    def update(self, device : Optional[torch.device] = None):
        if not self.training: return
        with torch.no_grad():
            for i in self.layersIndicies:
                (new_w,new_wsm) = self._rewiring(self.layersList[i].weight,self.weightSignMasks[i],device)
                self.layersList[i].weight.data=new_w #dont use weight = nn.Parameter, or it goes haywire
                self.weightSignMasks[i]=new_wsm

    def _rewiring(self, weights: torch.tensor, weightSignMask: torch.tensor, device : Optional[torch.device] = None):
        #add regularisation and noise
        #weightDiff = torch.randn_like(weights,dtype=DTYPE,device=device)*((2*self.learnRate*self.noiseTemp)**0.5) #- self.regulariser
        #weights = weights + weightDiff

        #remove connections below threshold
        remove_weights = ((weights * weightSignMask) >= 0)
        
        weightSignMask = weightSignMask * remove_weights

        #set disabled weights to zero
        weightMask = torch.abs(weightSignMask)
        weights = weights * weightMask

        #calculate connections to activate
        connection_count = torch.numel(weightMask)
        to_activate = int( (self.connectivity - (weightMask.sum()/connection_count) ) * connection_count )

        if to_activate>0:
            zero_indexes = torch.nonzero(weightMask == 0) #shape (no of zero elements, no. of weightSignMask dimensions)

            #randomly select disabled weights
            zero_ind_ind = torch.randint(0, zero_indexes.size(0), (to_activate,), device=device) #this produces indexes of the zero indexes list TODO use randint
            

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


        return (weights, weightSignMask)