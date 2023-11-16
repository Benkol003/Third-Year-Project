import torch


dtype=torch.double
class test:

    def __init__(self):
        self.learnRate = 0.08
        self.regulariser = 0.004
        self.noiseTemp = 0.0007
        self.connectivity = 0.7

    def rewiring(self, weights: torch.tensor, weightSignMask: torch.tensor):
        #add regularisation and noise
        #weightDiff = torch.randn_like(weights,dtype=dtype)*((2*self.learnRate*self.noiseTemp)**0.5)
        #weights = weights + weightDiff
        #TODO do i need to multiply the noise by weight sign? its already random

        #remove connections below threshold
        remove_weights = ((weights * weightSignMask) >= 0) 
        # sets connection signs to zero
        weightSignMask = weightSignMask * remove_weights

        #set disabled weights to zero
        weightMask = torch.abs(weightSignMask)
        weights = weights * weightMask

        #calculate connections to activate
        connection_count = torch.numel(weightMask)
        to_activate = int( (self.connectivity - (weightMask.sum()/connection_count) ) * connection_count )

        if to_activate>0:
            zero_indexes = torch.nonzero(weightMask == 0)

            #randomly select disabled weights
            zero_ind_ind = torch.randperm(zero_indexes.size(0))[:to_activate] #this produces indexes of the zero indexes list

            selected_weights = zero_indexes[zero_ind_ind]

            #enable weights selected with a random sign
            new_signs = ((torch.rand(to_activate) < 0.5).float() - 0.5) * 2
            weightSignMask[selected_weights[:,0],selected_weights[:,1]] = new_signs

            #assign initial values to weights equal to the learning rate
            weights[selected_weights[:,0],selected_weights[:,1]] = new_signs * self.learnRate

        return (weights, weightSignMask)


w = torch.full((9,9),-1.,dtype=dtype)
mask = torch.full((9,9),-1.,dtype=dtype)
test1 = test()
(w, mask)= test1.rewiring(w,mask)
print("w:",w)
print("-------")
print("mask: ",mask)

print("connectivity:",torch.sum(torch.abs(mask))/torch.numel(mask))