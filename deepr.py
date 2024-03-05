class WeightMasker(nn.Module):

    #TODO send tensors to CUDA

    def __init__(self, linearsList : nn.ModuleList, learnRate, temp = 1e-5, alpha = 1e-5, connectivity = 0.07):
        super().__init__()
        self.learnRate = learnRate
        self.regulariser = alpha * learnRate
        self.noiseTemp = temp
        self.connectivity = connectivity
        self.weightSignMasks = [] #element of 1/-1 means connection active
        self.linearsList = linearsList

        with torch.no_grad(): #is this needed?
        #initialise with weight connectivity %
            for i in self.linearsList:
                print("initial zero weights:",torch.sum(i.weight==0).item())
                weightMask = torch.bernoulli(torch.full_like(i.weight,self.connectivity,dtype=DTYPE,device=device))

                i.weight.data = i.weight * weightMask #hadamard product
                self.weightSignMasks.append( torch.sign(i.weight) * weightMask)


    def deepr_update(self):
        if not self.training: return
        with torch.no_grad():
            for i in range(len(self.linearsList)):
                (new_w,new_wsm) = self.rewiring(self.linearsList[i].weight,self.weightSignMasks[i])
                self.linearsList[i].weight.data=new_w #dont use weight = nn.Parameter, or it goes haywire
                self.weightSignMasks[i]=new_wsm

    def rewiring(self, weights: torch.tensor, weightSignMask: torch.tensor):
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
            zero_indexes = torch.nonzero(weightMask == 0)

            #randomly select disabled weights
            zero_ind_ind = torch.randint(0, zero_indexes.size(0), (to_activate,), device=device) #this produces indexes of the zero indexes list TODO use randint
            

            selected_weights = zero_indexes[zero_ind_ind]

            #enable weights selected with a random sign
            new_signs = ((torch.rand(to_activate,device=device) < 0.5).float() - 0.5) * 2
            weightSignMask[selected_weights[:,0],selected_weights[:,1]] = new_signs

            #assign initial values to weights equal to the learning rate
            weights[selected_weights[:,0],selected_weights[:,1]] = new_signs * self.learnRate

        return (weights, weightSignMask)