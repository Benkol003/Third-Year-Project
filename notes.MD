### DEEP R
is more sensitive to the initial learning rate - ie 1, 1e-1 are slow but 5e-1 is ideal; however still capable of converging fairly quickly to high accuracy (~80%) aftera  few batches
For lower connectivity need to increase the learning rate approx $\frac{lr}{connectivity\%}$, and is much higher even with 100% connectivity.  
For very low connectivity will produce instability spikes in loss
Not much loss in accuracy



TODO look at DVS gesture
https://research.ibm.com/interactive/dvsgesture/

DVS 1 (abandoned) - accuracy ~77%


DVS2 1200 iter 5e-4 lr - 84% accuracy
DVS2 1200 iter 1e-4 lr - 85% accuracy



### Line Profiler

#### num_workers = 1

Timer unit: 1e-07 s

Total time: 1.63687 s
File: C:\Users\Benko\AppData\Local\Temp\ipykernel_11124\3287360524.py
Function: test_pass at line 1

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
     1                                           def test_pass(net):
     2                                               
     3         1    7332919.0    7e+06     44.8      data, targets = next(iter(test_loader))
     4         1      13906.0  13906.0      0.1      optimiser = torch.optim.Adam(net.parameters(),lr=lr)
     5                                               
     6         1       4000.0   4000.0      0.0      net.train()
     7         1    8224698.0    8e+06     50.2      spk_rec = net(data.to(device))
     8         1     531444.0 531444.0      3.2      net.deepr.update(device=device)
     9         1       3139.0   3139.0      0.0      optimiser.zero_grad()
    10         1     255923.0 255923.0      1.6      loss_val = net.loss(spk_rec,targets.to(device))
    11         1       2651.0   2651.0      0.0      optimiser.step()

Total time: 0.767119 s
File: C:\Users\Benko\AppData\Local\Temp\ipykernel_11124\43336395.py
Function: forward at line 47

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    47                                               def forward(self,x): #x is input data
    48                                                   #events should be treated as spikes i.e. already encoded
    49                                           
    50                                                   # Initialize hidden states
    51         1         22.0     22.0      0.0          mem = []
    52         9         93.0     10.3      0.0          for i in range(8):
    53         8       6719.0    839.9      0.1              mem.append(self.neurons[i].init_leaky())
    54                                                   
    55                                                   # record spike outputs
    56         1          6.0      6.0      0.0          spk_rec = []
    57                                           
    58        61        305.0      5.0      0.0          for step in range(num_steps):
    59                                                       #form inputs
    60        60      18446.0    307.4      0.2              spk_i = x[:,step]
    61                                           
    62       540       5349.0      9.9      0.1              for i in range(8):
    63       480       2645.0      5.5      0.0                  if(i==6): #need to flatten from AvgPool to Linear
    64        60      13625.0    227.1      0.2                      spk_i = torch.flatten(spk_i,start_dim=1)
    65       480    3377385.0   7036.2     44.0                  cur_i = self.layers[i](spk_i)
    66       480    4162515.0   8671.9     54.3                  spk_i, mem[i] = self.neurons[i](cur_i,mem[i])
    67                                           
    68        60        718.0     12.0      0.0              spk_rec.append(spk_i)
    69                                                       
    70                                           
    71                                           
    72         1      83366.0  83366.0      1.1          return torch.stack(spk_rec, dim=0) #TODO do i really need to return membrane potentials

Total time: 0.0531018 s
File: c:\Users\Benko\git\benkol003\SNN\deepr.py
Function: update at line 35

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    35                                               def update(self, device : Optional[torch.device] = None):
    36         1         17.0     17.0      0.0          if not self.training: return
    37         2        414.0    207.0      0.1          with torch.no_grad():
    38         6         33.0      5.5      0.0              for i in self.layersIndicies:
    39         5     528871.0 105774.2     99.6                  (new_w,new_wsm) = self._rewiring(self.layersList[i].weight,self.weightSignMasks[i],device)
    40         5       1467.0    293.4      0.3                  self.layersList[i].weight.data=new_w #dont use weight = nn.Parameter, or it goes haywire
    41         5        216.0     43.2      0.0                  self.weightSignMasks[i]=new_wsm

Total time: 0.0527134 s
File: c:\Users\Benko\git\benkol003\SNN\deepr.py
Function: _rewiring at line 43

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    43                                               def _rewiring(self, weights: torch.tensor, weightSignMask: torch.tensor, device : Optional[torch.device] = None):
    44                                                   #add regularisation and noise
    45                                                   #weightDiff = torch.randn_like(weights,dtype=DTYPE,device=device)*((2*self.learnRate*self.noiseTemp)**0.5) #- self.regulariser
    46                                                   #weights = weights + weightDiff
    47                                           
    48                                                   #remove connections below threshold
    49         5       3673.0    734.6      0.7          remove_weights = ((weights * weightSignMask) >= 0)
    50                                                   
    51         5       2315.0    463.0      0.4          weightSignMask = weightSignMask * remove_weights
    52                                           
    53                                                   #set disabled weights to zero
    54         5      22156.0   4431.2      4.2          weightMask = torch.abs(weightSignMask)
    55         5       1354.0    270.8      0.3          weights = weights * weightMask
    56                                           
    57                                                   #calculate connections to activate
    58         5        346.0     69.2      0.1          connection_count = torch.numel(weightMask)
    59         5     139162.0  27832.4     26.4          to_activate = int( (self.connectivity - (weightMask.sum()/connection_count) ) * connection_count )
    60                                           
    61         5         32.0      6.4      0.0          if to_activate>0:
    62         3     200505.0  66835.0     38.0              zero_indexes = torch.nonzero(weightMask == 0) #shape (no of zero elements, no. of weightSignMask dimensions)
    63                                           
    64                                                       #randomly select disabled weights
    65         3      50218.0  16739.3      9.5              zero_ind_ind = torch.randint(0, zero_indexes.size(0), (to_activate,), device=device) #this produces indexes of the zero indexes list TODO use randint
    66                                                       
    67                                           
    68         3      84279.0  28093.0     16.0              selected_weights = zero_indexes[zero_ind_ind]
    69                                           
    70                                                       #enable weights selected with a random sign
    71         3      16072.0   5357.3      3.0              new_signs = ((torch.rand(to_activate,device=device) < 0.5).float() - 0.5) * 2
    72                                                       #TODO generalise this to weight matrix of n dimensions:
    73         3         71.0     23.7      0.0              if(weightSignMask.dim()==2):
    74         1        553.0    553.0      0.1                  weightSignMask[selected_weights[:,0],selected_weights[:,1]] = new_signs
    75                                                           #assign initial values to weights equal to the learning rate
    76         1        651.0    651.0      0.1                  weights[selected_weights[:,0],selected_weights[:,1]] = new_signs * self.learnRate          
    77                                                   
    78         2         11.0      5.5      0.0              elif(weightSignMask.dim()==3): #conv1d
    79                                                           weightSignMask[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2]] = new_signs
    80                                                           #assign initial values to weights equal to the learning rate
    81                                                           weights[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2]] = new_signs * self.learnRate
    82                                           
    83         2         14.0      7.0      0.0              elif(weightSignMask.dim()==4): #conv2d
    84         2       3697.0   1848.5      0.7                  weightSignMask[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2],selected_weights[:,3]] = new_signs
    85         2       1991.0    995.5      0.4                  weights[selected_weights[:,0],selected_weights[:,1],selected_weights[:,2],selected_weights[:,3]] = new_signs * self.learnRate
    86                                                          
    87                                                       else:
    88                                                           raise NotImplementedError("Not generalised enabling weights to n dimensions.")
    89                                           
    90                                           
    91         5         34.0      6.8      0.0          return (weights, weightSignMask)


#### Num_workers = 8
