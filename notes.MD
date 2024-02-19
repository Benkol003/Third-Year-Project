
### Spiking

Learning rate changes (i forgot how) with changing the no. of timesteps. 

### DEEP R
is more sensitive to the initial learning rate - ie 1, 1e-1 are slow but 5e-1 is ideal; however still capable of converging fairly quickly to high accuracy (~80%) aftera  few batches
For lower connectivity need to increase the learning rate approx $\frac{lr}{connectivity\%}$, and is much higher even with 100% connectivity.  
For very low connectivity will produce instability spikes in loss
Not much loss in accuracy



TODO look at DVS gesture
https://research.ibm.com/interactive/dvsgesture/