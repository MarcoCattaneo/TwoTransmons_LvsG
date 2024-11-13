# TwoTransmons_LvsG
Code for finding the steady state and transmission coefficients of two transmons in a waveguide interacting with both local and global thermal baths. The code is based on the theoretical model for the experiment described in *Sharafiev, Juan, Cattaneo and Kirchmair, Leveraging collective effects for thermometry in waveguide quantum electrodynamics, preprint* [*arXiv:2407.05958*](http://arxiv.org/abs/2407.05958)  *(2024)*.

### Required packages:

numpy                     1.24.4<br /> 
matplotlib                3.1.2               
scipy                     1.6.3               
qutip                     4.7.5    

### Usage
The file functions.py contains all the functions to create the Liouvillian driving the master equation in partial secular approximation for two transmons in a waveguide under the action of both a global symmetric bath and a local asymmetric bath and in the presence of an external driving. The locality (ratio between the local decay rates of each transmon) can be freely tuned. Moreover, the function transm_coeff in functions.py computes the steady state of the master equation and the transmission coefficients at the steady state.

The notebook example.ipynb shows how to use these functions to reproduce, as an example, the plot in Fig. G.1 of the paper.  
