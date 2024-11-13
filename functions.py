import numpy as np
from qutip import *
import scipy as sc

hbar=sc.constants.hbar
kb=sc.constants.Boltzmann

def coth(x):
    return 1./np.tanh(x)

def gammaDown(omega_1,omega_2,T,gamma_0):
    """
    Calculate the thermal emission coefficient for a term in the master equation associated with jump
    frequencies omega_1 and omega_2
    Args:
        omega_1: first jump frequency
        omega_2: second jump frequency
        T: temperature of the thermal bath
        gamma_0: zero-temperature decay constant (decay rate for spontaneous emission)
    Returns: 
        float: the emission coefficient
    """
    return gamma_0/4.*(coth(hbar*omega_1/(2*kb*T))+coth(hbar*omega_2/(2*kb*T))+2) # Here I am using a factor 1/4. to be consistent with your decay rates (theoretically I have 1/2, but it depends on how we define gammaD)
    
def gammaUp(omega_1,omega_2,T,gamma_0):
    """
    Calculate the thermal absorption coefficient for a term in the master equation associated with jump
    frequencies omega_1 and omega_2
    Args:
        omega_1: first jump frequency
        omega_2: second jump frequency
        T: temperature of the thermal bath
        gamma_0: zero-temperature decay constant (decay rate for spontaneous emission)
    Returns: 
        float: the absorption coefficient
    """
    return gamma_0/4.*(coth(hbar*omega_1/(2*kb*T))+coth(hbar*omega_2/(2*kb*T))-2)

def uncoupled_system_Ham(omega_tr1,omega_tr2,omega_drive,beta1,beta2,N_trunc):
    """
    Generate the Hamiltonian for the two-transmon system without transmon-transmon coupling and 
    in the frame rotating with the frequency of the drive
    Args:
        omega_tr1: frequency of transmon 1
        omega_tr2: frequency of transmon 2
        omega_drive: frequency of the driving field
        beta1: anharmonicity of transmon 1
        beta2: anharmonicity of transmon 2
        N_trunc: number of transmon levels we take into account   
    Returns: 
        Qobj: matrix of the uncoupled system Hamiltonian as a quantum object in QuTiP
    """
    a1 = tensor(destroy(N_trunc),qeye(N_trunc))#destroy operator for the first transmon 
    a2 = tensor(qeye(N_trunc),destroy(N_trunc))

    return (omega_tr1-omega_drive)*a1.dag() * a1 + beta1 *a1.dag()*a1.dag()*a1*a1+(omega_tr2-omega_drive)*a2.dag()*a2+beta2*a2.dag()*a2.dag()*a2*a2

def list_freq_ham(omega_tr1,omega_tr2,beta1,beta2,N_trunc):
    """
    Create a list of jump frequencies associated with the local transitions (i.e., generated
    by a_1 or a_2) in the two-transmon systems
    Args:
        omega_tr1: frequency of transmon 1
        omega_tr2: frequency of transmon 2
        beta1: anharmonicity of transmon 1
        beta2: anharmonicity of transmon 2
        N_trunc: number of transmon levels we take into account   
    Returns: 
        list: list_freq[0] (list_freq[1]) is the list of jump frequencies between the levels of transmon 1 (2)
    """
    
    a = destroy(N_trunc)
    
    Ham1 = (omega_tr1)*a.dag() * a + beta1 *a.dag()*a.dag()*a*a
    Ham2 = (omega_tr2)*a.dag() * a + beta2 *a.dag()*a.dag()*a*a
    
    list_freq = []
    
    list_freq_tr1 = []
    list_freq_tr2 = []
    for jj in range(N_trunc-1):
        list_freq_tr1.append(np.abs(expect(Ham1,basis(N_trunc,jj+1))-expect(Ham1,basis(N_trunc,jj))))
        list_freq_tr2.append(np.abs(expect(Ham2,basis(N_trunc,jj+1))-expect(Ham2,basis(N_trunc,jj))))
    
    list_freq.append(list_freq_tr1)
    list_freq.append(list_freq_tr2)
    
    return list_freq

def return_jump_op(N_trunc):
    """
    Create a list of local jump operators for transmon 1 and transmon 2 that generates the (local) jumps 
    between the states |n> and |n+1> (or viceversa) of a single transmon
    Args:
        N_trunc: number of transmon levels we take into account   
    Returns: 
        list: list_jump[0] (list_jump[1]) is the list of jump operators between the levels of transmon 1 (2)
    """
    
    list_jump_singleT = []

    for jj in range(N_trunc-1):
        temp_matrix = np.zeros((N_trunc,N_trunc),dtype=complex)
        temp_matrix[jj,jj+1] = (destroy(N_trunc))[jj,jj+1]
        list_jump_singleT.append(Qobj(temp_matrix))

    list_jump = []

    list_jump_tr1 = []
    list_jump_tr2 = []

    for jj in range(N_trunc-1):
        list_jump_tr1.append((tensor(list_jump_singleT[jj],qeye(4))))
        list_jump_tr2.append((tensor(qeye(4),list_jump_singleT[jj])))

    list_jump.append(list_jump_tr1)
    list_jump.append(list_jump_tr2)

    return list_jump
        
def globalDiss(gamma_glob,T_glob,list_freq,N_trunc):
    """
    Generate the dissipator of the master equation for the global bath. The dissipator is written as a
    superoperator, i.e., a matrix, see for instance Cattaneo et al., Phys. Rev. A 101, 042108 (2020)
    Args:
        gamma_glob: spontaneous emission constant for the global bath
        T_glob: temperature of the global bath
        list_freq: list of jump frequencies associated with the local jump operators
        N_trunc: number of transmon levels we take into account  
    Returns: 
        Qobj: matrix representation of the global dissipator as a quantum object (type=superoperator) in QuTiP
    """
    dissipatorGlob=Qobj(type=super)    
    
    list_jump=return_jump_op(N_trunc)

    for jj in range(2): # loop over the transmons (first jump operator in a term of the master equation)
        for kk in range(N_trunc-1): # loop over the jumps (first jump operator in a term of the master equation)
            for mm in range(2): # loop over the transmons (second jump operator in a term of the master equation)
                for nn in range(N_trunc-1): # loop over the jumps (second jump operator in a term of the master equation)
                    dissipatorGlob+=gammaDown(list_freq[jj][kk],list_freq[mm][nn],T_glob,gamma_glob)*(tensor(list_jump[jj][kk],list_jump[mm][nn])
                    -0.5*tensor((list_jump[mm][nn].trans())*list_jump[jj][kk],tensor(qeye(4),qeye(4)))-0.5*tensor(tensor(qeye(4),qeye(4)),(list_jump[jj][kk].trans())*list_jump[mm][nn]))+gammaUp(list_freq[jj][kk],list_freq[mm][nn],T_glob,gamma_glob)*(tensor((list_jump[jj][kk].trans()),(list_jump[mm][nn].trans()))
                    -0.5*tensor(list_jump[mm][nn]*(list_jump[jj][kk].trans()),tensor(qeye(4),qeye(4)))-0.5*tensor(tensor(qeye(4),qeye(4)),list_jump[jj][kk]*(list_jump[mm][nn].trans())));
    return dissipatorGlob

def localDiss(gamma_loc1,gamma_loc2,T_loc,list_freq,N_trunc):
    """
    Generate the dissipator of the master equation for the local bath. The dissipator is written as a
    superoperator, i.e., a matrix, see for instance Cattaneo et al., Phys. Rev. A 101, 042108 (2020)
    Args:
        gamma_loc1: spontaneous emission constant for the local decay of transmon 1
        gamma_loc2: spontaneous emission constant for the local decay of transmon 2
        T_loc: temperature of the local bath
        list_freq: list of jump frequencies associated with the local jump operators
        N_trunc: number of transmon levels we take into account  
    Returns: 
        Qobj: matrix representation of the local dissipator as a quantum object (type=superoperator) in QuTiP
    """
    dissipatorLoc=Qobj(type=super) 

    list_jump=return_jump_op(N_trunc)
    
    gamma_loc = []
    gamma_loc.append(gamma_loc1)
    gamma_loc.append(gamma_loc2)

    #another local dissipator with temperature T_local
    for jj in range(2): # loop over the transmons (first jump operator in a term of the master equation)
        for kk in range(N_trunc-1): # loop over the jumps (first jump operator in a term of the master equation)
            for mm in range(2): # loop over the transmons (second jump operator in a term of the master equation)
                for nn in range(N_trunc-1): # loop over the jumps (second jump operator in a term of the master equation)
                    dissipatorLoc+=gammaDown(list_freq[jj][kk],list_freq[mm][nn],T_loc,np.sqrt(gamma_loc[jj]*gamma_loc[mm]))*(tensor(list_jump[jj][kk],list_jump[mm][nn])
                    -0.5*tensor((list_jump[mm][nn].trans())*list_jump[jj][kk],tensor(qeye(4),qeye(4)))-0.5*tensor(tensor(qeye(4),qeye(4)),(list_jump[jj][kk].trans())*list_jump[mm][nn]))+gammaUp(list_freq[jj][kk],list_freq[mm][nn],T_loc,np.sqrt(gamma_loc[jj]*gamma_loc[mm]))*(tensor((list_jump[jj][kk].trans()),(list_jump[mm][nn].trans()))
                    -0.5*tensor(list_jump[mm][nn]*(list_jump[jj][kk].trans()),tensor(qeye(4),qeye(4)))-0.5*tensor(tensor(qeye(4),qeye(4)),list_jump[jj][kk]*(list_jump[mm][nn].trans())));

    return dissipatorLoc

    
def Liouvillian(omega_tr1,omega_tr2,omega_drive,beta1,beta2,g,readout_drive,gamma_glob,gamma_loc1,gamma_loc2,T_glob,T_loc,N_trunc):
    """
    Generate the Liouvillian of the master equation including both global and local terms. The Liouvillian is 
    written as a superoperator, i.e., a matrix, see for instance Cattaneo et al., Phys. Rev. A 101, 042108 (2020)
    Args:
        omega_tr1: frequency of transmon 1
        omega_tr2: frequency of transmon 2
        omega_drive: frequency of the driving field
        beta1: anharmonicity of transmon 1
        beta2: anharmonicity of transmon 2
        g: transmon-transmon coupling constant
        readout_drive: magnitude of the readout_drive (mean value of the input field)
        gamma_glob: spontaneous emission constant for the global bath
        gamma_loc1: spontaneous emission constant for the decay of transmon 1
        gamma_loc2: spontaneous emission constant for the decay of transmon 2
        T_glob: temperature of the global bath
        T_loc: temperature of the local bath
        N_trunc: number of transmon levels we take into account  
    Returns: 
        Qobj: matrix representation of the Liouvillian as a quantum object (type=superoperator) in QuTiP
    """    
    a1 = tensor(destroy(N_trunc),qeye(N_trunc))#destroy operator for the first transmon 
    a2 = tensor(qeye(N_trunc),destroy(N_trunc))
    
    Hbare = uncoupled_system_Ham(omega_tr1,omega_tr2,omega_drive,beta1,beta2,N_trunc) 
    H0 = Hbare+g*(a1*a2.dag()+a1.dag()*a2)
 
    list_freq = list_freq_ham(omega_tr1,omega_tr2,beta1,beta2,N_trunc)
    
    E1 = - 1j * np.sqrt(gamma_glob * omega_drive/(2*omega_tr1)) * readout_drive
    E2 = - 1j * np.sqrt(gamma_glob * omega_drive/(2*omega_tr2)) * readout_drive
    H_drive = E1 * a1.dag() + np.conj(E1) * a1+E2 * a2.dag() + np.conj(E2) * a2
    H=H0+H_drive
    
    Liouvillian_superop = Qobj(-1.j*(tensor(H,tensor(qeye(N_trunc),qeye(N_trunc)))-tensor(tensor(qeye(N_trunc),qeye(N_trunc)),(H.trans()))) + globalDiss(gamma_glob,T_glob,list_freq,N_trunc)+localDiss(gamma_loc1,gamma_loc2,T_loc,list_freq,N_trunc) ,dims=[[[N_trunc, N_trunc], [N_trunc, N_trunc]], [[N_trunc, N_trunc], [N_trunc, N_trunc]]])#
        
    return Liouvillian_superop

def transm_coeff(omega_tr1,omega_tr2,omega_drive,beta1,beta2,g,readout_drive,gamma_glob,gamma_loc1,gamma_loc2,T_glob,T_loc,N_trunc):
    """
    Find the steady state of the master equation with driving and the transmission coefficients
    Args:
        omega_tr1: frequency of transmon 1
        omega_tr2: frequency of transmon 2
        omega_drive: frequency of the driving field
        beta1: anharmonicity of transmon 1
        beta2: anharmonicity of transmon 2
        g: transmon-transmon coupling constant
        readout_drive: magnitude of the readout_drive (mean value of the input field)
        gamma_glob: spontaneous emission constant for the global bath
        gamma_loc1: spontaneous emission constant for the decay of transmon 1
        gamma_loc2: spontaneous emission constant for the decay of transmon 2
        T_glob: temperature of the global bath
        T_loc: temperature of the local bath
        N_trunc: number of transmon levels we take into account  
    Returns: 
        float: t2 is the transmission coefficient
        float: t2_dB is the transmission coefficient in dB
        Qobj: steady_state is the steady state of the Liouvillian as a matrix (quantum object in QuTiP)
    """  
    a1 = tensor(destroy(N_trunc),qeye(N_trunc))#destroy operator for the first transmon 
    a2 = tensor(qeye(N_trunc),destroy(N_trunc))

    L = Liouvillian(omega_tr1,omega_tr2,omega_drive,beta1,beta2,g,readout_drive,gamma_glob,gamma_loc1,gamma_loc2,T_glob,T_loc,N_trunc)
    
    steady_state = steadystate(L,method="direct") # Steady state

    operator = np.sqrt(gamma_glob/2) * (a1+a2)
    
    aLout = readout_drive+expect(operator, steady_state)
    t2 = np.abs((aLout/(readout_drive)))**2
    
    t2_dB=10*np.log10(t2)
    
    return [t2,t2_dB,steady_state]

def transm_coeff_fromL(Liouvillian,readout_drive,gamma_glob,N_trunc):
    """
    Find the steady state of the master equation with driving and the transmission coefficients
    Args:
        Liouvillian: Liouvillian superoperator as a quantum object in QuTiP (type=super)
        readout_drive: magnitude of the readout_drive (mean value of the input field)
        gamma_glob: spontaneous emission constant for the global bath
        N_trunc: number of transmon levels we take into account  
    Returns: 
        float: t2 is the transmission coefficient
        float: t2_dB is the transmission coefficient in dB
        Qobj: steady_state is the steady state of the Liouvillian as a matrix (quantum object in QuTiP)
    """  
    a1 = tensor(destroy(N_trunc),qeye(N_trunc))#destroy operator for the first transmon 
    a2 = tensor(qeye(N_trunc),destroy(N_trunc))

    steady_state = steadystate(Liouvillian,method="direct") # Steady state

    operator = np.sqrt(gamma_glob/2) * (a1+a2)
    
    aLout = readout_drive+expect(operator, steady_state)
    t2 = np.abs((aLout/(readout_drive)))**2
    
    t2_dB=10*np.log10(t2)
    
    return [t2,t2_dB,steady_state]
