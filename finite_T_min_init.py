import numpy as np
from numpy import pi, sqrt, exp

# needs to be precalculated only once in the main script
def setting_up_grids__and_k_vect_len(phys_params):
    
    k_max = phys_params["max k-point"]
    grid_points = phys_params["number of k-points"]
    
    k = np.linspace(-k_max, k_max, 2*grid_points-1)
    dk = abs(k[0] - k[1])
    k_measure = dk**2 / (2*pi)**2
    
    dx = 2*pi/(len(k)*dk)
    x_max = dx*(len(k)-1)/2
    x = np.linspace( - x_max, x_max, len(k) )
    
    kk = sqrt(np.add.outer(k*k,k*k))
    
    zero_ind = int( 0.5 * ( len(k) - 1 ) )
    
    num_of_vars = int( 0.5 * len(k) * (len(k) + 1) * 3 + 1 )
    
    return_dict = { "k-grid" : k,
                    "dk" : dk,
                    "x-grid" : x,
                    "dx" : dx,
                    "sqrt(k_x^2 + k_y^2)" : kk,
                    "zero momentum index" : zero_ind,
                    "k_measure" : k_measure,
                    "number of variables" : num_of_vars
                    }

    return return_dict

# needs to be precalculated only once in the main script
def Coulomb_modified(phys_params):
    dx = phys_params["dx"]
    k = phys_params["k-grid"]
    E_0 = phys_params["binding energy E_0"]
    V_q = np.zeros([len(k), len(k)])
    for i in range(len(k)):
        for j in range(len(k)):
            kk = sqrt( k[i]**2 + k[j]**2 )
            if kk > 0:
                V_q[i,j] = E_0 * pi / kk 
            else:
                V_q[i,j] = 0
                
    V_x = np.real( fourier_2d_from_momentum_to_real(V_q, dx) )


    return V_x


# Fermi distribution
def Fermi(energy, bet):
    xx = bet * energy
    #nf = 1 / ( exp(bet * energy) + 1 )
    nf = 0.5 * (1 - np.tanh(xx/2))
    return nf

def Ferm_sq_times_exp(energy, bet):
    xx = bet * energy
    nf = 0.5 * (1 - np.tanh(xx/2))
    x = nf * (1 - nf)
    return x    
    
# log function for A-term
def log_funct(energy, bet):
    temp = 1.0 / bet
    lf = temp * np.log( 1.0 + np.exp(- bet * abs(energy) ) ) + 0.5 * ( abs(energy) - energy )
    return lf

# needs to be precalculated only once in the main script
def cosine(k):
    c = np.zeros([len(k), len(k)])
    for i in range(len(k)):
        k_y = k[i]
        for j in range(len(k)):
            k_x = k[j]
            if k_x == 0.0:
                c[i,j] = 0
            else:
                c[i,j] = k_x / sqrt( k_x**2 +k_y**2 )
    return c


def prepare_data(data, before_fourier):

    '''
    Prepares "physically" ordered in the momentum space data for Fourier transform, i.e. rearranges frequencies/coordinates.
    AND
    Rearranges raw (standardly ordered) data after Fourier transform to "physically" ordered
    
    1st index - y coordinate, 2nd index - x coordinate
    
    set before_fourier = True if need to rearrange data before an FFT
    set before_fourier = True if need to rearrange data after an FFT
    '''
    
    f_kk = data
    len_k = np.shape(data)[0]
    zero_ind = int( 0.5 * ( len_k - 1 ) )
    f_kk_new = np.zeros([len_k, len_k], dtype = complex)
    
    if before_fourier:
        '''
        this is a generalisation of a 1D case
        f_x_new[ : zero_ind + 1] = f_x[zero_ind : ]  # positive and zero
        f_x_new[zero_ind + 1: ] = f_x[  : zero_ind ] # negative
        '''
        f_kk_new[ : zero_ind + 1 , : zero_ind + 1 ] = f_kk[ zero_ind : , zero_ind : ]
        f_kk_new[ zero_ind + 1 : , : zero_ind + 1 ] = f_kk[ : zero_ind , zero_ind : ]
        f_kk_new[ : zero_ind + 1 , zero_ind + 1 : ] = f_kk[ zero_ind : , : zero_ind ]
        f_kk_new[ zero_ind + 1 : , zero_ind + 1 : ] = f_kk[ : zero_ind , : zero_ind ] 
    else:
        '''
        this is a generalisation of a 1D case
        f_k_new[ : zero_ind ] = f_k[ zero_ind + 1 : ]
        f_k_new[ zero_ind : ] = f_k[ : zero_ind + 1 ]
        '''
        f_kk_new[ zero_ind : , zero_ind : ] = f_kk[ : zero_ind + 1 , : zero_ind + 1 ]
        f_kk_new[ zero_ind : , : zero_ind ] = f_kk[ : zero_ind + 1 , zero_ind + 1 : ]
        f_kk_new[ : zero_ind , zero_ind : ] = f_kk[ zero_ind + 1 : , : zero_ind + 1 ]
        f_kk_new[ : zero_ind , : zero_ind ] = f_kk[ zero_ind + 1 : , zero_ind + 1 : ]
        
              

    return f_kk_new


def fourier_2d_from_momentum_to_real(funct_in_momentum_space, dx):

    '''
    Takes functions in the momentum space with a 1D array of k-points
    Returns Fourier transform of these terms (in real space) 
    '''

    # preparing for the Fourier transform 
    funct_in_momentum_space__prepared = prepare_data(funct_in_momentum_space, True)

    # Fourier Transform & remove an "unphysical" (numerical) prefactor
    funct_in_real_space_raw = np.fft.ifft2( funct_in_momentum_space__prepared ) / (dx**2)

    # Rearrange data to have "physical" data 
    funct_in_real_space =  prepare_data(funct_in_real_space_raw, False) 


    return funct_in_real_space


def fourier_2d_from_real_to_momentum(funct_in_real_space, k_points):

    '''
    Takes functions in the real space with a 1D array of x-points
    Returns Fourier transform of these terms (momentum space) 
    '''
    
    # preparing for the Fourier transform 
    funct_in_real_space__prepared = prepare_data(funct_in_real_space, True)

    dk = abs( k_points[0] - k_points[1] )
    N_an = len(k_points) * dk / (2*pi)

    # Fourier Transform & remove an "unphysical" (numerical) prefactor
    funct_in_momentum_space_raw = np.fft.fft2( funct_in_real_space__prepared ) / (N_an**2)

    # Rearrange data to have "physical" data 
    funct_in_momentum_space =  prepare_data(funct_in_momentum_space_raw, False) 


    return funct_in_momentum_space



# From 1D array of variables to physical 2D arrays Gap, eta_e, eta_h 
def unpacking_vars(data, k):
    
    zero_ind = int( 0.5 * ( len(k) - 1 ) )
    
    # filling arrays using symmetry k_y -> - k_y
    '''
    Allowing FS distorsion in x-direction only
    '''
    
    # setting up a 2D gap function
    Gap = np.zeros([len(k), len(k)])
    step = 0
    for i in range(zero_ind, len(k)):
        Gap[i,:] = Gap[2*zero_ind - i, :] = data[step*len(k): (step+1)*len(k)]
        step = step + 1
            
    # setting up a 2d array of eta_e
    eta_e = np.zeros([len(k), len(k)])
    for i in range(zero_ind, len(k)):
        eta_e[i,:] = eta_e[2*zero_ind - i, :] = data[step*len(k): (step+1)*len(k)]
        step = step + 1
            
    # setting up a 2d array of eta_h
    eta_h = np.zeros([len(k), len(k)])
    for i in range(zero_ind, len(k)):
        eta_h[i,:] = eta_h[2*zero_ind - i, :] = data[step*len(k): (step+1)*len(k)]
        step = step + 1
    
    # photon field expectation
    phi = data[-1]      
    
    return Gap, eta_e, eta_h, phi


# From physical 2D arrays Gap, eta_e, eta_h  to 1D array of variables
def packing_vars_to_1D_array(Gap, eta_e, eta_h, phi, k):
    
    zero_ind = int( 0.5 * ( len(k) - 1 ) )
    
    # Combining all vars and converting them into a 1D array
    num_of_vars = int( 0.5 * len(k) * (len(k) + 1) * 3 + 1 )
    tot_vars = np.zeros([num_of_vars])
    step = 0
    for i in range(zero_ind, len(k)):
        tot_vars[step*len(k): (step+1)*len(k)] = Gap[i,:]
        step = step + 1
    for i in range(zero_ind, len(k)):
        tot_vars[step*len(k): (step+1)*len(k)] = eta_e[i,:]
        step = step + 1
    for i in range(zero_ind, len(k)):
        tot_vars[step*len(k): (step+1)*len(k)] = eta_h[i,:]
        step = step + 1

    tot_vars[-1] = phi 
    
    return tot_vars



def aux_calcs(variables, phys_params, q):
    
    '''
    takes 1D array of variables
    returns 2D arryas
    '''
    
    # setting up physical parameters
    k = phys_params["k-grid"]
    eh_mass_rat = phys_params["electron to hole mass ratio"]
    eps = phys_params["background dielectric permittivity"]
    E_0 = phys_params["binding energy E_0"]
    om_0 = phys_params["photon cut-off freq"]
    E_g = phys_params["gap energy"]
    mu_ex = phys_params["excitation chem potential"]
    cs = phys_params["cosine"]
    kk = phys_params["sqrt(k_x^2 + k_y^2)"]
    zero_ind = phys_params["zero momentum index"] # index corresponding to k_y = 0 (or equiv to k_x = 0, of course)
    
 
    # Setting up variables   
    Gap, eta_e, eta_h, phi = unpacking_vars(variables, k)
    
    
    # chem potentials
    mu_e = - E_g + 0.5 * ( mu_ex )
    mu_h = 0.5 * ( mu_ex )
    
    
    # bare kinetic energies
    cosine = cs
    he_mass_rat = 1.0 / eh_mass_rat 
    E_0_electron = 0.25 * (E_0 / (1 + eh_mass_rat)) * ( kk**2 + 0.25*q**2 + kk*q*cosine ) - mu_e
    E_0_hole =     0.25 * (E_0 / (1 + he_mass_rat)) * ( kk**2 + 0.25*q**2 - kk*q*cosine ) - mu_h 
    

    # phot freq
    fine_struct = 1.0 / 137.036
    inv_phot_mass = 0.25 * ( eps / fine_struct )**2 * ( (E_0) / (om_0) )**2
    om_q = om_0 * ( 1.0 + 0.5 * inv_phot_mass * q**2 )

    # half sum and difference of eta-functions
    half_sum_disp = 0.5 * (eta_e + eta_h)
    half_diff_disp = 0.5 * (eta_e - eta_h) 

    
    # Diagonalization parameters
    E_k_unreg = sqrt( half_sum_disp**2 + Gap**2 )
    '''
    regularisation of E_k at k=(0,0) to prevent 1/0 problem
    '''
    small_number = 1e-20
    reg = np.ones([len(k), len(k)]) * small_number
    E_k = E_k_unreg + reg
    u_k = sqrt( 0.5 * (1 + half_sum_disp / E_k) )
    signum_delta = np.heaviside(Gap, 1) - np.heaviside(-Gap, 0)
    v_k = - signum_delta * sqrt( 0.5 * (1 - half_sum_disp / E_k) )

    eps_1 = E_k + half_diff_disp
    eps_2 = E_k - half_diff_disp
    
    
    # temperature-dependent functions
    Temp = phys_params["temperature"]
    bet = 1 / Temp
    
    # Fermi functions
    f_1 = Fermi(eps_1, bet)
    f_2 = Fermi(eps_2, bet)
    
    # electron/hole density and coherences
    el_dens = u_k**2 * f_1 + v_k**2 * ( 1 - f_2  )
    hole_dens = v_k**2 * ( 1 - f_1 ) + u_k**2 * f_2
    coherences = u_k * v_k * (  1 - f_1 - f_2 )
    
    # density difference in the real space n_c
    k_measure = phys_params["k_measure"]
    n_c = k_measure * np.einsum('ij -> ', el_dens - hole_dens)
    
    
    aux_output = {"quasipart energy eps_1" : eps_1,
                  "quasipart energy eps_2" : eps_2,
                  "variable (eta_e + eta_h)" : eta_e + eta_h,
                  "bare electron kinetic energy" : E_0_electron,
                  "bare hole kinetic energy" : E_0_hole,
                  "diag parameter u_k" : u_k,
                  "diag parameter v_k" : v_k,
                  "q-photon energy" : om_q,
                  "photon field expectation" : phi,
                  "E_k energy" : E_k,
                  "Gap function" : Gap,
                  "eta_e variable" : eta_e,
                  "eta_h variable" : eta_h, 
                  "electron density" : el_dens, 
                  "hole density" : hole_dens, 
                  "coherence" : coherences, 
                  "fermi funct epsilon 1" : f_1, 
                  "fermi funct epsilon 2" : f_2,
                  "electron hole density difference in the real space" : n_c}
    
    
    return aux_output




# defining variatinal mean-field free energy
def vMFFE(variables, phys_params, q):    
    
    # setting up physical parameters
    k_points = phys_params["k-grid"]
    dk = phys_params["dk"]
    k_measure = phys_params["k_measure"]
    dx = phys_params["dx"]
    kk = phys_params["sqrt(k_x^2 + k_y^2)"]
    Coulomb = phys_params["Coulomb interaction in real space"]
    
    g_0 = phys_params["matt-light coupling g_0"]
    k_c = phys_params["photon cutoff momentum"]
    
    E_0 = phys_params["binding energy E_0"]

    mu_ex = phys_params["excitation chem potential"]
    
    
    aux_precalculations = aux_calcs(variables, phys_params, q)
    eps_1 = aux_precalculations["quasipart energy eps_1"]
    eps_2 = aux_precalculations["quasipart energy eps_2"]
    E_0_electron = aux_precalculations["bare electron kinetic energy"]
    E_0_hole = aux_precalculations["bare hole kinetic energy"]
    u_k = aux_precalculations["diag parameter u_k"]
    v_k = aux_precalculations["diag parameter v_k"]
    om_q = aux_precalculations["q-photon energy"]
    phi = aux_precalculations["photon field expectation"]
    Gap = aux_precalculations["Gap function"]    
    el_dens = aux_precalculations["electron density"]
    hole_dens = aux_precalculations["hole density"]
    coherences = aux_precalculations["coherence"]
    f_1 = aux_precalculations["fermi funct epsilon 1"]
    f_2 = aux_precalculations["fermi funct epsilon 2"]
    bet = 1 / phys_params["temperature"]

    
    # Phot free term
    Phot_free =  phi**2 * ( om_q - mu_ex )    
    
    # A-term integrand
    A_ind = ( log_funct(eps_1, bet) + log_funct(eps_2, bet) + 
                 eps_1 * f_1 + eps_2 * f_2 )

    # B-term integrand
    B_ind = 2 * phi * u_k * v_k * g_0 * np.exp(-kk/k_c) * ( f_1 + f_2 - 1 )

    # D-free term integrand (without self-energy)
    D_free_ind = E_0_electron * el_dens + E_0_hole * hole_dens

    tot_free_ind = - A_ind + B_ind + D_free_ind

    # summation over k_x and k_y
    tot_free = np.einsum('ij -> ', tot_free_ind) * k_measure + Phot_free

    

    # Fourier transform from momentum to real space
    f_pair_xx = fourier_2d_from_momentum_to_real(coherences, dx)
    f_repe_xx = fourier_2d_from_momentum_to_real(el_dens, dx)
    f_reph_xx = fourier_2d_from_momentum_to_real(hole_dens, dx)
    
    # interaction terms integrand
    Int_terms_integrand = abs(f_pair_xx)**2  + 0.5 * ( abs(f_repe_xx)**2  + abs(f_reph_xx)**2 )

    # summation over x and y
    Interaction =  - dx**2 * np.einsum('ij, ij ->', Int_terms_integrand, Coulomb)

    # sum of all the terms
    free_energy_tot = tot_free + Interaction
    
    
    # additional electrostatic term
    alpha = phys_params['alpha']
    n_0 = phys_params['target particle real space density n_0']
    n_c = aux_precalculations['electron hole density difference in the real space']
    H_es = alpha * ( n_c**2 - 2 * n_0 * n_c )
    
    # combining it with previous terms
    free_energy_tot = free_energy_tot + H_es
    
    
    return free_energy_tot

    
    
def vMFFE_normal(variables, phys_params, q):
    '''
    Returns normal state energy by setting gap function to zero
    '''
    k = phys_params["k-grid"]

    ngrid = int( 0.5 * len(k) * (len(k) + 1))
    
    full_variables = np.zeros(ngrid * 3 + 1)
    full_variables[ngrid:] = variables[ngrid:]
    
    return vMFFE(full_variables, phys_params, q)
    
    

def eta_deriv(param_for_eta_deriv, phys_params, variables, q, eta_e_or_h):
    
    '''
    Calculates derivatives over eta-variables
    '''
    
    sum_eta_e_h = param_for_eta_deriv["sum_eta_e_h"]
    f_1_sq_exp = param_for_eta_deriv["f_1_sq_exp"]
    f_2_sq_exp = param_for_eta_deriv["f_2_sq_exp"]
    E_k = param_for_eta_deriv["E_k"]
    eps_1 = param_for_eta_deriv["eps_1"]
    eps_2 = param_for_eta_deriv["eps_2"]
    u_k = param_for_eta_deriv["u_k"]
    v_k = param_for_eta_deriv["v_k"]
    E_0_hole = param_for_eta_deriv["E_0_hole"]
    E_0_electron = param_for_eta_deriv["E_0_electron"]
    Phi_k = param_for_eta_deriv["Phi_k"]
    phi = param_for_eta_deriv["phi"]
    Gap = param_for_eta_deriv["Gap"]
    bet = param_for_eta_deriv["bet"]
    f_1, f_2 = param_for_eta_deriv["f_1"], param_for_eta_deriv["f_2"]
    pairing_Coulomb_FT_xx = param_for_eta_deriv["pairing_Coulomb_FT_xx"]
    el_dens_Coulomb_FT_xx = param_for_eta_deriv["el_dens_Coulomb_FT_xx"]
    hole_dens_Coulomb_FT_xx = param_for_eta_deriv["hole_dens_Coulomb_FT_xx"]   
    k_measure = param_for_eta_deriv["k_measure"]
    E_0 = param_for_eta_deriv["E_0"]
    zero_ind = param_for_eta_deriv["zero_ind"]
    
    
    if eta_e_or_h:
        N_1_e = f_1_sq_exp * (sum_eta_e_h + 2 * E_k)
        N_2_e = f_2_sq_exp * (sum_eta_e_h - 2 * E_k)
    else:
        N_1_e = f_1_sq_exp * (sum_eta_e_h - 2 * E_k)
        N_2_e = f_2_sq_exp * (sum_eta_e_h + 2 * E_k)
    
    # free terms eta_e derivative
    dFf_deta_e_1 = (bet/(4*E_k)) * ( 
                                     N_1_e * (eps_1 - u_k**2 * E_0_electron + v_k**2 * E_0_hole) + 
                                     N_2_e * (eps_2 - u_k**2 * E_0_hole + v_k**2 * E_0_electron)
                                    )
    dFf_deta_e_2 = (Gap/(4*E_k**3)) * (f_1+f_2-1) * ( 
                                                      Gap * (E_0_electron + E_0_hole) + 
                                                      sum_eta_e_h * Phi_k * phi
                                                        )
    dFf_deta_e_3 = ((Gap*bet)/(4*E_k**2)) * Phi_k * phi * ( N_1_e + N_2_e )
    
    dFf_deta_e = k_measure * ( dFf_deta_e_1 + dFf_deta_e_2 + dFf_deta_e_3 )
    
    
    # Interacting part
    dfp_deta_e = ((sum_eta_e_h*Gap)/(8*E_k**3)) * (1-f_1-f_2) - ((bet*Gap)/(8*E_k**2)) * (N_1_e + N_2_e)
    
    dfe_deta_e = (Gap**2/(4*E_k**3)) * (f_1+f_2-1) - (bet/(4*E_k)) * (u_k**2 * N_1_e - v_k**2 * N_2_e )
    
    dfh_deta_e = (Gap**2/(4*E_k**3)) * (f_1+f_2-1) - (bet/(4*E_k)) * (u_k**2 * N_2_e - v_k**2 * N_1_e )
    
    dFint_deta_e = - 2 * k_measure * np.real(
                                                     dfp_deta_e * pairing_Coulomb_FT_xx + 
                                                     0.5 * dfe_deta_e * el_dens_Coulomb_FT_xx + 
                                                     0.5 * dfh_deta_e * hole_dens_Coulomb_FT_xx
                                                  )
    # electrostatic energy term
    alpha = phys_params['alpha']
    n_0 = phys_params['target particle real space density n_0']
    aux_precalculations = aux_calcs(variables, phys_params, q)
    n_c = aux_precalculations['electron hole density difference in the real space']
    dn_c_deta = - (bet/(4*E_k)) * ( N_1_e - N_2_e )
    electrostat_der = 2 * alpha * dn_c_deta * (n_c - n_0) * k_measure
    
    
    # total eta_e derivative
    dF_deta_e = 2.0 * (dFf_deta_e + dFint_deta_e + electrostat_der)
    dF_deta_e[zero_ind, : ] = 0.5 * dF_deta_e[zero_ind, : ]

    
    return dF_deta_e
    
        


# defining variatinal mean-field free energy derivative
def vMFFE_der(variables, phys_params, q):
    
    # setting up physical parameters
    k_points = phys_params["k-grid"]
    dk = phys_params["dk"]
    k_measure = phys_params["k_measure"]
    x_points = phys_params["x-grid"]
    dx = phys_params["dx"]
    kk = phys_params["sqrt(k_x^2 + k_y^2)"]
    Coulomb = phys_params["Coulomb interaction in real space"]
    
    g_0 = phys_params["matt-light coupling g_0"]
    k_c = phys_params["photon cutoff momentum"]
    
    E_0 = phys_params["binding energy E_0"]
    
    eh_mass_rat = phys_params["electron to hole mass ratio"]
    mu_ex = phys_params["excitation chem potential"]
    zero_ind = phys_params["zero momentum index"] # index corresponding to k_y = 0 (or equiv to k_x = 0, of course)
    bet = 1 / phys_params["temperature"]
    
    # simple precalculations
    aux_precalculations = aux_calcs(variables, phys_params, q)
    eps_1 = aux_precalculations["quasipart energy eps_1"]
    eps_2 = aux_precalculations["quasipart energy eps_2"]
    sum_eta_e_h = aux_precalculations["variable (eta_e + eta_h)"]
    E_0_electron = aux_precalculations["bare electron kinetic energy"]
    E_0_hole = aux_precalculations["bare hole kinetic energy"]
    u_k = aux_precalculations["diag parameter u_k"]
    v_k = aux_precalculations["diag parameter v_k"]
    om_q = aux_precalculations["q-photon energy"]
    phi = aux_precalculations["photon field expectation"]
    E_k = aux_precalculations["E_k energy"]
    Gap = aux_precalculations["Gap function"]
    el_dens = aux_precalculations["electron density"]
    hole_dens = aux_precalculations["hole density"]
    coherences = aux_precalculations["coherence"]
    f_1 = aux_precalculations["fermi funct epsilon 1"]
    f_2 = aux_precalculations["fermi funct epsilon 2"]
    
    # aux Fermi-like functions   
    f_1_sq_exp = Ferm_sq_times_exp(eps_1, bet)
    f_2_sq_exp = Ferm_sq_times_exp(eps_2, bet)

    
    # Various functions arising in derivatives
    N_f = 1 - f_1 - f_2
    Phi_k = g_0 * np.exp(-kk/k_c) 


    
    
    # Setting up Delta_p derivative
    
    # Pairing (coherence), el and hole densities Delta_p derivatives
    dfp_dDeltap = - ((N_f * sum_eta_e_h**2) / (8*E_k**3)) - ((bet * Gap**2)/(2*E_k**2)) * (f_1_sq_exp + f_2_sq_exp)
    
    dfe_dDeltap = ((Gap*sum_eta_e_h)/(4*E_k**3)) * N_f + ((bet*Gap)/(E_k)) * ( v_k**2 * f_2_sq_exp - u_k**2 * f_1_sq_exp )
    
    dfh_dDeltap = ((Gap*sum_eta_e_h)/(4*E_k**3)) * N_f + ((bet*Gap)/(E_k)) * ( v_k**2 * f_1_sq_exp - u_k**2 * f_2_sq_exp )
    
    # free terms Delta_p derivative
    dFfree_dDeltap = k_measure * (
                                    ((bet*Gap)/(E_k)) * (eps_1 * f_1_sq_exp + eps_2 * f_2_sq_exp) - 
        
                                     2 * Phi_k * phi * dfp_dDeltap + 
                            
                                     E_0_electron * dfe_dDeltap + E_0_hole * dfh_dDeltap
    
                                  )
    
    
    # Interacting part 
    el_int_xx = fourier_2d_from_momentum_to_real(el_dens, dx) * Coulomb 
    hole_int_xx = fourier_2d_from_momentum_to_real(hole_dens, dx) * Coulomb 
    pairing_int_xx = fourier_2d_from_momentum_to_real(coherences, dx) * Coulomb 
    
    pairing_Coulomb_FT_xx = fourier_2d_from_real_to_momentum(pairing_int_xx, k_points)
    el_dens_Coulomb_FT_xx = fourier_2d_from_real_to_momentum(el_int_xx, k_points)
    hole_dens_Coulomb_FT_xx = fourier_2d_from_real_to_momentum(hole_int_xx, k_points) 
    
    
    dFint_dDeltap = - 2 * k_measure * np.real (
                                        
                                dfp_dDeltap * pairing_Coulomb_FT_xx  + 
        
                                0.5 * dfe_dDeltap * el_dens_Coulomb_FT_xx  + 
                                        
                                0.5 * dfh_dDeltap * hole_dens_Coulomb_FT_xx 
    
                                        )
    
    # electrostatic energy term
    alpha = phys_params['alpha']
    n_0 = phys_params['target particle real space density n_0']
    aux_precalculations = aux_calcs(variables, phys_params, q)
    n_c = aux_precalculations['electron hole density difference in the real space']
    dnc_dDelta = - ((bet*Gap)/(E_k)) * ( f_1_sq_exp - f_2_sq_exp )
    electrostat_der = 2 * alpha * dnc_dDelta * (n_c - n_0) * k_measure
    
   
    
    
    # total Delta derivative
    dF_dDelta = 2.0 * (dFfree_dDeltap + dFint_dDeltap + electrostat_der)
    dF_dDelta[zero_ind, : ] = 0.5 * dF_dDelta[zero_ind, : ]
    '''
    Notice a trick dye to k_y -> -k_y symmetry here!!!
    
    When a derivative is calculated numerically, for all points with k_y \neq 0, we have a double effect because 
    when we vary only one variable with some finite k_y, what happens in the code is the equivalent variation of 
    two variables with \pm k_y. 
    
    However, when I calculate a derivative analytically, I vary only one given variable and then use the \pm k_y 
    symmetry to fill derivatives arrays only. 
    
    Therefore, the role of the symmetry is quantitatively different in calculations of function and its derivative!
    '''
    
    
    
    param_for_eta_deriv = {"sum_eta_e_h" : sum_eta_e_h,
                           "f_1_sq_exp" : f_1_sq_exp,
                           "f_2_sq_exp" : f_2_sq_exp,
                           "E_k" : E_k,
                           "eps_1" : eps_1,
                           "eps_2" : eps_2, 
                           "u_k" : u_k,
                           "v_k" : v_k,
                           "E_0_hole" : E_0_hole,
                           "E_0_electron" : E_0_electron,
                           "Phi_k" : Phi_k,
                           "phi" : phi, 
                           "Gap" : Gap, 
                           "bet" : bet, 
                           "f_1" : f_1,
                           "f_2" : f_2,
                           "pairing_Coulomb_FT_xx" : pairing_Coulomb_FT_xx, 
                           "el_dens_Coulomb_FT_xx" : el_dens_Coulomb_FT_xx, 
                           "hole_dens_Coulomb_FT_xx" : hole_dens_Coulomb_FT_xx,
                           "k_measure" : k_measure,
                           "E_0" : E_0,
                           "zero_ind" : zero_ind}
    
    dF_deta_e = eta_deriv(param_for_eta_deriv, phys_params, variables, q, True)
    dF_deta_h = eta_deriv(param_for_eta_deriv, phys_params, variables, q, False)
  
    
    # derivative over photonic phi
    d_F_phi = 2 * phi * (om_q - mu_ex) + 2 * k_measure * np.einsum('ij, ij ->', Phi_k, - coherences)
    
    
    # Combining all derivatives and converting them into a 1D array
    tot_der = packing_vars_to_1D_array(dF_dDelta, dF_deta_e, dF_deta_h, d_F_phi, k_points)
    
    
    return tot_der
    
    
    
def vMFFE_normal_der(variables, phys_params, q):
    
    '''
    Normal state free energy derivatives
    '''
    
    k = phys_params["k-grid"]

    ngrid = int( 0.5 * len(k) * (len(k) + 1))
    
    full_variables = np.zeros(ngrid * 3 + 1)
    full_variables[ngrid:] = variables[ngrid:]
    
    full_der = vMFFE_der(full_variables, phys_params, q)
    
    return full_der
    
    
    
def initial_conditions_normal(phys_params, q):
    
    '''
    Creates normal state initial condition
    '''
    
    # setting up physical parameters
    k = phys_params["k-grid"]
    zero_ind = phys_params["zero momentum index"]
    
    # self-energy correction and electrostatic correction to the normal state
    normal_solution = improved_normal_state(phys_params, q)
    
    # converting into a 1D array
    aux_vars = aux_calcs(normal_solution, phys_params, q)
    E_electron = aux_vars['eta_e variable']
    E_hole = aux_vars['eta_h variable']
    Gap_suppressed = np.zeros([ len(k), len(k) ])    
    phot = 0
    norm_state = packing_vars_to_1D_array(Gap_suppressed, E_electron, E_hole, phot, k)
        
    return norm_state





def initial_conditions_coherent(phys_params, q, normal_state):
    
    '''
    Creates initial condition for a coherent state
    '''
    
    # setting up physical parameters
    k = phys_params["k-grid"]
    
    Gap_suppressed = np.zeros([ len(k), len(k) ])
    phot = 0.1 # np.random.rand()
    for i in range(len(k)):
        for j in range(len(k)):
            Gap_suppressed[i,j] = 0.5 / ( 1 + 0.5*(k[i]**2 + k[j]**2) )            

    
    # converting into a 1D array
    aux_vars = aux_calcs(normal_state, phys_params, q)
    E_electron = aux_vars['eta_e variable']
    E_hole = aux_vars['eta_h variable']
    coh_init_0 = packing_vars_to_1D_array(Gap_suppressed, E_electron, E_hole, phot, k)
    
    # Improving gap function initial condition
    vars_updated = variables_with_improved_gap(coh_init_0, phys_params, q, phot) 
    
    
    return vars_updated






def improved_normal_state(phys_params, q):
    
    '''
    Improves normal state initial condition 
    1. At high temperature and zero electrostatic energy adds self-energy correction to 
    kinetic e/h energies
    2. At high temperature goes from zero to target electrostatic energy
    3. Goes from high to target low temperature
    '''
    
    # aux stuff
    alpha_target = phys_params['alpha']
    T_target  = phys_params['temperature']
    k = phys_params['k-grid']
    num_of_vars = int( 0.5 * len(k) * (len(k) + 1) * 3 + 1 ) 
    
    # setting initial alpha = 0.0 and T = 0.5 for starting calculations
    alpha_init = phys_params['alpha'] = 0.0
    T_init = phys_params['temperature'] = 0.5
    
    
    
    # Self-energy correction due to repulsion
    

    # bare electron/hole energies
    aux = aux_calcs(np.zeros([(num_of_vars)]), phys_params, q)
    E_bare_electron = aux['bare electron kinetic energy']
    E_bare_hole = aux['bare hole kinetic energy']
    
    
    # self-energy correction to the bare kinetic energies
    E_electron = self_energy_correction(phys_params, q, E_bare_electron, E_bare_hole, True)
    E_hole = self_energy_correction(phys_params, q, E_bare_electron, E_bare_hole, False)
    
    
    # Electrostatic correction
    
    # alpha-part
    
    # setting alpha-array
    if alpha_target > 2.0:
        num_of_steps = int(2*alpha_target + 3)
    else:
        num_of_steps = 7
    alpha_array = np.linspace(alpha_init, alpha_target, num_of_steps)
    
    # initial conditions  
    Gap = np.zeros([ len(k), len(k) ])
    var0 = packing_vars_to_1D_array(Gap, E_electron, E_hole, 0, k)
    
    # going from an initial to a target alpha
    for i in range(len(alpha_array)):
    
        phys_params['alpha'] = alpha_array[i]
        extra_param = { 'Is coherent?' : False, 'number of minimization repetitons' : 1, 'q-vector' : q }
        solution = get_new_minimum(var0, phys_params, extra_param)
        var0 = solution[1]
    
    
    # temperature part
    
    # setting temperature array
    if T_target >= 0.5:
        temp_array = np.linspace(T_target, T_target, 1)

    if T_target >= 0.1 and T_target < 0.5:
        temp_array = np.linspace(T_init, T_target, 6)
    
    if T_target < 0.1:
        temp_array_high = np.linspace(T_init, 0.1, 6)
        temp_array_low = np.logspace(np.log(0.1) / np.log(10), np.log(T_target) / np.log(10), 6  )
        temp_array = np.append(temp_array_high, temp_array_low[1:])

    # going from high to low temperature 
    for i in range(len(temp_array)):
    
        phys_params['temperature'] = temp_array[i]
        extra_param = { 'Is coherent?' : False, 'number of minimization repetitons' : 1, 'q-vector' : q }
        result_T = get_new_minimum(var0, phys_params, extra_param)
        var0 = result_T[1]
        
        
    return var0





def self_energy_correction(phys_params, q, electron_bare_en, hole_bare_en, electrons_or_holes):
    
    '''
    electrons_or_holes should be  
    True for electrons 
    False for holes
    
    Improves initial conditions for e/h kinetic energies taking into account 
    ee and hh repulsion
    '''
    
    # aux stuff
    k = phys_params["k-grid"]
    dx = phys_params["dx"]
    Coulomb = phys_params["Coulomb interaction in real space"]
    
    if electrons_or_holes:
        bare_energy = electron_bare_en
    else:
        bare_energy = hole_bare_en
        
    
    # calculation of exchange self-energy
    bare_distr = Fermi(bare_energy , 1/phys_params['temperature'] )
    n_ex = fourier_2d_from_momentum_to_real(bare_distr, dx)
    XSE = np.real( fourier_2d_from_real_to_momentum(n_ex * Coulomb, k) )

    # first update 
    updated_energy = bare_energy - XSE 
   
    
    for i in range(100):
        
        # update self energies
        distrib = Fermi(updated_energy, 1/phys_params['temperature'])
        n_ex = fourier_2d_from_momentum_to_real(distrib, dx)
        XSE = np.real( fourier_2d_from_real_to_momentum(n_ex * Coulomb, k) )

        # updating energies        
        updated_energy = bare_energy - XSE  

        
    return updated_energy





    
def variables_with_improved_gap(vars_init, phys_params, q, phi):
    
    '''
    Improves gap function by interating a gap equation taking into account 
    eh attraction only
    '''
    
    k = phys_params["k-grid"]
    Coulomb = phys_params["Coulomb interaction in real space"]
    dx = phys_params["dx"]
    
    aux = aux_calcs(vars_init, phys_params, q)
    C_k = aux["coherence"]
    C_x = fourier_2d_from_momentum_to_real(C_k, dx)
    eta_e = aux["eta_e variable"]
    eta_h = aux["eta_h variable"]
    gap_init = - np.real( fourier_2d_from_real_to_momentum(C_x * Coulomb, k) )
    gap_funct = gap_init

    for i in range(50):

        # update variables accordingly
        vars_updated = packing_vars_to_1D_array(gap_funct, eta_e, eta_h, phi, k)
        aux_new = aux_calcs(vars_updated, phys_params, q)
        C_k = aux_new["coherence"]
        C_x = fourier_2d_from_momentum_to_real(C_k, dx)
        
        # update a gap function
        gap_funct = - np.real( fourier_2d_from_real_to_momentum(C_x * Coulomb, k) )
        
        
    if np.count_nonzero(gap_funct) == 0:
        gap = gap_init
    else:
        gap = gap_funct
        
    vars_updated = packing_vars_to_1D_array(gap, eta_e, eta_h, phi, k)
 
        
    return vars_updated

    
    

# Fourier filtering (smoothing initial conditions)
def removing_high_freq(data, phys_params, smoothing_param):
    '''
    Smoothing parameter should be roughly from 2 to 3
    '''
    
    # k-grid 
    k = phys_params["k-grid"]
    
    # prepare 2D data
    Gap, eta_e, eta_h, phi = unpacking_vars(data, k)
    
    # from momentum to physical real space
    dx = phys_params["dx"]
    Gap_x   = fourier_2d_from_momentum_to_real(Gap, dx)
    eta_e_x = fourier_2d_from_momentum_to_real(eta_e, dx)
    eta_h_x = fourier_2d_from_momentum_to_real(eta_h, dx)
    
    # filtering function
    x = phys_params["x-grid"]
    xx = np.sqrt( np.add.outer(x*x,x*x) )
    y = xx / max(k)    
    s = smoothing_param
    gauss = np.exp( - (y * s)**2 )
    
    # filtering high freq components and FFT back to the momentum space taking real part only
    Gap_k_new   = np.real( fourier_2d_from_real_to_momentum( Gap_x   * gauss, k) )
    eta_e_k_new = np.real( fourier_2d_from_real_to_momentum( eta_e_x * gauss, k) )
    eta_h_k_new = np.real( fourier_2d_from_real_to_momentum( eta_h_x * gauss, k) )
    
    # combining all the data back to a single 1D array
    filtered_data = packing_vars_to_1D_array(Gap_k_new, eta_e_k_new, eta_h_k_new, phi, k)
    
    
    return filtered_data  




    
    
    
    
def get_new_minimum(var_initial, phys_params, other):
    
    ''' 
    returns energy and solution
    '''
    
    from scipy import optimize
    from scipy.optimize import minimize
    
    Coherent = other['Is coherent?']
    N_attempt = int( other['number of minimization repetitons'] )
    q = other['q-vector']

    smoothing_parameter = 3.0 
    solutions = np.zeros([N_attempt+1, len(var_initial)])
    energies = np.zeros([N_attempt+1])
    
    En_normal = lambda var_s: vMFFE_normal(var_s, phys_params, q)
    der_normal = lambda var_s: vMFFE_normal_der(var_s, phys_params, q)
    
    En = lambda var_s: vMFFE(var_s, phys_params, q)
    der = lambda var_s: vMFFE_der(var_s, phys_params, q)
    
    if Coherent:
        energy = En
        derivative = der
    else:
        energy = En_normal
        derivative = der_normal
         
    # first minimization
    ee  = minimize(energy, var_initial, method='TNC',
                             options={'disp': False, 'maxiter': 20000}, jac = derivative, tol = 1e-10)  
    
    energies[0] = ee.fun
    solutions[0,:] = ee.x

    
    # minimizing few times with smooth init condition
    for i in range(N_attempt):
        
        # smoothing
        var_smoothed = removing_high_freq(ee.x, phys_params, smoothing_parameter)
        
        # minimize once again, but starting from smooth initial conditions
        ee  = minimize(energy, var_smoothed, method='TNC',
                                 options={'disp': False, 'maxiter': 20000}, jac = derivative, tol = 1e-10)
        if i == 0:
            epsilon = abs(ee.fun/1000000)
        
        #print('En = ' + str(ee.fun) + ' ' + 'min step ' + str(i+1) + ' out of ' + str(N_attempt))
        #print('\n')
        
        energies[i+1] = ee.fun
        solutions[i+1,:] = ee.x
        
        if i > 1:
            if ( abs(energies[i] - energies[i-1]) < epsilon and 
                abs(energies[i] - energies[i-2]) < epsilon and 
                abs(energies[i-1] - energies[i-2])< epsilon ):
                break
        if i >= N_attempt - 1:
            #print('does not seem to have converged')
            break
            
    j = np.argmin(energies)
    solution = solutions[j,:]
    energy = energies[j]

    return energy, solution



def el_hole_dens_and_coherence(var, phys_params, q):

    aux_coh = aux_calcs(var, phys_params, q)
    k_measure = phys_params["k_measure"]

    el_dens = aux_coh["electron density"]
    hole_dens = aux_coh["hole density"]
    coherence = aux_coh["coherence"]
    gap = aux_coh['Gap function']
    
    Ne = np.einsum('ij -> ', el_dens) * k_measure  # real space density
    Nh = np.einsum('ij -> ', hole_dens) * k_measure  # real space density
    Nph = aux_coh['photon field expectation']**2
    
    Ne_minus_Nh = Ne - Nh
    Ne_plus_Nh  = Ne + Nh
    fractional_charge = (Ne - Nh) / (Ne + Nh)
    
    phi_sq = Nph 
    phi_sq_over_tot_dens = phi_sq / (phi_sq + 0.5 * (Ne + Nh))
    
    COS = phys_params["cosine"]
    
    anisotr_param = np.einsum('ij, ij -> ', el_dens, COS ) / np.einsum('ij ->', el_dens)
    
    epsilon_1 = aux_coh['quasipart energy eps_1']
    epsilon_2 = aux_coh['quasipart energy eps_2']
    
    
    results = {'electron density' : el_dens,
               'hole density' : hole_dens, 
               'coherence' : coherence, 
               'gap function' : gap, 
               'eh real space density difference' : Ne_minus_Nh, 
               'Ne plus Nh' : Ne_plus_Nh,
               'phi squared' : phi_sq, 
               'photon fraction' : phi_sq_over_tot_dens,
               'fract charge' : fractional_charge,
               'anisotropy parameter' : anisotr_param,
               'eps_1' : epsilon_1,
               'eps_2' : epsilon_2}
    
    return results



def which_coherent_state(el_dens, hole_dens, coherence, phys_params):
    
    k_measure = phys_params["k_measure"]

    Ne = np.einsum('ij -> ', el_dens) * k_measure  # real space density
    Nh = np.einsum('ij -> ', hole_dens) * k_measure  # real space density
    charge = Ne - Nh

    ne_minus_nh_dens = el_dens - hole_dens
    charge_threshold = 0.01
    l = len(el_dens[0,:])
    coh_threshold = 0.01 

    if np.einsum('ij ->', abs(coherence)/ l**2) < coh_threshold:
        # no coherence -> normal state
        state = 7
    else:   
        # charged or not charged
        if charge < charge_threshold:
            # coherence, but no imbalance -> balanced condensate
            state = 1
        else:
            # existense of unpaired electrons
            for i in range(l):
                for j in range(l):
                    if abs(coherence[i,j]) < 0.01:
                        coherence[i,j] = 0
            finite_coh = np.where(coherence == 0)
            if len(finite_coh[0]) > 0 and len(finite_coh[1]) > 0:
                imb_in_incoh_region = sum( ne_minus_nh_dens[ finite_coh ] )
                if imb_in_incoh_region > 0:
                    norm_electrons = True
                else:
                    norm_electrons = False
            else:
                # coherence everywhere
                norm_electrons = False
       
          

            # coherence and imbalance -> imbalanced condensate
            if not norm_electrons:
                # all electrons are paired
                state = is_isotropic(ne_minus_nh_dens, phys_params) + 1

            else:
                # unpaired electrons exist
                state = is_isotropic(ne_minus_nh_dens, phys_params) + 3
                
    return(int(state))



def is_isotropic(data, phys_params):

    cs = phys_params["cosine"]
    check = np.einsum('ij, ij -> ', data, cs)
    if abs(check) < 0.01:
        state = 2 # isotropic
    else:
        state = 1 # anisotropic 
  
    return int(state)
            
            
