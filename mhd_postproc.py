import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mhd_generator import MHD_gen

e_charge = 1.602e-19 #C
e_mass   = 9.109e-31 #kg

def post_proc_disk(soln,gen,norm=False):
    r   = soln.t
    vr  = soln.y[0,:]
    vth = soln.y[1,:]
    p   = soln.y[2,:]

    #initialize output arrays
    N = len(r)
    T_arr    = np.empty(N)
    Te_arr   = np.empty(N)
    n_arr    = np.empty(N)
    ns_arr   = np.empty(N)
    ne_arr   = np.empty(N)
    beta_arr = np.empty(N)
    nu_t_arr = np.empty(N)
    S_arr    = np.empty(N)
    dS_arr   = np.empty(N)
    Jr_arr   = np.empty(N)
    Jth_arr  = np.empty(N)
    Er_arr   = np.empty(N)
    M_arr    = np.empty(N)
    xi_arr   = np.empty(N)
    
    #reset before running
    gen.Te_last = 1.
    
    for i in np.arange(N):
        #recalculate array values
        gen.alg(r[i],[vr[i],vth[i],p[i]])
        #fill arrays
        T_arr[i] = gen.T
        Te_arr[i] = gen.Te
        n_arr[i] = gen.n
        ns_arr[i] = gen.ns
        ne_arr[i] = gen.ne
        beta_arr[i] = gen.beta
        nu_t_arr[i] = gen.nu_t
        S_arr[i] = gen.s
        dS_arr[i] = gen.ds
        Jr_arr[i] = gen.Jr
        Jth_arr[i] = gen.Jth
        Er_arr[i] = gen.Er
        xi_arr[i] = gen.xi

        #derived quantities
        M_arr[i] = gen.M0 * np.sqrt((vr[i]**2 + vth[i]**2)
                                    /(1+np.tan(gen.chi0)**2) * gen.n/p[i])

    f = 1 - T_arr[-1] * (3 + M_arr[-1]**2)/(3 + M_arr[0]**2)
    f *= 100.
  
    if norm==False:
        #undo normalizations
        r       *= gen.lamb0
        vr      *= gen.vr0
        vth     *= gen.vr0
        p       *= gen.p0

        T_arr   *= gen.T0
        Te_arr  *= gen.Te0
        n_arr   *= gen.n0
        ns_arr  *= gen.ns0
        ne_arr  *= gen.ne0
        S_arr   *= gen.s0
        dS_arr  *= gen.s0/gen.lamb0
        
        Er_arr  *=-(gen.vr0*gen.B0)
        Jr_arr  *= (gen.vr0*gen.ne0*e_charge)
        Jth_arr *=-(gen.vr0*gen.ne0*e_charge)

        #calculate JdotE power dissipation and generator loading
        JdotE = Er_arr * Jr_arr
        P_l = -integrate.simpson(JdotE*S_arr,r)
        P_c = integrate.simpson(S_arr*(-vr*gen.B0*xi_arr*Jth_arr+vth*gen.B0*xi_arr*Jr_arr),r)

        eta = (e_mass / e_charge**2) * (nu_t_arr / ne_arr)
        ohmic = eta*(Jr_arr**2 + Jth_arr**2)
        P_dis = -integrate.simpson(ohmic*S_arr,r)
        
        Rpdr = (1+beta_arr**2)*eta/S_arr
        R_P = integrate.simpson(Rpdr,r)
        
        I_H = gen.J0*(gen.ne0*gen.vr0*e_charge)*gen.s0
        V_H = -integrate.simpson(Er_arr,r)
        R_H = V_H/I_H

        I_H2 = integrate.simpson(beta_arr*vr*gen.B0*xi_arr + vth*gen.B0*xi_arr,r)
        I_H2 /= R_H + R_P
        
        P_oute = I_H * V_H
        h_in = vr[0]*S_arr[0]*(0.5*gen.mgas*n_arr[0]*(vr[0]**2 + vth[0]**2) + 2.5*p[0])
        h_out = vr[-1]*S_arr[-1]*(0.5*gen.mgas*n_arr[-1]*(vr[-1]**2 + vth[-1]**2) + 2.5*p[-1])
        p_in = vr[0]*S_arr[0]*(0.5*gen.mgas*n_arr[0]*(vr[0]**2 + vth[0]**2) + 1.5*p[0])
        p_out = vr[-1]*S_arr[-1]*(0.5*gen.mgas*n_arr[-1]*(vr[-1]**2 + vth[-1]**2) + 1.5*p[-1])
        #output stagnation pressure
        fac = (1.+M_arr[-1]**2/3.)**(5./2.)
        pstag_out = fac*p[-1]
        pratio = gen.pstag/pstag_out

        ie = 100.*P_l / gen.Th_in / (1 - (1/pratio)**(2/5))
        print('')
        print('Channel Area Ratio: ', S_arr[-1]/S_arr[0])
        print('')
        print('h_in  : ', h_in/1.e6, ' MW')
        print('h_exh : ', h_out/1.e6, ' MW')
        print('Stag p-ratio: ', pratio)
        print('')
        print('P_l:    ', P_l/1.e6, ' MW')
        print('P_c:    ', P_c/1.e6, ' MW')
        print('P_dis:  ', -P_dis/1.e6, 'MW')
        print('')
        print('Hall Current ', I_H/1000.,'kA')
        print('Hall Voltage ', V_H, 'V')
        print('Hall Resis: ', R_H, 'Ohms')
        print('Plas Resis: ', R_P, 'Ohms')
        print("I_H ratio (should be 1): ", I_H/I_H2)
        print('Self Consistency 1: ', abs(P_c - P_l + P_dis)/1.e6, 'MW (should be nearly zero)') #ohm's law
        print('Self Consistency 2: ', abs(h_out+P_l - h_in)/1.e6, 'MW (should be nearly zero)') #energy consv
        print('')
        print('Electrical Effficiency: ', 100.*P_l/P_c, '%') 
        print('Enthalpy Extraction (method 1): ', 100.*(1 - h_out / h_in), '%')
        print('Enthalpy Extraction (method 2): ', f ,'%')
        print('Enthalpy Extraction (Murakami): ', 100.*P_l / gen.Th_in, '%')
        print('Isentropic Efficiency (Murakami): ', ie, '%')
        print('method1/method2: ', 100.*(1 - h_out / h_in)/f)
        
        

    generator_soln = {'norm':norm, 'Npts':N, 'r':r , 'vr':vr, 'vth':vth, 'p':p, \
                      'T':T_arr, 'Te':Te_arr, 'n':n_arr, 'ns':ns_arr, 'ne':ne_arr, \
                      'S':S_arr, 'Er':Er_arr, 'Jr':Jr_arr, 'Jth':Jth_arr, 'M':M_arr,\
                      'beta':beta_arr, 'JdotE': JdotE, 'f':f, 'ohmic':ohmic, 'nu_t':nu_t_arr}

    return generator_soln

def post_proc_lin(soln,gen,norm=False):
    x   = soln.t
    vx  = soln.y[0,:]
    p   = soln.y[1,:]

    #initialize output arrays
    N = len(r)
    T_arr    = np.empty(N)
    Te_arr   = np.empty(N)
    n_arr    = np.empty(N)
    ns_arr   = np.empty(N)
    ne_arr   = np.empty(N)
    beta_arr = np.empty(N)
    nu_t_arr = np.empty(N)
    S_arr    = np.empty(N)
    dS_arr   = np.empty(N)
    Jx_arr   = np.empty(N)
    Jy_arr  = np.empty(N)
    Ex_arr   = np.empty(N)
    Ey_arr   = np.empty(N)
    M_arr    = np.empty(N)
    
    #reset before running
    gen.Te_last = 1.
    
    for i in np.arange(N):
        #recalculate array values
        gen.alg(x[i],[vx[i],p[i]])
        #fill arrays
        T_arr[i] = gen.T
        Te_arr[i] = gen.Te
        n_arr[i] = gen.n
        ns_arr[i] = gen.ns
        ne_arr[i] = gen.ne
        beta_arr[i] = gen.beta
        nu_t_arr[i] = gen.nu_t
        S_arr[i] = gen.s
        dS_arr[i] = gen.ds
        Jx_arr[i] = gen.Jx
        Jy_arr[i] = gen.Jy
        Ex_arr[i] = gen.Ex
        Ey_arr[i] = gen.Ey
        
        #derived quantities
        M_arr[i] = gen.M0*vx[i]*np.sqrt(gen.n/p[i])

    f = 1 - T_arr[-1] * (3 + M_arr[-1]**2)/(3 + M_arr[0]**2)
    f *= 100.
  
    if norm==False:
        #undo normalizations
        x       *= gen.lamb0
        vx      *= gen.vx0
        p       *= gen.p0

        T_arr   *= gen.T0
        Te_arr  *= gen.Te0
        n_arr   *= gen.n0
        ns_arr  *= gen.ns0
        ne_arr  *= gen.ne0
        S_arr   *= gen.s0
        dS_arr  *= gen.s0/gen.lamb0
        
        Ex_arr  *=-(gen.vx0*gen.B0)
        Ey_arr  *= (gen.vx0*gen.B0)
        Jx_arr  *= (gen.vx0*gen.ne0*e_charge)
        Jy_arr  *=-(gen.vx0*gen.ne0*e_charge)

        #calculate JdotE power dissipation and generator loading
        JdotE = Ex_arr * Jx_arr + Ey_arr * Jy_arr
        P_l = -integrate.simpson(JdotE*S_arr,x)
        P_c = -integrate.simpson(S_arr*(vx*gen.B0*Jy_arr),x)

        eta = (e_mass / e_charge**2) * (nu_t_arr / ne_arr)
        ohmic = eta*(Jx_arr**2 + Jy_arr**2)
        P_dis = -integrate.simpson(ohmic*S_arr,x)
        
        Rpdr = (1+beta_arr**2)*eta/S_arr
        R_P = integrate.simpson(Rpdr,x)

        #Hall Gen. Case
        if(gen.J0 != 0.0):
            I_H = gen.J0*(gen.ne0*gen.vx0*e_charge)*gen.s0
            V_H = -integrate.simpson(Ex_arr,x)
            R_H = V_H/I_H

            I_H2 = integrate.simpson(beta_arr*vx*gen.B0 + vth*gen.B0,x)
            I_H2 /= R_H + R_P

        #input and output energy/enthalpy
        h_in = vx[0]*S_arr[0]*(0.5*gen.mgas*n_arr[0]*vx[0]**2 + 2.5*p[0])
        h_out = vx[-1]*S_arr[-1]*(0.5*gen.mgas*n_arr[-1]*vx[-1]**2 + 2.5*p[-1])
        p_in = vx[0]*S_arr[0]*(0.5*gen.mgas*n_arr[0]*vx[0]**2 + 1.5*p[0])
        p_out = vx[-1]*S_arr[-1]*(0.5*gen.mgas*n_arr[-1]*vx[-1]**2 + 1.5*p[-1])

        #output stagnation pressure
        fac = (1.+M_arr[-1]**2/3.)**(5./2.)
        pstag_out = fac*p[-1]
        pratio = gen.pstag/pstag_out

        ie = 100.*P_l / gen.Th_in / (1 - (1/pratio)**(2/5))
        print('')
        print('Channel Expansion Ratio: ', S_arr[-1]/S_arr[0])
        print('')
        print('h_in  : ', h_in/1.e6, ' MW')
        print('h_exh : ', h_out/1.e6, ' MW')
        print('Stag p-ratio: ', pratio)
        print('')
        print('P_l:    ', P_l/1.e6, ' MW')
        print('P_c:    ', P_c/1.e6, ' MW')
        print('P_dis:  ', P_dis/1.e6, 'MW')
        print('')
        if(gen.J0 != 0.0):
            print('Hall Current ', I_H/1000.,'kA')
            print('Hall Voltage ', V_H, 'V')
            print('Hall Resis: ', R_H, 'Ohms')
            print('Plas Resis: ', R_P, 'Ohms')
            print("I_H ratio (should be 1): ", I_H/I_H2)

        print('Self Consistency 1: ', abs(P_c - P_l + P_dis)/1.e6, 'MW (should be nearly zero)') #ohm's law
        print('Self Consistency 2: ', abs(h_out+P_l - h_in)/1.e6, 'MW (should be nearly zero)') #energy consv
        print('')
        print('Enthalpy Extraction (method 1): ', 100.*(1 - h_out / h_in), '%')
        print('Enthalpy Extraction (method 2): ', f ,'%')
        print('Enthalpy Extraction (Murakami): ', 100.*P_l / gen.Th_in, '%')
        print('Isentropic Efficiency (Murakami): ', ie, '%')
        print('method1/method2: ', 100.*(1 - h_out / h_in)/f)

    generator_soln = {'norm':norm, 'Npts':N, 'x':r , 'vx':vx, 'p':p, \
                      'T':T_arr, 'Te':Te_arr, 'n':n_arr, 'ns':ns_arr, 'ne':ne_arr, \
                      'S':S_arr, 'Ex':Ex_arr, 'Ey': Ey_arr, 'Jx':Jx_arr, 'Jy':Jy_arr, 'M':M_arr,\
                      'beta':beta_arr, 'JdotE': JdotE, 'f':f, 'ohmic':ohmic, 'nu_t':nu_t_arr}

def disk_plots(disk):
    f,axs = plt.subplots(nrows=2,ncols=4)
    
    axs[0,0].plot(disk['r'],disk['vr'])
    axs[0,1].plot(disk['r'],disk['vth'])
    axs[0,2].plot(disk['r'],disk['p'])
    axs[0,3].plot(disk['r'],disk['Jr'],label='Jr')
    axs[0,3].plot(disk['r'],disk['Jth'],label='Jth')

    axs[1,0].plot(disk['r'],disk['M'])
    axs[1,1].plot(disk['r'],disk['T']/disk['T'][0],label='T')
    axs[1,1].plot(disk['r'],disk['Te']/disk['Te'][0],label='Te')
    axs[1,2].plot(disk['r'],disk['n']/disk['n'][0],label='n')
    axs[1,2].plot(disk['r'],disk['ne']/disk['ne'][0],label='ne')
    axs[1,3].plot(disk['r'],disk['beta'])

    axs[0,3].legend()
    axs[1,1].legend()
    axs[1,2].legend()
    
    axs[0,0].set_title('vr')
    axs[0,1].set_title('vth')
    axs[0,2].set_title('p')
    axs[0,3].set_title('J')
    axs[1,0].set_title('M')
    axs[1,1].set_title('T (norm)')
    axs[1,2].set_title('n (norm)')
    axs[1,3].set_title('$\\beta$')
    plt.tight_layout()
    plt.show()

def lin_plots(lin):
    f,axs = plt.subplots(nrows=2,ncols=4)

    axs[0,0].plot(lin['x'],lin['vx'])
    axs[0,1].plot(lin['x'],lin['p'])
    axs[0,2].plot(lin['x'],lin['Ex'],label='$\hat{x}$')
    axs[0,2].plot(lin['x'],lin['Ey'],label='$\hat{y}$')
    axs[0,3].plot(lin['x'],lin['Jx'],label='$\hat{x}$')
    axs[0,3].plot(lin['x'],lin['Jy'],label='$\hat{y}$')

    axs[1,0].plot(lin['x'],lin['M'])
    axs[1,1].plot(lin['x'],lin['T']/lin['T'][0],label='T')
    axs[1,1].plot(lin['x'],lin['Te']/lin['Te'][0],label='Te')
    axs[1,2].plot(lin['x'],lin['n']/lin['n'][0],label='n')
    axs[1,2].plot(lin['x'],lin['ne']/lin['ne'][0],label='ne')
    axs[1,3].plot(lin['x'],lin['beta'])

    axs[0,2].legend()
    axs[0,3].legend()
    axs[1,1].legend()
    axs[1,2].legend()
    
    axs[0,0].set_title('vx')
    axs[0,1].set_title('p')
    axs[0,2].set_title('E')
    axs[0,3].set_title('J')
    axs[1,0].set_title('M')
    axs[1,1].set_title('T (norm)')
    axs[1,2].set_title('n (norm)')
    axs[1,3].set_title('$\\beta$')
    
    plt.tight_layout()
    plt.show()
