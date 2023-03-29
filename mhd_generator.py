# S. Frank
# Jan. 2023
#
# MHD generator class.
#

#imports
import numpy as np
from scipy.optimize import fsolve

#global constants
clight = 3.0e8 #m/s
e_charge = 1.602e-19 #C
e_mass = 9.1094e-31 #kg
kB = 1.381e-23 #J/K
h_planck = 6.626e-34 #Js

#object defining MHD generator parameters
class MHD_gen():
    def __init__(self):
        
        #initial conditions
        self.gen_type = 'cyl'
        self.B0 = 10.      # [T]   Magnetic Field
        self.M0 = 2.       #       Channel Inlet Mach Number
        self.pstag = 1.e5  # [Pa]  Stagnation Pressure
        self.Tstag = 2500. # [K]   Stagnation Temperature
        self.gas = 'Ar'    #       Choice of Background Gas
        self.seed= 'K'     #       Choice of Seed
        self.kw = 1.       #       Channel Shape Factor
        self.w0 = 0.10     # [m]   Initial Generator Channel Width
        self.fseed = 2e-5  #       Seed Fraction
        self.J0 = 0.9      #       Normalized Channel Inlet Current
        self.lamb0 = 1.    # [m]   Normalized length
        self.S_RF = 0.     # [W]  RF power

        #fixed Te option
        self.Te_fixed = False
        self.Te0 = 5000.
        self.Te_last = 1.
        
        #cyl specific
        self.chi0 = 0.0    # [rad] Swirl Vane Angle
        self.r0 = 0.1      # [m]   Inlet Radius
        self.vr0=0.
        self.coil = 'uni'  # 'uni' uniform or helm 'helmholtz'
        self.R0 = 0.3      # [m] radial location of helmholtz coil 
        
        #lin specific
        self.h0 = self.w0  # [m]   Initial Generator Channel Height
        self.E0 = 0.0      # [m]   Faraday Field
        self.kh = 1.       #       Channel shape factor (height)
        self.vx0=0.
        
        self._setup = False

    # ----------------------------------------------------------------
    # EXTERNAL CALLS
    # ----------------------------------------------------------------
        
    def setup(self, debug=False):
        
        if(self.gas=='Ar'):
            self.mgas = 6.6335e-26
            self.sigen = 0.35e-20 #[m^-2]
        elif(self.gas=='Xe'):
            self.mgas = 2.1801e-25 #[kg]
            self.sigen = 0.35e-20 #[m^-2]
        elif(self.gas=='He'):
            self.mgas = 6.6465e-27 #[kg]
            self.sigen = 5.4e-20 #[m^-2]
        elif(self.gas=='methane'):
            #assumes methane w/ perfect combustion
            self.mgas = (7.3079e-26 + 2*2.9915e-26)/3.0 #[kg]
            self.sigen = (15e-20+2*75e-20)/3.0 #[m^-2]
        elif(self.gas=='coal'):
            #assumes carbon w/ perfect combustion
            self.mgas = 7.3079e-26 #[kg]
            self.sigen = 15e-20 #[m^-2]
            
        #get ionization energy of seed
        if (self.seed=='K'):
            self.EI = 4.34 * e_charge #[J]
            self.mseed = 6.492e-26 #[kg]
            self.sigej = 400.0e-20 #[m^-2]
        elif (self.seed=='Cs'):
            self.EI = 3.89 * e_charge #[J]
            self.mseed = 2.207e-25 #[kg]
            self.sigej = 500.0e-20 #[m^-2]

        #convert stag pressure to in pressure
        fac = (1. + self.M0**2./3.)
        self.p0 = self.pstag / fac**(5./2.)
        self.T0 = self.Tstag / fac
        
        #algebraic quantities
        self.n0 = self.p0/(kB*self.T0) #[m^-3]
        self.ns0 = self.n0*self.fseed #[m^-3]

        if(self.gen_type == 'cyl'):
            self.vr0 = self.M0*np.sqrt(5*self.p0/(3*self.mgas*self.n0))\
                /np.sqrt(1+np.tan(self.chi0)**2) #[m/s]
            self.s0 = self._s(self.r0,False) #[m^2]
            self.M0r = self.M0/np.sqrt(1+np.tan(self.chi0)**2)
        elif(self.gen_type == 'lin'):
            self.vx0 = self.M0*np.sqrt(3*self.p0/(5*self.mgas*self.n0))
            self.s0 = self._s(0.,False) #[m^2]


            
        #if Te0 .ne. T0 then this is a system of transcendental
        #equations that needs to be solved simultaneously.
        if((self.gas == 'coal') or (self.gas=='methane')): #delta -> inf
            self.Te0 = self.T0
            self.ne0 = self._ne_setup(self.Te0)
            self.beta0 = self._beta_setup(self.Te0,self.ne0)
        else: #monoatomic delta -> 1
            [tmp,info,ier,msg] = fsolve(self._Te0_root,self.T0,full_output=True)
            self.Te0 = tmp[0]
            if (ier==1):
                self.ne0 =  self._ne_setup(self.Te0)
                self.beta0 = self._beta_setup(self.Te0,self.ne0)
            else:
                raise Exception('Te0 initialization failed. Root solver did not converge:',msg)

        #if using the fixed Te model fix a few quantities
        if(self.Te_model == 'fixed'):
            self.Te = self.Te0
            
        #normalized length
        if(self.gen_type == 'cyl'):
            self.lamb0 = 1/self.beta0 * (self.n0/self.ne0) * (self.mgas/e_mass) \
                    *(self.vr0/self.nu_t0)
            vth0 = self.vr0*np.tan(self.chi0)
            v2 = self.vr0**2 + vth0**2
            self.P_in = (0.5*self.mgas*self.n0*v2 +1.5*self.p0)*self.vr0*self.s0
            self.mflow = self.mgas*self.n0*self.s0*self.vr0
            self.Th_in = 5197.*self.mflow*self.Tstag #He only right now
        else:
            self.lamb0 = 1/self.beta0 * (self.n0/self.ne0) * (self.mgas/e_mass) \
                    *(self.vx0/self.nu_t0)
            v2 = self.vx0**2
            self.P_in = (0.5*self.mgas*self.n0*v2 +1.5*self.p0)*self.vx0*self.s0
            self.mflow = self.mgas*self.n0*self.s0*self.vx0
            self.Th_in = 5197.*mflow*self.Tstag #He only right now
            
        #tell the code that the generator has been set up
        self._setup = True

        #if in debug mode print the final output values
        if (debug):

            print('Gen. Type   : ', self.gen_type)
            print(' ')
            print('Ion Frac.   : ', self.ne0/self.ns0)
            print('Beta0       : ', self.beta0)
            print('Lambda0     : ', self.lamb0)
            print(' ')
            print('Gas         : ', self.gas)
            print('Seed        : ', self.seed)
            print('Seed Frac.  : ', self.fseed)
            print('P_in    [MW]: ', self.P_in/1e6)
            print('Th_in   [MW]: ', self.Th_in/1e6)
            print('B0       [T]: ', self.B0)
            print('M0          : ', self.M0)
            print('pstag  [MPa]: ', self.pstag/1.e6)
            print('p0     [MPa]: ', self.p0/1.e6)
            print('J0          : ', self.J0)
            if (self.gen_type =='cyl'):
                print('I_H      [A]: ', self.J0 * self.ne0*self.vr0*e_charge*self.s0)
            else:
                print('I_H      [A]: ', self.J0 * self.ne0*self.vx0*e_charge*self.s0)
            print('E0          : ', self.E0)
            print('V_F         : ', self.E0*self.h0*self.vx0*self.B0)
            print('Stag. T  [K]: ', self.Tstag)
            print('MFlow [kg/s]; ', self.mflow)
            print('T0       [K]: ', self.T0)
            print('Te0      [K]: ', self.Te0)
            print(' ')
            if (self.gen_type == 'cyl'):
                print('vr0    [m/s]: ', self.vr0)
                print('vth0   [m/s]: ', self.vr0 * np.tan(self.chi0))
                print('M0rchk [m/s]: ', np.sqrt(3*self.mgas*self.n0*self.vr0**2/(5*self.p0))/self.M0r)  
            else:
                print('vxo    [m/s]: ', self.vx0)
                
            
            print('s0     [m^2]: ', self.s0)
            print('n0    [m^-3]: ', self.n0)
            print('ns0   [m^-3]: ', self.ns0)
            print('ne0   [m^-3]: ', self.ne0)
            print('nu_ei0/nu_t0: ', self.nu_ei0/self.nu_t0)
            print('nu_en0/nu_t0: ', self.nu_en0/self.nu_t0)
            print('nu_es0/nu_t0: ', self.nu_es0/self.nu_t0)
            print('nu_t0       : ', self.nu_t0)


    # MHD_gen.norm_init_conds()
    # intializes normalized initial conditions for the solver
    # ----------------------------------------------------------------
    def norm_init_conds(self,l = 3.0):
        if (self._setup == False):
            raise Exception('Generator was not set up before running solver. please set up the generator object before trying to solve')

        #set initial conditions for integrator
        p_ini = 1.
        
        if(self.gen_type == 'cyl'):            
            vr_ini = 1.
            vth_ini = np.tan(self.chi0)

            r_ini = self.r0/self.lamb0
            r_fin = r_ini * l

            rspn = [r_ini,r_fin]
            if(self.Te_model=='float'):
                y0 = [vr_ini,vth_ini,p_ini]
            elif(self.Te_model=='fixed'):
                y0 = [vr_ini,vth_ini,p_ini,1.0]
            
        else: #gen_type == 'lin'
            vx_ini = 1.
            
            x_ini = 0
            x_fin = l/self.lamb0

            rspn = [x_ini,x_fin]
            if(self.Te_model=='float'):
                y0 = [vx_ini,p_ini]
            elif(self.Te_model=='fixed'):
                y0 = [vx_ini,p_ini,1.0]
        
        #constants used later in code
        self.A_N = 2.415e21*self.Te0**(3/2)/self.ns0
        self.B_N = self.EI/(kB*self.Te0)
        self.n = 1.
        self.ns = 1.
        self.s = 1.
        self.Te = 1.
        self.T = 1.

        return rspn, y0

    # MHD_gen.alg(r,y) & ode
    # > r - normalized position
    # > y - (vr,vth,p) array or (vx,p) array (can now also include s for
    #       fixed Te mode)
    # These wrap the routines below and ensure the correct one is chosen
    # based on generator type
    # ----------------------------------------------------------------
    def alg(self,r,y):
        if(self.gen_type == 'cyl'):
            vr = y[0]
            vth = y[1]
            p = y[2]
            self._alg_cyl(r,vr,vth,p)  
        elif(self.gen_type == 'lin'):
            vx = y[0]
            p = y[1]
            self._alg_lin(r,vx,p)
    
    def ode(self,r,y):
        if(self.Te_model == 'float'):
            if(self.gen_type == 'cyl'):
                vr = y[0]
                vth = y[1]
                p = y[2]
                sol = self._ode_cyl(r,vr,vth,p)
            elif(self.gen_type == 'lin'):
                vx = y[0]
                p = y[1]
                sol = self._ode_lin(r,vx,p)
        elif(self.Te_model == 'fixed'):
            if(self.gen_type == 'cyl'):
                vr = y[0]
                vth = y[1]
                p = y[2]
                s = y[3]
                sol = self._ode_cyl_fixTe(r,vr,vth,p,s)
            
            if(self.gen_type == 'lin'):
                vx = y[0]
                p = y[1]
                s = y[2]
                sol = self._ode_lin_fixTe(r,vx,p,s)
        else:
            raise Exception('select a valid Te_model')
        return sol

    # ----------------------------------------------------------------
    # INTERNAL CALLS
    # ----------------------------------------------------------------
    
    # MHD_gen._alg_cyl(r,vr,vth,p)
    # inputs:
    # > r - normalized position 
    # > vr - normalized radial velocity
    # > vth - normalized poloidal velocity
    # > p - normalized pressure
    # Performs the algebraic calculations for the disk generator at a given step in
    # the ODE solve.
    # ----------------------------------------------------------------
    def _alg_cyl(self,r,vr,vth,p):
        #area and its derivative
        self.s = self._s(r)
        self.ds = self._ds(r)

        #B-field shape factor
        self.xi = self._xi(r)
        if (self.Te_model == 'fixed'):
            self.dxi = self._dxi(r)
        
        #neutral species density relations
        self.n = 1/(self.s*vr)
        self.ns = self.n

        #saha system of equations
        self.T = p/self.n
        if((self.gas=='methane')or(self.gas=='coal')): #delta -> inf
            self.Te = self.T
        elif (self.Te_model == 'float'):
            [tmp,info,ier,msg] = fsolve(self._Te_fun,self.Te_last, \
                                            args = (r,vr,p), full_output=True)
            self.Te = tmp[0]
            self.Te_last = self.Te
        self.ne = self._ne(self.Te)
        self.beta = self._beta(r,self.Te,self.ne)

        
        #ohm's law
        self.Jr = self._Jr()
        self.Jth = self._Jth(vr)
        self.Er = self._Er(vr,vth)

    def _alg_lin(self,x,vx,p):
        #area and its derivative
        self.h = self._h(x)
        self.s = self._s(x)
        self.ds = self._ds(x)

        #neutral species density relations
        self.n = 1/(self.s*vx)
        self.ns = self.n

        #saha system of equations
        self.T = p/self.n
        if((self.gas=='methane')or(self.gas=='coal')): #delta -> inf
            self.Te = self.T
        else:
            [self.Te,info,ier,msg] = fsolve(self._Te_fun,self.Te_last, \
                                            args = (x,vx,p), fulloutput=True)
        self.ne = self._ne(self.Te)
        self.beta = self._beta(x,self.Te,self.ne)

        #Ohm's Law
        self.Jx = self._Jx()
        self.Jy = self._Jy(vx)
        self.Ex = self._Ex(vx)
        self.Ey = self._Ey()
        
    # MHD_gen._ode_cyl(r,vr,vth,p)
    # inputs:
    # > r - normalized position 
    # > vr - normalized radial velocity
    # > vth - normalized poloidal velocity
    # > p - normalized pressure
    # returns:
    # > rhs - right hand sides of the d (vr,vth,p) /dr ODEs
    # Performs the algebraic calculations for the disk generator at a given step in
    # the ODE solve.
    # ----------------------------------------------------------------
    def _ode_cyl(self,r,vr,vth,p):
        #pressure rhs
        dpdr  = -(5./2.)*p*(vth**2/(vr*r) - self.s*self.Jth*self.xi) \
               -(5./2.)*p*vr*self.ds/self.s \
               +(5./3.)*self.M0r**2/(self.beta*self.ne)*(self.Jth**2+self.Jr**2)*self.xi
        dpdr /= (3./2.)*(vr-p*self.s/self.M0r**2)

        #vr rhs
        dvrdr = vth**2/(vr*r) - (3./5.)*(self.s/self.M0r**2)*dpdr - self.Jth*self.s*self.xi

        #vth rhs
        dvthdr = -self.J0*self.xi-vth/r 

        #put everything in solution vector
        rhs = [dvrdr, dvthdr, dpdr]
        return rhs

    # MHD_gen._ode_cyl_fixTe(r,vr,vth,p,s)
    # inputs:
    # > r - normalized position 
    # > vr - normalized radial velocity
    # > vth - normalized poloidal velocity
    # > p - normalized pressure
    # > s - normalized x-sectional area
    # returns:
    # > rhs - right hand side of the d(vr,vth,p,s)/dr ODEs
    # Performs the calculations of the RHS of the ODEs for the disk generator
    # solve. This version of the routine fixes Te by varying x-sectional area
    # s. See note for derivation
    # ----------------------------------------------------------------
    def _ode_cyl_fixTe(self,r,vr,vth,p,s):
        #consts needed for x-section rhs
        A_T = 4 / self.N
        C_T = 5/9 * self.M0r**2
        D_T = 1 - self.n * A_T / (2* np.sqrt(1+self.n*A_T)*(1+np.sqrt(1+self.n*A_T)))
        E_T = D_T * (1-1/(2*self.logLamb))
        F = 2*C_T*self.Jth**2/self.ne**2 * (self.nu_en + self.nu_es0*(self.n - (self.ns0/self.ne0)*D_T*self.ne)+E_T*self.nu_ei)/self.nu_t 
        G = 2*C_T*self.beta**2 * self.Jr/self.ne * (vr - self.Jr/self.ne * ( 1- 1/self.beta**2))
        H = 2*C_T*self.beta**2 * (vr-self.Jr/self.ne)
        #x-section rhs
        dsdr = C_T*self.Jth**2/self.ne**2*self.dxi/self.xi \
             + (1-s/(3*C_T*vr)*(p+F-D_T*G+H))/(vr-p*s/self.M0r**2) \
             * (-(5/3)*p*(vth**2/(vr*r) - s*self.Jth*self.xi) \
                + 2*C_T*self.xi/(self.ne*self.beta)*(self.Jr**2 + self.Jth**2))
        dsdr /= p+F+G*(1-D_T) - (5/3)*p*vr*(1 + s/(3*C_T*vr)*(p+F-D_T*G+H))/(vr-p*s/self.M0r**2)
        dsdr *= -s
        #pressure & velocity eqns.
        self.s = s
        self.ds = dsdr
        [dvrdr,dvthdr,dpdr] = self._ode_cyl(r,vr,vth,p)

        #return rhs
        rhs = [dvrdr,dvthdr,dpdr,dsdr]
        return rhs

    # MHD_gen._ode_lin(x,vx,p)
    # inputs:
    # > x - normalized position 
    # > vx - normalized velocity
    # > p - normalized pressure
    # returns:
    # > rhs - right hand sides of the d (vx,p) /dx ODEs
    # Performs the algebraic calculations for the linear generator at a given step in
    # the ODE solve.
    # ----------------------------------------------------------------    
    def _ode_lin(self,x,vx,p):
        #pressure rhs
        dpdx = (5./2.)*p*self.s*self.Jy - (5./2.)*p*vx*self.ds/self.s\
              +(5./3.)*self.M0**2/(self.beta*self.ne)*(self.Jx**2 + self.Jy**2)
        dpdx /= (3./2.)*(vx-p*self.s/self.M0**2)

        #vx rhs
        dvxdx = -(3./5.)*(self.s/self.M0**2)*dpdx - self.Jy*self.s

        #put everything in solution vector
        rhs = [dvxdx, dpdx]
        return rhs
    
    # Generator Shape Functions
    # ----------------------------------------------------------------

    # MHD_gen._h(r,norm)
    # inputs:
    # > x - position along gen in [m] or unitless
    # > norm (opt=True) - whether or not the function expects normalized
    #                     input and ouput
    # MHD generator height function
    # -----------------------------------------------------------------
    def _h(self,x,norm=True):
        if (norm):
            x *= self.lamb0

        hval = self.h0*(1+self.kh*x)

        if (norm):
            hval /= self.h0

        return hval
    
    # MHD_gen.s(r,norm) #ok
    # inputs:
    # > r - position along gen in [m] or unitless
    # > norm (opt=True) - whether or not the function expects normalized
    #                     input and ouput
    # MHD generator area function
    # -----------------------------------------------------------------
    def _s(self,r,norm=True):

        #denormalize input 
        if (norm):
            r *= self.lamb0
        
        if (self.gen_type == 'cyl'):
            w = self.w0*(1.+self.kw*(r-self.r0))
            area = 4.*np.pi*r*w
        else: #linear generator
            w = self.w0*(1+self.kw*r)
            h = self.h0*(1+self.kh*r)
            area = 4.*w*h

        #normalize output
        if (norm):
            area /= self.s0
            
        return area

    # MHD_gen.ds(r,norm) #ok
    # inputs:
    # > r - position along gen in [m] or unitless
    # > norm (opt=True) - whether or not the function expects normalized
    #                     input and ouput
    # MHD generator area derivative function
    # -----------------------------------------------------------------
    def _ds(self,r,norm=True):
        #denormalize input 
        if (norm):
            r *= self.lamb0
            
        if (self.gen_type == 'cyl'):
            cir = 2*np.pi*r
            dcir = 2*np.pi
            w = 2*self.w0*(1+self.kw*(r-self.r0))
            dw = 2*self.w0*self.kw*r
            darea = cir*dw + dcir * w
        else: #linear generator
            w = self.w0*(1+self.kw*r)
            dw = self.w0*self.kw
            h = self.h0*(1+self.kh*r)
            dh = self.h0*self.kh
            darea = 4*(w*dh + dw*h)

        #normalize output
        if (norm):
            darea *= self.lamb0/self.s0
            
        return darea
    
    # Saha Eqn. Setup (Only runs on setup step)
    # -----------------------------------------------------------------
    def _Te0_root(self,Te):
        ne_tmp = self._ne_setup(Te)
        beta_tmp = self._beta_setup(Te,ne_tmp)
        root = self.T0 - Te + (5./9.)*self.T0*(self.M0**2)\
               /np.sqrt(1+np.tan(self.chi0)**2)\
               *(self.J0**2 + (beta_tmp*(1.-self.J0))**2)
        return root

    def _ne_setup(self,Te):
        N = 2.415e21 * Te**(3/2) * np.exp(-self.EI/(kB*Te))
        return 2*self.ns0/(1+np.sqrt(1+4*self.ns0/N))
    
    def _beta_setup(self,Te,ne):  
        vtn0 = np.sqrt(2*kB*self.T0/self.mgas) 
        vte0 = np.sqrt(2*kB*Te/e_mass)
        vtj0 = np.sqrt(2*kB*self.T0/self.mseed)
        self.nu_en0 = self.n0 * self.sigen * np.sqrt(vte0**2)
        self.nu_es0 = (self.ns0-ne) * self.sigej * np.sqrt(vte0**2)
        lambdad = 69.01 * np.sqrt(Te/ne)
        self.Lambda0 = 4*np.pi*ne*lambdad**3
        self.nu_ei0 = 3.633e-6 * ne * Te**(-3/2) * np.log(self.Lambda0) 
        self.nu_t0 = self.nu_en0 + self.nu_ei0 + self.nu_es0

        out = 1.7588e11 * self.B0/self.nu_t0
        
        return out

    # Saha Eqn. Step
    # -----------------------------------------------------------------
    
    # MHD_gen._Te_fun(Te,r,v,p)
    # inputs:
    # > Te - normalized electron temperature
    # > r - normalized position
    # > v - normalized velocity
    # > p - normalized pressure
    # Calculates Te root equation for _saha_sys
    # -----------------------------------------------------------------
    def _Te_fun(self,Te,r,v,p):
        self.ne = self._ne(Te)
        self.beta = self._beta(r,Te,self.ne)

        if(self.gen_type == 'lin'):
            J2 = self._Jx()**2 + self._Jy(v)**2
        elif(self.gen_type == 'cyl'):
            J2 = self._Jr()**2 + self._Jth(v)**2
            
        f_tmp = self.T - (self.Te0/self.T0)*Te + (5/9)*(self.M0/self.ne)**2 \
               /np.sqrt(1+np.tan(self.chi0)**2)*J2
        return f_tmp
    
    # MHD_gen._ne(r,vr,Te)
    # inputs:
    # > Te - normalized electron temperature
    # Calculates normalized electron density w/ saha
    # -----------------------------------------------------------------
    def _ne(self,Te):
        self.N = self.A_N * Te**(3/2) * np.exp(-self.B_N/Te)
        return (self.ns0/self.ne0) * 2*self.ns \
               /(1+np.sqrt(1+4*self.ns/self.N))

    # MHD_gen._beta(r,v,Te,ne)
    # inputs:
    # > Te - normalized electron temperature
    # > ne - normalized electron density
    # Calculates hall parameter
    # -----------------------------------------------------------------
    def _beta(self,r,Te,ne):
        #update collision freqs
        self.nu_en = self.nu_en0 * self.n * np.sqrt(Te)
        self.nu_es = self.nu_es0 * (self.ns - self.ne0*ne/self.ns0) * np.sqrt(Te)        
        self.logLamb = np.log(self.Lambda0 * (Te)**(3/2)/np.sqrt(ne))
        self.nu_ei = self.nu_ei0 * ne / Te**(3/2)\
              * self.logLamb/np.log(self.Lambda0)
        self.nu_t = self.nu_en + self.nu_es + self.nu_ei

        return 1.7588e11 * self.B0*self.xi/self.nu_t
        
    # Ohm's Law Funcs.
    # -----------------------------------------------------------------

    # Cylindrical Generator
    def _Jr(self): #ok
        return self.J0/self.s

    def _Jth(self,vr): #ok
        return self.beta * (self.ne * vr - self._Jr())

    def _Er(self,vr,vth): #ok
        return self.xi*(self.beta*vr + vth - self._Jr()*(1 + self.beta**2)/(self.beta*self.ne))

    # Linear Generator
    def _Jx(self):
        return self.J0/self.s

    def _Jy(self,vx):
        return self.beta * (self.ne*vx - self._Jx() - self.ne*self._Ey())

    def _Ex(self,vx):
        return -(self._Jx() + self.beta * self._Jy(vx))/(self.beta*self.ne)
    
    def _Ey(self):
        return self.E0/self.h

    # Helmholtz Coil B_0 Shape Factor
    # -----------------------------------------------------------------

    def _xi(self,r):
        if (self.coil == 'helm'):
            tmp = (1.+(self.r0/self.R0)**2)**(3/2)/(1.+(self.lamb0*r/self.R0)**2)**(3/2)
        else:
            tmp = 1.
        return tmp

    def _dxi(self,r):
        if (self.coil == 'helm'):
            tmp = -3*(self.lamb0/self.R0)**2*r*self._xi(r)/(1.+(self.lamb0*r/self.R0)**2)**(1/2)
        else:
            tmp = 0.
        return tmp
