#Main logic for MHD solver
import numpy as np
import matplotlib.pyplot as plt
from mhd_generator import MHD_gen
from mhd_postproc import post_proc_disk, disk_plots
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from functools import wraps

#wrapper function for setting event on integrator
def eventAttr():
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               return func(*args, **kwargs)
           wrapper.direction=0
           wrapper.terminal=True
           return wrapper
       return decorator

#initialize generator and mhd system of equations
genr = MHD_gen()

#T Murakami et al. IEEE 2004 Generator Parameters
genr.gen_type = 'cyl'
genr.Te_model = 'fixed'
genr.gas = 'He'
genr.seed = 'Cs'
genr.coil = 'helm'

genr.R0 = 0.25       #[m]   Coil Location
genr.B0 = 3.0        #[T]   Coil field at channel inlet
genr.M0 = 1.8        #      Mach number at channel inlet
genr.pstag = 1.5e5   #[Pa]  Stagnation pressure at channel inlet
genr.Tstag = 2250.   #[K]   Stagnation temperature at channel inlet
genr.r0 = 0.1        #[m]   Radius at channel inlet
genr.kw = 0.02       #      Channel shape factor
genr.w0 = 0.01       #[m]   Channel height at inlet
genr.J0 = 0.885      #      Normalized inlet current
genr.fseed =1.75e-4   #      Seed fraction
genr.chi0 = np.pi/4. #[rad] Swirl angle

#setup problem
genr.setup(debug=True)
[rspn,y0] = genr.norm_init_conds(l = 3.)

#create dummy function for solve_ivp to call
def mhd_sys(r,y,gen):
    gen.alg(r,y)       #get algebraic quantities
    sol = gen.ode(r,y) #get ode rhs
    return sol

#create event that checks if the generator
#is approaching the subsonic transition M ~ 1.2
@eventAttr()
def subsonic(r,y,gen):
     if(gen.gen_type=='cyl'):
        vr = y[0]
        vth = y[1]
        p = y[2]
        v2 = (vr**2 + vth**2)/(1+np.tan(gen.chi0)**2)
     elif(gen.gen_type=='lin'):
        vx = y[0]
        p = y[1]
        v2 = vx**2
     
     M2 = gen.M0**2 * v2 * gen.n / p
     kill = 1.2**2 - M2
     return kill

#run solve_ivp
soln = solve_ivp(mhd_sys,rspn,y0,events=subsonic,max_step=genr.lamb0/4,args=[genr],method='RK45')

print('Success?: ', soln.success)
print(soln.message)

#run post_proc and plot
proc_soln = post_proc_disk(soln,genr)
disk_plots(proc_soln)

