import jax.numpy as jnp
import jax
from .OTSconstants import *
from .tools.Faddeeva import *
from .tools.CauchyPV import *

#%%########################################################################
#    plasma response integral [p106 Froula2011]

# = 1-2zexp(-z^2)\int_0^z exp(p^2)dp
#determines real factor of plasma response for maxwellian plasma species 
#inputs: z=phase velocity / sqrt(2T/m) 
@jax.jit
def plaZre(z):  
    val=1.0-2.0*z*dawsn(z)
    val = jnp.where(jnp.isnan(val), 0, val)
    val = jnp.where(jnp.isinf(val), 0, val)
    return val    


# = sqrt(pi) z exp(-z^2)
#determines imaginary factor of plasma response for maxwellian plasma species
#inputs: z=phase velocity / sqrt(2T/m) 
@jax.jit
def plaZim(z):
    val = sqrt_pi*z*jnp.exp(-(z**2))
    val = jnp.where(jnp.isnan(val), 0, val)
    val = jnp.where(jnp.isinf(val), 0, val)
    return val

# Custom plasma dispersion function
@jax.jit
def custZprime(v, dists, f_params, maxv, integration_scale = 1.1):  

    dfdv = dists['df/dv']

    # Adjust integration range
    v_scale = integration_scale*maxv
    x = v/v_scale

    # Jitted by lax.scan
    def PV_calc(sum,tau):
        y = KellerWrobel_PV(tau, dfdv, f_params, v_scale)
        sum += y
        return sum, y
    
    _,dWr = jax.lax.scan(PV_calc,jnp.array([0.0]),x)
    dWi = -jnp.pi*dists['df/dv_vmap'](v,*f_params)

    # Safety
    dWr = jnp.where(jnp.isnan(dWr), 0.0, dWr)
    dWr = jnp.where(jnp.isinf(dWr), 0.0, dWr)
    dWi = jnp.where(jnp.isnan(dWi), 0.0, dWi)
    dWi = jnp.where(jnp.isinf(dWi), 0.0, dWi)

    zout = jnp.vstack((dWr.flatten(),dWi.flatten()))
    return zout

#%%########################################################################
#    electron and ion susceptibilty [p50 Froula2011]

#chi e
#chi_e= \omega_pe^2/k^2 \int_{-\infty}^{\infty} (k \cdot df/dv) / (w - kv -igamma)
#determines plasma response to electron denisty fluctuations
#inputs: 
#    kw_dict = dictionary of pre-computed scattering parameters
#    Te = electron temperature [keV]
#    ve=electron flow speed [c]
@jax.jit
def chith_e(kw_dict,ve,Te):
    omgpe = kw_dict['omgpe']
    k     = kw_dict['k']
    v     = kw_dict['vel']

    vte   = jnp.sqrt(Te/me)  #c
    kd    = omgpe/(vte*c)
    kkd   = k/kd

    xe    = (v-ve*c)/(c*vte*jnp.sqrt(2.0))
    imxe  = (kkd**-2)*plaZim(xe)
    rexe  = (kkd**-2)*plaZre(xe)
    return rexe+1j*imxe

#chi_i= \omega_pi^2/k^2 \int_{-\infty}^{\infty} (k \cdot df/dv) / (w - kv -igamma)
#determines plasma response to electron denisty fluctuations
#inputs: 
#    kw_dict = dictionary of pre-computed scattering parameters
#    Te = electron temperature [keV]
#    Ti = ion temperature [keV]
#    Z = ion charge = q/e
#    Ai=ion protons + nuetrons 
#    vi=ion flow speed [c]
@jax.jit
def chith_i(kw_dict,Te,Ti,Z,Ai,vi):
    omgpe = kw_dict['omgpe']
    k     = kw_dict['k']
    v     = kw_dict['vel']
    vte   = jnp.sqrt(Te/me)  #c
    vti   = jnp.sqrt(Ti/(amu_eV*Ai)) #c
    kd    = omgpe/(vte*c)
    kkd   = k/kd
    xi    = (v-vi*c)/(c*vti*jnp.sqrt(2.0))
    imxi  = (kkd**-2)*plaZim(xi)*Z*Te/Ti
    rexi  = (kkd**-2)*plaZre(xi)*Z*Te/Ti
    return rexi+1j*imxi

#%%########################################################################
#    advanced electron and ion susceptibilty [p50 Froula2011]

#chi_i= \omega_pi^2/k^2 \int_{-\infty}^{\infty} (k \cdot df/dv) / (w - kv -igamma)
#determines plasma response to electron denisty fluctuations
#inputs: 
#    kw_dict = dictionary of pre-computed scattering parameters
#    dists = distributin functions  ####################################### todo label how
@jax.jit
def chicust_i(kw_dict,dists,f_params):
    omgpi = kw_dict['omgpi']
    k     = kw_dict['k']
    vel   = kw_dict['vel']/c
    vemax = jnp.max(jnp.abs(vel))
    ##################### numerical
    Zpi = custZprime(vel, dists, f_params, vemax)   

    chiEre= (-(1.0/(c*k[:]/omgpi)**2))*(Zpi[0,:])  #me/mi
    chiEim= (-(1.0/(c*k[:]/omgpi)**2))*(-1j*Zpi[1,:])  #me/mi 
    return chiEre+chiEim

#chi e
@jax.jit
def chicust_e(kw_dict,dists,f_params):
    omgpe = kw_dict['omgpe']
    k     = kw_dict['k']
    vel   = kw_dict['vel']/c
    vemax = jnp.max(jnp.abs(vel))
    ##################### numerical
    Zpe = custZprime(vel, dists, f_params, vemax)  
    
    chiEre= (-(1.0/(c*k[:]/omgpe)**2))*(Zpe[0,:]) 
    chiEim= (-(1.0/(c*k[:]/omgpe)**2))*(-1j*Zpe[1,:]) 
    return chiEre+chiEim