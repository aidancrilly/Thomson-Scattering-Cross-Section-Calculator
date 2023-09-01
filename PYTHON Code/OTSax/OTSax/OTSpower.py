import jax.numpy as jnp
import jax
from .OTSconstants import *
from .OTSplasma import *
from .OTSutils import *

#s(k,omega)
#form factor for waves in thermal plasma
#inputs:
#    omg = wave frequency [rad/s]
#    sa = scattering angle [deg]
#    omgL = probe laser frequency [rad/s]
#    Te = electron temperature [keV]
#    Ti = ion temperature [keV]
#    Z = ion charge = q/e
#    Ai=ion protons + nuetrons 
#    ne = electron density [cm^-3]
#    ve=electron velocity [c]
#    vi=ion velocity [c]
@jax.jit
def skw(omg,sa,omgL,Te,Ti,Z,Ai,ne,ve,vi):
    kw_dict = get_kw_vals(omg,omgL,sa,Z,Ai,ne)
    k = kw_dict['k']
    v = kw_dict['vel']
    
    chie  = chith_e(kw_dict,ve,Te)
    chii  = chith_i(kw_dict,Te,Ti,Z,Ai,vi)
    eps   = chie+chii+1.0
    dispe = jnp.abs((1+chii)/eps)**2
    dispi = jnp.abs((chie)/eps)**2

    vte=jnp.sqrt(Te/me)  #c
    vti=jnp.sqrt(Ti/(mproton*Ai)) #c
    fe=jnp.sqrt(1.0/(2*jnp.pi*vte**2*c**2))*jnp.exp(-0.5*(v-ve*c)**2/(vte**2*c**2))
    fi=jnp.sqrt(1.0/(2*jnp.pi*vti**2*c**2))*jnp.exp(-0.5*(v-vi*c)**2/(vti**2*c**2))

    skwe=(2*jnp.pi/k)*dispe*fe
    skwi=(2*jnp.pi*Z/k)*dispi*fi
    return (skwe+skwi)

#s(k,omega)
#form factor for waves with custom ion distribution
#inputs:
#    omg = wave frequency [rad/s]
#    sa = scattering angle [deg]
#    omgL = probe laser frequency [rad/s]
#    Z = ion charge = q/e
#    Ai=ion protons + nuetrons 
#    ne = electron density [cm^-3]
#    ve=electron velocity [c]
#    dist [1/c] distribution function of ions, at phase velocity omg/k
@jax.jit
def skwic(omg,sa,omgL,Te,Z,Ai,ne,ve,dists):
    kw_dict = get_kw_vals(omg,omgL,sa,Z,Ai,ne)
    k = kw_dict['k']
    v = kw_dict['vel']

    vte=jnp.sqrt(Te/me)  #c
    fe=jnp.sqrt(1.0/(2*jnp.pi*vte**2*c**2))*jnp.exp(-0.5*(v-ve*c)**2/(vte**2*c**2))
    
    chie = chith_e(kw_dict,ve,Te)
    chii = chicust_i(kw_dict,dists)
    
    eps=chie+chii+1.0
    dispe=jnp.abs((1+chii)/eps)**2
    dispi=jnp.abs((chie)/eps)**2
    
    fi=1.0*dists['fi'](v/c,*dists['fi_params'])/c
    
    skwe=(2*jnp.pi/k)*dispe*fe
    skwi=(2*jnp.pi*Z/k)*dispi*fi
    return (skwe+skwi)