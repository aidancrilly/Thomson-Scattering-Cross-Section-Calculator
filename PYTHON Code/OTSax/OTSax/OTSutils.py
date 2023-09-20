import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from .OTSconstants import *
from .tools.IAEA_Z_table import IaeaTable

def create_dist_func_dict(dist_func,Nparams):
    graddfdv = d_distfunc_dv(dist_func,Nparams)
    dists = {'f' : Partial(dist_func), 'df/dv' : Partial(jax.jit(jax.grad(dist_func))), 'df/dv_vmap' : Partial(graddfdv)}
    return dists

def d_distfunc_dv(dist_func,Nparams):
    in_axes_tuple = (0,) + (None,)*Nparams
    graddfdv = jax.vmap(jax.grad(dist_func),in_axes=in_axes_tuple,out_axes=0)
    return jax.jit(graddfdv)

def create_Max_dist_func_dict():
    dists = {'f' : Partial(Maxwellian), 'df/dv' : Partial(GradMaxwellian), 'df/dv_vmap' : Partial(GradMaxwellian)}
    return dists

def eval_multi_ion_dist(v,multi_ion_params,number_of_ion_species):

    dists = jnp.zeros((number_of_ion_species,len(v)))
    for i_ion in range(number_of_ion_species):
        ion_params = multi_ion_params[i_ion]
        fi = jax.lax.cond(ion_params['isMax'],
                        lambda x : Maxwellian(x,ion_params['vi'],ion_params['Ti'],ion_params['Ai']*amu_eV)/c,
                        lambda x : ion_params['dists']['f'](x,*ion_params['dists_params'])/c, v/c)
        dists = dists.at[i_ion,:].set(ion_params['frac']*fi)
    return dists

eval_multi_ion_dist = jax.jit(eval_multi_ion_dist,static_argnums=(2,))

def get_kw_vals(omg,omgL,sa,Z,Ai,ne):
    mi    = amu_eV*Ai
    omgpe = omgpe_const*jnp.sqrt(ne) #rad/s
    omgpi = omgpe*jnp.sqrt(Z*me/mi)
    sarad = sa*deg2rad
    kL    = jnp.sqrt(omgL**2-omgpe**2)/c
    omgs  = omg+omgL
    ks    = jnp.sqrt(omgs**2-omgpe**2)/c  
    k     = jnp.sqrt(kL**2+ks**2-2*ks*kL*jnp.cos(sarad))
    vel   = omg/k
    return {'vel' : vel, 'k' : k, 'ks' : ks, 'kL' : kL, 'omgpe' : omgpe, 'omgpi' : omgpi}

def get_laser_and_electron_kw_vals(laser_params,ne):
    omg,omgL,sa = laser_params['omega'],laser_params['laser omega'],laser_params['scattering angle']
    omgpe = omgpe_const*jnp.sqrt(ne) #rad/s
    sarad = sa*deg2rad
    kL    = jnp.sqrt(omgL**2-omgpe**2)/c
    omgs  = omg+omgL
    ks    = jnp.sqrt(omgs**2-omgpe**2)/c  
    k     = jnp.sqrt(kL**2+ks**2-2*ks*kL*jnp.cos(sarad))
    vel   = omg/k
    return {'vel' : vel, 'k' : k, 'ks' : ks, 'kL' : kL, 'omgpe' : omgpe}

def add_ion_kw_vals(kw_dict,Z,Ai):
    mi    = amu_eV*Ai
    omgpi = kw_dict['omgpe']*jnp.sqrt(Z*me/mi)
    kw_dict['omgpi'] = omgpi
    return kw_dict

@jax.jit
def Maxwellian(v,vd,T,m):
    sig2 = T/m
    norm = jnp.sqrt(2*jnp.pi*sig2)
    return jnp.exp(-0.5*(v-vd)**2/sig2)/norm

@jax.jit
def GradMaxwellian(v,vd,T,m):
    sig2 = T/m
    norm = jnp.sqrt(2*jnp.pi*sig2)
    return -(v-vd)/sig2*jnp.exp(-0.5*(v-vd)**2/sig2)/norm