import jax.numpy as jnp
import jax
from .OTSconstants import *
from .OTSplasma import *
from .OTSutils import *

def calc_S_kw(laser_params,electron_params,multi_ion_params,number_of_ion_species):
    """
    
    laser_params
    electron_params
    multi_ion_params
    number_of_ion_species
    
    """
    kw_dict = get_laser_and_electron_kw_vals(laser_params,electron_params['ne'])
    k = kw_dict['k']
    v = kw_dict['vel']

    chie,fe = jax.lax.cond(electron_params['isMax'],
                           calc_MaxElectron_chi_and_fe,
                           calc_NonMaxElectron_chi_and_S_kw,
                           kw_dict,electron_params)

    def single_ion_susceptibility(ion_params):
        ion_kw_dict = add_ion_kw_vals(kw_dict,ion_params['Z'],ion_params['Ai'])
        chii = jax.lax.cond(ion_params['isMax'],
                        lambda ion_kw_dict : chith_i(ion_kw_dict,electron_params['Te'],ion_params['Ti'],ion_params['Z'],ion_params['Ai'],ion_params['vi']),
                        lambda ion_kw_dict : chicust_i(ion_kw_dict,ion_params['dists']),ion_kw_dict)
        return ion_params['frac']*chii
    
    def single_ion_S_kw(ion_params):
        fi = jax.lax.cond(ion_params['isMax'],
                        lambda x : Maxwellian(x,ion_params['vi'],ion_params['Ti'],ion_params['Ai']*amu_eV)/c,
                        lambda x : ion_params['dists']['f'](x,*ion_params['dists']['f_params'])/c, v/c)
        
        skwi=(2*jnp.pi*ion_params['Z']/kw_dict['k'])*dispi*ion_params['frac']*fi
        return skwi

    total_chii = jnp.zeros_like(k)
    for i_ion in range(number_of_ion_species):
        total_chii += single_ion_susceptibility(multi_ion_params[i_ion])

    eps=chie+total_chii+1.0
    dispe=jnp.abs((1+total_chii)/eps)**2
    dispi=jnp.abs((chie)/eps)**2

    skwe=(2*jnp.pi/k)*dispe*fe
    
    total_skwi = jnp.zeros_like(k)
    for i_ion in range(number_of_ion_species):
        total_skwi += single_ion_S_kw(multi_ion_params[i_ion])

    return (skwe+total_skwi)

@jax.jit
def calc_MaxElectron_chi_and_fe(kw_dict,electron_params):
    v = kw_dict['k']

    chie = chith_e(kw_dict,electron_params['ve'],electron_params['Te'])
    fe = Maxwellian(v/c,electron_params['ve'],electron_params['Te'],me)/c

    return chie,fe

@jax.jit
def calc_NonMaxElectron_chi_and_S_kw(kw_dict,electron_params):
    v = kw_dict['k']

    chie = chicust_e(kw_dict,electron_params['dists'])
    fe = electron_params['dists']['f'](v/c,*electron_params['dists']['f_params'])/c

    return chie,fe

calc_S_kw = jax.jit(calc_S_kw,static_argnums=(3,))