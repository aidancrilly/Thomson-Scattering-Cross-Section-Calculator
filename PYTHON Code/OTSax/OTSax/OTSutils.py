import jax
import jax.numpy as jnp
from .OTSconstants import *
from jax.tree_util import Partial

def d_distfunc_dv(dist_func,Nparams):
    in_axes_tuple = (0,) + (None,)*Nparams
    graddfdv = jax.vmap(jax.grad(dist_func),in_axes=in_axes_tuple,out_axes=0)
    # return jax.jit(lambda v,*_params : graddfdv(jnp.atleast_1d(v),*_params))
    return jax.jit(graddfdv)

def create_dist_func_dict(dist_func,*_params):
    Nparams = len(_params)
    graddfdv = d_distfunc_dv(dist_func,Nparams)
    dists = {'fi' : Partial(dist_func), 'dfi/dv' : Partial(jax.jit(jax.grad(dist_func))), 'dfi/dv_vmap' : Partial(graddfdv), 'fi_params' : [*_params]}
    return dists

def get_kw_vals(omg,omgL,sa,Z,Ai,ne):
    mi    = mproton*Ai
    omgpe = omgpe_const*jnp.sqrt(ne) #rad/s
    omgpi = omgpe*jnp.sqrt(Z*me/mi)
    sarad = sa*deg2rad
    kL    = jnp.sqrt(omgL**2-omgpe**2)/c
    omgs  = omg+omgL
    ks    = jnp.sqrt(omgs**2-omgpe**2)/c  
    k     = jnp.sqrt(kL**2+ks**2-2*ks*kL*jnp.cos(sarad))
    vel   = omg/k
    return {'vel' : vel, 'k' : k, 'ks' : ks, 'kL' : kL, 'omgpe' : omgpe, 'omgpi' : omgpi}

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:20:41 2022

@author: Colin
"""

##### flycheck lookup tables, from Jack Halladay

IEAA_DATA_PATH = r'C:\\Users\\Aidan Crilly\\Documents\\GitHub\\OTSax\\data\\iontable\\'

class IaeaTable:
    ''' Class to generate a function which relates average ionisation to plasma 
    density and temperature. Function interpolates over lookup tables of 
    average ionisation which were produced using the nLTE code FLYCHK. They 
    were downloaded from the database hosted at [1]. 
    
    The database includes the following caveat about the validity of tabulated 
    data:
        FLYCHK calculations are known to give better results for highly ionized
        plasmas and intermediate electron densities.
    
    Cite as:
        FLYCHK: Generalized population kinetics and spectral model for rapid 
        spectroscopic analysis for all elements, H.-K. Chung, M.H. Chen, 
        W.L. Morgan, Y. Ralchenko and R.W. Lee, 
        High Energy Density Physics, Volume 1, Issue 1, December 2005
    
    Data exists for all elements with atomic numbers in the range 1 (Hydrogen) 
    through to 79 (Gold). 
    
    ************************************************
    |  Access the generated function by calling    |
    |      self.model(Te, ne)                      |
    |  where:                                      |
    |      Te = electron temperature [eV]          |
    |      ne = electron density [cm^-3]           |
    ************************************************
        
    [1] - https://www-amdis.iaea.org/FLYCHK/ZBAR/csd014.php
    '''
    extension='.zvd'
    def __init__(self, AtomicNumber):
        ''' Description of arguments:
        1) AtomicNumber - atomic number of desired element (in the range [1,79]
            inclusive).
        '''
        dpath = IEAA_DATA_PATH + str(AtomicNumber) + self.extension
        ne = []
        Z = []
        Te = []
        with open(dpath, 'r') as f:
            lines = list(f)
            i = 0
            while(i<637):
                i += 1
                line = lines[i]
                ne.append( float(line[10:17]) ) 
                i += 11
                j=0
                Z_row = []
                while(j<36):
                    line = lines[i+j]
                    s=line.strip()
                    TT, ZZ = s.split()
                    Te.append( float(TT) )
                    Z_row.append( float(ZZ) )
                    j += 1
                Z.append(jnp.array(Z_row))
                i += 37
        self.Z = jnp.array(Z)
        self.ne = jnp.array(ne)
        self.Te = jnp.array(Te[0:36])

        self.N_ne = self.ne.shape[0]
        self.N_Te = self.Te.shape[0]

    # @jax.jit
    def interp(self,ne,Te):
        Te = jnp.where(Te < self.Te[0] , self.Te[0], Te)
        Te = jnp.where(Te > self.Te[-1] , self.Te[-1], Te)

        ne = jnp.where(ne < self.ne[0] , self.ne[0], ne)
        ne = jnp.where(ne > self.ne[-1] , self.ne[-1], ne)

        ix = jnp.clip(jnp.searchsorted(self.Te, Te, side="right"), 1, self.N_Te - 1)
        iy = jnp.clip(jnp.searchsorted(self.ne, ne, side="right"), 1, self.N_ne - 1)

        # Using Wikipedia's notation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
        z_11 = self.Z[ix - 1, iy - 1]
        z_21 = self.Z[ix, iy - 1]
        z_12 = self.Z[ix - 1, iy]
        z_22 = self.Z[ix, iy]

        z_xy1 = (self.Te[ix] - Te) / (self.Te[ix] - self.Te[ix - 1]) * z_11 + (Te - self.Te[ix - 1]) / (
            self.Te[ix] - self.Te[ix - 1]
        ) * z_21
        z_xy2 = (self.Te[ix] - Te) / (self.Te[ix] - self.Te[ix - 1]) * z_12 + (Te - self.Te[ix - 1]) / (
            self.Te[ix] - self.Te[ix - 1]
        ) * z_22

        z = (self.ne[iy] - ne) / (self.ne[iy] - self.ne[iy - 1]) * z_xy1 + (ne - self.ne[iy - 1]) / (
            self.ne[iy] - self.ne[iy - 1]
        ) * z_xy2

        return z