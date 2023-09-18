
import jax
import jax.numpy as jnp

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