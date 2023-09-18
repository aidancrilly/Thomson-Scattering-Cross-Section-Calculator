import scipy.constants as sc
import jax.numpy as jnp

c       = sc.c #v light in m/s
me      = sc.value('electron mass energy equivalent in MeV')*1e6
deg2rad = 2*sc.pi/360.0
sqrt_pi = jnp.sqrt(jnp.pi)
omgpe_const = 56400
amu_eV  = sc.value('atomic mass constant energy equivalent in MeV')*1e6