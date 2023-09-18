"""
Code collected from Zaghoul and Ali's Algorithm 916 (https://dl.acm.org/doi/pdf/10.1145/2049673.2049679) 
and ExoJax implementation (https://github.com/HajimeKawahara/exojax/blob/master/src/exojax/special/faddeeva.py)

Faddeeva (wofz= w of z) functions (real and imag parts)

Order of Algorithm 916 is defined by n_Algorithm916

"""

from jax import jit
from jax.scipy.special import erfc
import jax.numpy as jnp

n_Algorithm916 = 32

an = 0.5*jnp.arange(1,n_Algorithm916+1)
a2n2 = an**2

@jit
def voigt_profile(x,sigma,gamma):
    """
    Voigt line shape function

    Effectively rescaling of real part of Faddeeva
    """
    sqrt2sigma = sigma*jnp.sqrt(2.0)
    xprime = x/sqrt2sigma
    y = gamma/sqrt2sigma*jnp.ones_like(x)
    voigt = rewofz(xprime, y)/sqrt2sigma/jnp.sqrt(jnp.pi)
    return voigt

@jit
def dawsn(z):
    """
    Dawson's Integral - see https://mathworld.wolfram.com/DawsonsIntegral.html
    
    """
    return jnp.sqrt(jnp.pi)/2.0*imwofz(z,jnp.zeros_like(z))

@jit
def erfcx(x):
    """
    Scaled Complementary Error Function
    
    """
    return jnp.exp(x**2)*erfc(x)

@jit
def rewofz(x, y):
    """Real part of wofz (Faddeeva) function based on Algorithm 916.

    We apply a=0.5 for Algorithm 916.

    Args:
        x: x < ncut/2
        y:

    Returns:
         jnp.array: Real(wofz(x+iy))
    """
    xy = x * y
    exx = jnp.exp(-x * x)
    f = exx * (erfcx(y) * jnp.cos(2.0 * xy) + x * jnp.sin(xy) / jnp.pi * jnp.sinc(xy / jnp.pi))
    y2 = y * y
    Sigma23 = jnp.sum(
        (jnp.exp(-(an[:,None] + x[None,:])**2) + jnp.exp(-(an[:,None] - x[None,:])**2)) / (a2n2[:,None] + y2[None,:]),axis=0)
    Sigma1 = faddeeva_sigma1(exx, y2)
    f = f + y / jnp.pi * (-jnp.cos(2.0 * xy) * Sigma1 + 0.5 * Sigma23)
    return f


@jit
def imwofz(x, y):
    """Imaginary part of wofz (Faddeeva) function based on Algorithm 916.

    We apply a=0.5 for Algorithm 916.

    Args:
        x: x < ncut/2
        y:

    Returns:
         jnp.array: Imag(wofz(x+iy))
    """
    wxy = 2.0 * x * y
    exx = jnp.exp(-x * x)
    f = -exx * erfcx(y) * jnp.sin(wxy) + x / jnp.pi * exx * jnp.sinc(
        wxy / jnp.pi)
    y2 = y * y
    Sigma1 = faddeeva_sigma1(exx, y2)
    Sigma45 = jnp.sum(an[:,None] * (-jnp.exp(-(an[:,None] + x[None,:])**2) + jnp.exp(-(an[:,None] - x[None,:])**2)) /
                      (a2n2[:,None] + y2[None,:]),axis=0)
    f = f + 1.0 / jnp.pi * (y * jnp.sin(wxy) * Sigma1 + 0.5 * Sigma45)

    return f

def faddeeva_sigma1(exx, y2):
    """
    
    Summation 1 as defined in Algorithm 916

    Used in both imaginary and real parts of Faddeeva
    
    """
    summation = jnp.sum(jnp.exp(-a2n2[:,None])/(a2n2[:,None]+y2[None,:]),axis=0)
    Sigma1 = exx * summation
    return Sigma1