import jax
import jax.numpy as jnp
from .adaptive_quad import quad

def KellerWrobel_PV(tau, dfdv, f_params, v_scale):
    """
    
    Calculates Cauchy princpal value of:

    \int_{-1}^{+1} f(x)/(x-tau) dx

    See https://www.sciencedirect.com/science/article/pii/S0377042715004422?via%3Dihub
    
    """

    def f(x,v_scale,*_parameters):
        v = x*v_scale
        return dfdv(v,*_parameters)

    def g_integrand(x,tau,v_scale,*_parameters):
        fx = f(x,v_scale,*_parameters)
        ftau = f(tau,v_scale,*_parameters)
        g = (fx-ftau)/(x-tau)
        return jnp.reshape(g, ())

    def h_integrand(x,tau,v_scale,*_parameters):
        ftaup = f(tau+x,v_scale,*_parameters)
        ftaum = f(tau-x,v_scale,*_parameters)
        h = (ftaup-ftaum)/x
        return jnp.reshape(h, ())

    delta = jnp.where(tau < 0.0, 1.0+tau, 1.0-tau)
    I_tau = f(tau,v_scale,*f_params)*jnp.log((1.0-tau)/(1.0+tau))

    integrand_params = [tau,v_scale,*f_params]

    I_gp = quad(g_integrand,delta+tau,1.0,integrand_params)
    I_gm = quad(g_integrand,-1.0,tau-delta,integrand_params)
    I_g = I_gp+I_gm

    I_h = quad(h_integrand,0.0,delta,integrand_params)
    return I_tau+I_g+I_h