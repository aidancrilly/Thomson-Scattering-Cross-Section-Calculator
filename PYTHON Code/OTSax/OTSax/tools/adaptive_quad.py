import functools as ft
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

relative_tolerance = 1e-4
absolute_tolerance = 0.0
divmax = 32

#########################################################################
# Following Patrick Kidger in https://github.com/google/jax/issues/9014 #
#########################################################################

@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def quad(func, a, b, parameters):
    """Calculates the integral

    \int_a^b func(t, *parameters) dt
    """

    result = GLQ_adaptive_integral(func, a, b, parameters)
    
    return result

def quad_fwd(func, a, b, parameters):
    result = quad(func, a, b, parameters)
    aux = (a, b, parameters)
    return result, aux

def quad_bwd(func, aux, grad):
    a, b, parameters = aux

    grad_a = -grad * func(a, *parameters)
    grad_b = grad * func(b, *parameters)

    grad_args = []
    for i in range(1,len(parameters)+1):
        def _vjp_func(_t, *_parameters):
            return jax.grad(func, argnums=i)(_t, *_parameters)
        grad_args.append(grad * quad(_vjp_func, a, b, parameters))

    return grad_a, grad_b, grad_args

quad.defvjp(quad_fwd, quad_bwd)
    
###################################################################################################################################
# Gauss-Legendre Quadrature                                                                                                       #
# The original code was written by Koelbig (CERN) and modified by J. Vorberger (presently at HZDR, Germany) and D. Chapman (AWE). #
# A. Crilly (Imperial) converted into Python/JAX                                                                                  #
###################################################################################################################################

const_GLQ_points = jnp.array([0.9602898564975362316835608686e0,0.7966664774136267395915539365e0,0.5255324099163289858177390492e0,0.1834346424956498049394761424e0,\
    0.9894009349916499325961541735e0,0.9445750230732325760779884155e0,0.8656312023878317438804678977e0,0.7554044083550030338951011948e0,0.6178762444026437484466717640e0,0.4580167776572273863424194430e0,\
    0.2816035507792589132304605015e0,0.09501250983763744018531933542e0])
const_GLQ_weights = jnp.array([0.10122853629037625915253135431e0,0.2223810344533744705443559944e0,0.3137066458778872873379622020e0,0.3626837833783619829651504493e0,\
    0.02715245941175409485178057246e0,0.06225352393864789286284383699e0,0.09515851168249278480992510760e0,0.1246289712555338720524762822e0,0.1495959888165767320815017305e0,0.1691565193950025381893120790e0,\
    0.1826034150449235888667636680e0, 0.1894506104550684962853967232e0])

def _GLQ_bound_cond(input_dict):
    new_upper_bound = input_dict['new_upper_bound']
    upper_bound     = input_dict['upper_bound']
    return new_upper_bound != upper_bound

def _GLQ_control_loop(input_dict):
    # Reset the new bounds of the current sub-interval.
    input_dict['new_lower_bound'] = 1.0*input_dict['new_upper_bound']
    input_dict['new_upper_bound'] = 1.0*input_dict['upper_bound']
    # Reset values to avoid triggering convergence criteria
    input_dict['integral_1'] = 1e30
    input_dict['integral_2'] = 0.0
    input_dict['trans_const_2'] = 100.0*input_dict['min_bin_width']/input_dict['relative_tolerance']

    # Calculate the GL quadaratures until convergence is reached.
    input_dict = jax.lax.while_loop(_GLQ_convergence_cond,_GLQ_main_loop,input_dict)

    # Add the result of the higher-order GL quadrature to the running total.
    input_dict['integral'] += input_dict['integral_2']

    return input_dict

def _GLQ_convergence_cond(input_dict):
    integral_1 = input_dict['integral_1']
    integral_2 = input_dict['integral_2']
    rtol = input_dict['relative_tolerance']
    atol = input_dict['absolute_tolerance']
    min_bin_width = input_dict['min_bin_width']
    trans_const_2 = input_dict['trans_const_2']

    # Test to see if the integral has converged. This is determined by requiring that the difference between the to estimates of
    # the integral is smaller than their mean value multiplied by the tolerance set by the user.
    rtol_target = rtol * 0.5e0 * jnp.abs(integral_1 + integral_2)
    cond1_val = jnp.where(jnp.abs(integral_2 - integral_1) <= jnp.max(jnp.array([atol,rtol_target])),True,False)

    # If the current sub-interval is smaller than minimum interval width then exit the loop.
    cond2_val = jnp.where(2.0e0 * trans_const_2 < min_bin_width,True,False)

    cond_val = cond1_val | cond2_val

    return jnp.logical_not(cond_val)

def _GLQ_main_loop(input_dict):
    # Transformation constants required to scale the integration onto the interval [-1,1].
    input_dict['trans_const_1'] = 0.5 * (input_dict['new_upper_bound'] + input_dict['new_lower_bound'])
    input_dict['trans_const_2'] = 0.5 * (input_dict['new_upper_bound'] - input_dict['new_lower_bound'])

    # Peform the low-resolution (8-point) GL quadrature in the current sub-interval.
    input_dict['integral_1'] = 0.0
    input_dict = jax.lax.fori_loop(0,4,_GLQ_8point_loop,input_dict)
    input_dict['integral_1'] = input_dict['trans_const_2'] * input_dict['integral_1']

    # Peform the high-resolution (16-point) GL quadrature in the current sub-interval.
    input_dict['integral_2'] = 0.0
    input_dict = jax.lax.fori_loop(4,12,_GLQ_16point_loop,input_dict)
    input_dict['integral_2'] = input_dict['trans_const_2'] * input_dict['integral_2']

    # Set the new upper bound to the mid point of the current sub-interval.
    input_dict['new_upper_bound'] = jnp.where(jnp.abs(input_dict['integral_2'] - input_dict['integral_1']) <= input_dict['relative_tolerance'] * 0.5e0 * jnp.abs(input_dict['integral_1'] + input_dict['integral_2']),
                                              input_dict['new_upper_bound'],
                                              input_dict['trans_const_1'])

    return input_dict

def _GLQ_8point_loop(i,input_dict):
    f = input_dict['integrand_function']
    trans_const_1 = input_dict['trans_const_1']
    trans_const_2 = input_dict['trans_const_2']
    parameters = input_dict['parameters']

    input_dict['integral_1'] += const_GLQ_weights[i] * (f(trans_const_1 + trans_const_2 * const_GLQ_points[i], *parameters) \
                                                      + f(trans_const_1 - trans_const_2 * const_GLQ_points[i], *parameters))
    return input_dict

def _GLQ_16point_loop(i,input_dict):
    f = input_dict['integrand_function']
    trans_const_1 = input_dict['trans_const_1']
    trans_const_2 = input_dict['trans_const_2']
    parameters = input_dict['parameters']

    input_dict['integral_2'] += const_GLQ_weights[i] * (f(trans_const_1 + trans_const_2 * const_GLQ_points[i], *parameters) + \
                                                        f(trans_const_1 - trans_const_2 * const_GLQ_points[i], *parameters))
    return input_dict

def GLQ_adaptive_integral(integrand_function, lower_bound, upper_bound, parameters):

    # Create PyTree of required variables
    min_bin_width = (upper_bound - lower_bound)/divmax
    # For integral function to be passed in PyTree in needs to be a wrapped partial function
    partial_f = Partial(integrand_function)

    input_dict = {'integral' : 0.0, 'lower_bound' : lower_bound, 'upper_bound' : upper_bound,
             'integrand_function' : partial_f, 'parameters' : parameters, 
             'relative_tolerance' : relative_tolerance, 'absolute_tolerance' : absolute_tolerance, 
             'new_lower_bound' : 0.0, 'new_upper_bound' : 1.0*lower_bound, 'min_bin_width': min_bin_width,
             'trans_const_1' : 0.0, 'trans_const_2' : 0.0, 'integral_1' : 0.0, 'integral_2' : 0.0}

	# The main loop recursively splits the interval performing Gauss-Legendre quadrature using 4 and 8 points. The results are
	# compared and if convergence is not achieved the new upper bound is set to the current mid point (i.e. the interval uniformly
	# bisected) and the integral is recomputed in both new intervals. The loop runs until the last sub-interval considered reaches
	# the original upper integration limit.
    input_dict = jax.lax.cond(lower_bound == upper_bound,lambda x : x, lambda x : jax.lax.while_loop(_GLQ_bound_cond,_GLQ_control_loop,x),input_dict)

    return input_dict['integral']


