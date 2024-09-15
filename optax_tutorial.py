import jax
import jax.numpy as jnp
import optax
import functools # Higher order functions, can act on or return functions
import matplotlib.pyplot
# Fitting a linear model


"""
    * Things to consider when coding this problem
        
        * Don't use symbolic derivatives, defeats the purpose of JAX
        * Pure functions as much as possible
        * Try to use JAX function transformations
"""

@functools.partial(jax.vmap,in_axes=(None,0))
def network(params,x):
    return jnp.dot(params,x)

