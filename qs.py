import jax.numpy as jnp
from jax import random
from jax import jit


# simple activation function
def selu(x, alpha=1.67, lmbda=1.05):
    alpha = jnp.array(alpha)
    lmbda = jnp.array(lmbda)
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(5.0)
print(selu(x))

