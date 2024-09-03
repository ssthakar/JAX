from enum import global_flag_repr
import jax
import jax.numpy as jnp
from jax import jit
import timeit
import time

global_list =[]

def log2(x):
    """
    * function is impure because while being deterministic, it modifies external state
    """
    # impure function due to list append
    global_list.append(x)
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2)
    return ln_x/ln_2

def log2_list(x):
    """
        * Does not modify external state, but is not deterministic as the lenght of 
          global list can vary.
    """
    # impure function due to list append
    list_len = global_list.__len__()
    # another example
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2)
    ln_2_by_len = (ln_x/ln_2)/list_len
    return ln_2_by_len
 

# transform python function to set of primitive operations

def log2_pure(x):
    """
        * example of pure function,
        * does not modify any external data, no side effects like pring statements
        * deterministic, for a value of x, function will return the same value every time
        * JAX documentation tells to use these types of functions for JIT
    """
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2)
    return ln_x/ln_2

# print(jax.make_jaxpr(log2)(3.0))


# some more funcs and their jaxprs
def func2(inner,first,second):
    temp = first + inner(second)*3
    return jnp.sum(temp)

def inner(second):
    if second.shape[0] > 4:
        return jnp.sin(second)
    else:
        assert False

def func3(first,second):
    return func2(inner,first,second)

print(f'jaxpr for func2\n')

func3_jit = jit(func3)
_ = func3_jit(jnp.zeros(8),jnp.ones(8))

t2 = timeit.timeit(
            lambda: func3_jit(jnp.ones(8),jnp.ones(8)),
            timer=time.perf_counter,
            number=1
            )

print(t2)


# this function call was giving errors
# interpreter complains that function does not have a dtpye attr
# function args should be able to be converted to dtypes
# print(jax.make_jaxpr(func2)(inner,jnp.zeros(8),jnp.ones(8)))



print(f'jaxpr for func3\n')
# in this case, function does not have any ambigous vars, dtypes can be 
# identified ??
print(jax.make_jaxpr(func3)(jnp.zeros(8),jnp.ones(8)))

# in case of func3, the jaxpr does not follow the order of a typical python code,
# call to inner func, and if conditional are inlined

# understanding pytrees






























