import timeit
import time



import jax.numpy as jnp
from jax import random
from jax import jit


# simple activation function for jit benchmarking
def selu(x, alpha=1.67, lmbda=1.05):
    # convert to jnp array 
    alpha = jnp.array(alpha)
    # var named s.t won't be confused for anon function
    lmbda = jnp.array(lmbda)
    # activation function
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)



if __name__ == '__main__':  
    
    # generate large array of random numbers
    key = random.key(1701)
    x = random.normal(key,(1_000_000,))
    # see if GPU
    print(x.devices())
    # profiling the selu function
    t1 = timeit.timeit(
            lambda: selu(x,alpha=1.67,lmbda=1.05).block_until_ready(), # pass the function as an anon function
            timer=time.perf_counter,
            number=1
            )

    # just in time compilation
    selu_jit = jit(selu)

    # compiles the function because it is called
    _ = selu_jit(x) 
    
    t2 = timeit.timeit(
            lambda: selu_jit(x,alpha=1.67,lmbda=1.05).block_until_ready(),
            timer=time.perf_counter,
            number=1
            )

    # display times and speed up
    print(f'time without JIT: {t1} secs\ntime with JIT: {t2} secs')
    
    # taking derivatives

    


