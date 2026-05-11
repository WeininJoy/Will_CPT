import jax
import jax.numpy as jnp
import numpy as np


def np_func(x): 
    return np.sin(np.sin(x)) * np.cos(x)

@jax.jit
def jax_func(x):
    return jnp.sin(jnp.sin(x)) * jnp.cos(x)

x_np = np.ones((1000,1000))
x_jax = jnp.ones((1000,1000))

import time
start = time.time()
np_func(x_np)
print("NumPy function took %s seconds" % (time.time() - start))
jax_func(x_jax)
start = time.time()
jax_func(x_jax)
print("JAX function took %s seconds" % (time.time() - start))

params = {
    'layer1': {
        'weights': jnp.ones((1000,1000)),
        'biases': jnp.ones((1000,))
    },
    'layer2': {
        'weights': jnp.ones((1000,1000)),
        'biases': jnp.ones((1000,))
    }
}
jax.tree.map(lambda x: x * 2, params)