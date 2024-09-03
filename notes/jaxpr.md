# Understanding JIT and jaxprs

* jaxpr is JAX's internal representation, statically typed slightly low level lang
* transforms normal python functions into this low level lang
* jaxprs work only on pure python functions
* Results depend only on the input variables
* No Free variables are captured from enclosing scopes
* No side effects allowed, can cause tracer leaks

### Why not use impure functions and what are tracers?
* under transformations, can fail silently without throwing std errors.
* JAX wraps all args to functions with tracer objs
    *   tracers record all operations performed on them during the function exec
    *   JAX then uses the tracer record to reconstruct the entire function (comp graph?)
    *   adding side effects and free vars, or modifying external scopes can lead to disrupting this record
    *   tracer leaks

### Pytrees
* no tuples in jaxpr lang
* jax flattens this into lists of inputs and outputs, called pytrees
