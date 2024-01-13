"""
"""
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import start_over


def test_something():
    ran_key = jran.PRNGKey(0)

    p50 = 0.25
    args = DEFAULT_MAH_PARAMS, p50, ran_key
    _res = start_over.mc_diffstar_u_params_singlegal(*args)
    u_params_q, u_params_ms, frac_quench, u_params = _res
    assert u_params.shape == (8,)
    assert u_params_ms.shape == (8,)
    assert u_params_q.shape == (8,)
    assert frac_quench.shape == ()
