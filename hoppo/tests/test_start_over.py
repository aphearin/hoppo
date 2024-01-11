"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from jax import random as jran

from .. import start_over


def test_something():
    ran_key = jran.PRNGKey(0)
    t0 = 13.7
    nt = 100
    t_table = np.linspace(0.1, t0, nt)

    p50 = 0.25
    args = t_table, DEFAULT_MAH_PARAMS, p50, ran_key
    _res = start_over.draw_single_sfh_MIX_with_exsitu(*args)
    mean_ms, cov_ms, shifts_ms, mean_q, cov_q, shifts_q = _res
