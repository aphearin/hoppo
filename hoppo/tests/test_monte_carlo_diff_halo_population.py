"""
"""
import numpy as np
from jax import random as jran

from .. import monte_carlo_diff_halo_population as mcpop


def test_mcpop_evaluates():
    ran_key = jran.PRNGKey(0)
    n_times = 100
    t_table = np.linspace(0.1, 13.8, n_times)

    n_mah_halos = 20
    n_mah_parmas = 4
    mah_params = np.zeros((n_mah_halos, n_mah_parmas))

    logmh = 12.0
    n_histories = 30

    ran_key, p50_key = jran.split(ran_key, 2)
    p50 = jran.uniform(p50_key, minval=0, maxval=1, shape=(n_mah_halos,))

    args = (t_table, logmh, mah_params, p50, n_histories, ran_key)
