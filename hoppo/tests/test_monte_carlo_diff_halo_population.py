"""
"""
import numpy as np
from diffmah.monte_carlo_halo_population import mc_halo_population
from jax import random as jran

from .. import monte_carlo_diff_halo_population as mcpop


def test_mcpop_evaluates():
    ran_key = jran.PRNGKey(0)
    n_times = 100
    t0 = 13.8
    t_table = np.linspace(0.1, t0, n_times)

    n_mah_halos = 200
    logm0 = 12.0
    logm0_arr = logm0 + np.zeros(n_mah_halos)

    ran_key, mah_key = jran.split(ran_key, 2)
    _mah_res = mc_halo_population(t_table, t0, logm0_arr, mah_key)
    dmhdt, log_mah, early, late, lgtc, mah_type = _mah_res
    mah_params = np.array((logm0_arr, lgtc, early, late)).T

    n_histories = 30

    ran_key, p50_key = jran.split(ran_key, 2)
    p50 = jran.uniform(p50_key, minval=0, maxval=1, shape=(n_mah_halos,))

    args = (t_table, logm0, mah_params, p50, n_histories, ran_key)
    _res = mcpop.draw_sfh_MIX(*args)
    for _x in _res:
        assert np.all(np.isfinite(_x))
    mstar, sfr, p50_sampled, weight = _res
    assert np.all(mstar >= 0)
    assert np.any(mstar > 0)
    assert np.all(mstar < 1e13)
    assert np.all(sfr >= 0)
