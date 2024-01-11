"""
"""
from collections import OrderedDict
from functools import partial

import numpy as np
from diffstar.defaults import FB, LGT0
from diffstar.kernels.kernel_builders import get_ms_sfh_from_mah_kern
from diffstar.kernels.main_sequence_kernels import (
    DEFAULT_MS_PDICT as DEFAULT_SFR_PARAMS_DICT,
)
from diffstar.kernels.main_sequence_kernels import _get_unbounded_sfr_params
from diffstar.kernels.quenching_kernels import DEFAULT_Q_PDICT as DEFAULT_Q_PARAMS_DICT
from diffstar.kernels.quenching_kernels import _get_unbounded_q_params
from diffstar.sfh import sfh_galpop
from dsps.utils import cumulative_mstar_formed
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS, get_smah_means_and_covs_mainseq
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_MAINSEQ_PARAMS,
    DEFAULT_R_QUENCH_PARAMS,
    _get_slopes_mainseq,
    _get_slopes_quench,
)
from .pdf_quenched import (
    DEFAULT_SFH_PDF_QUENCH_PARAMS,
    frac_quench_vs_lgm0,
    get_smah_means_and_covs_quench,
)

DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_SFR_PARAMS_DICT.values())
)
DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_SFR_PARAMS_DICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)
DEFAULT_UNBOUND_Q_PARAMS = np.array(
    _get_unbounded_q_params(*tuple(DEFAULT_Q_PARAMS_DICT.values()))
)
UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]


ms_sfh_galpop_u_params = get_ms_sfh_from_mah_kern(galpop_loop="vmap", tobs_loop="scan")
cumulative_mstar_formed_vmap = jjit(vmap(cumulative_mstar_formed, in_axes=(None, 0)))


@partial(jjit, static_argnames=["n_histories"])
def draw_sfh_MIX(
    t_table,
    logmh,
    mah_params,
    p50,
    n_histories,
    ran_key,
    pdf_parameters_Q=np.array(list(DEFAULT_SFH_PDF_QUENCH_PARAMS.values())),
    pdf_parameters_MS=np.array(list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.values())),
    R_model_params_Q=np.array(list(DEFAULT_R_QUENCH_PARAMS.values())),
    R_model_params_MS=np.array(list(DEFAULT_R_MAINSEQ_PARAMS.values())),
    lgt0=LGT0,
    fb=FB,
):
    """
    Generate Monte Carlo realization of the star formation histories of
    a mixed population of quenched and main sequence galaxies
    for a single halo mass bin.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    n_histories : int
        Number of SFH histories to generate by DiffstarPop.
    ran_key : ndarray of shape 2
        JAX random key.
    index_select: ndarray of shape (n_times_fstar, )
        Snapshot indices used in fstar computation
    index_high: ndarray of shape (n_times_fstar, )
        Indices of np.searchsorted(t, t - fstar_tdelay)[index_select]
    fstar_tdelay: float
        Time interval in Gyr for fstar definition.
        fstar = (mstar(t) - mstar(t-fstar_tdelay)) / fstar_tdelay[Gyr]
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    mstar : ndarray of shape (n_histories x 2, n_times)
        Stores cumulative stellar mass history in units of Msun/yr

    sfr : ndarray of shape (n_histories x 2, n_times)
        Stores star formation rate history in units of Msun/yr

    p50 : ndarray of shape (n_histories x 2, n_times)
        Stores star formation rate history in units of Msun/yr

    weights : ndarray of shape (n_histories x 2, )
        Stores star formation rate history in units of Msun/yr

    """
    logmh = jnp.atleast_1d(logmh)

    (choice_key, quench_key, mainseq_key, fquench_key, ran_key) = jran.split(ran_key, 5)
    n_mah = len(mah_params)

    sampled_mahs_inds = jran.choice(
        choice_key, n_mah, shape=(n_histories,), replace=True
    )
    mah_params_sampled = mah_params[sampled_mahs_inds]
    p50_sampled = p50[sampled_mahs_inds]

    _res = get_smah_means_and_covs_mainseq(logmh, *pdf_parameters_MS)
    means_mainseq, covs_mainseq = _res
    means_mainseq = means_mainseq[0]
    covs_mainseq = covs_mainseq[0]

    R_vals_mainseq = _get_slopes_mainseq(logmh, *R_model_params_MS)
    R_vals_mainseq = jnp.array(R_vals_mainseq)[:, 0]
    shifts_mainseq = jnp.einsum("p,h->hp", R_vals_mainseq, (p50_sampled - 0.5))

    _res = get_smah_means_and_covs_quench(logmh, *pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    _res = _get_slopes_quench(logmh, *R_model_params_Q)
    R_Fquench, R_vals_quench = _res[0], _res[1:]
    R_vals_quench = jnp.array(R_vals_quench)[:, 0]
    shifts_quench = jnp.einsum("p,h->hp", R_vals_quench, (p50_sampled - 0.5))
    shifts_Fquench = R_Fquench * (p50_sampled - 0.5)
    fquench_x0 = pdf_parameters_Q[0] + shifts_Fquench
    frac_quench = frac_quench_vs_lgm0(logmh, fquench_x0, *pdf_parameters_Q[1:4])

    sfh_params_Q = jran.multivariate_normal(
        quench_key, means_quench, covs_quench, shape=(n_histories,)
    )
    sfh_params_Q = sfh_params_Q + shifts_quench

    sfr_params_Q = sfh_params_Q[:, 0:4]
    sfr_params_Q = jnp.array(
        (
            sfh_params_Q[:, 0],
            sfh_params_Q[:, 1],
            sfh_params_Q[:, 2],
            UH + jnp.zeros(sfh_params_Q.shape[0]),
            sfh_params_Q[:, 3],
        )
    ).T
    q_params_Q = sfh_params_Q[:, 4:8]

    args_Q = (t_table, mah_params_sampled, sfr_params_Q, q_params_Q)
    sfr_Q = sfh_galpop(
        *args_Q, ms_param_type="unbounded", q_param_type="unbounded", lgt0=LGT0, fb=FB
    )
    mstar_Q = cumulative_mstar_formed_vmap(t_table, sfr_Q)

    sfr_params_MS = jran.multivariate_normal(
        mainseq_key, means_mainseq, covs_mainseq, shape=(n_histories,)
    )
    sfr_params_MS = sfr_params_MS + shifts_mainseq
    sfr_params_MS = jnp.array(
        (
            sfr_params_MS[:, 0],
            sfr_params_MS[:, 1],
            sfr_params_MS[:, 2],
            UH + jnp.zeros(sfr_params_MS.shape[0]),
            sfr_params_MS[:, 3],
        )
    ).T

    args_MS = (t_table, mah_params_sampled, sfr_params_MS)
    sfr_MS = ms_sfh_galpop_u_params(*args_MS, lgt0, fb)
    mstar_MS = cumulative_mstar_formed_vmap(t_table, sfr_MS)

    mstar = jnp.concatenate((mstar_Q, mstar_MS))
    sfr = jnp.concatenate((sfr_Q, sfr_MS))

    weight = jnp.concatenate((frac_quench, (1.0 - frac_quench)))
    p50_sampled = jnp.concatenate((p50_sampled, p50_sampled))
    return mstar, sfr, p50_sampled, weight


@jjit
def draw_single_sfh_MIX_with_exsitu(
    t_table,
    mah_params,
    p50,
    ran_key,
    pdf_parameters_Q=np.array(list(DEFAULT_SFH_PDF_QUENCH_PARAMS.values())),
    pdf_parameters_MS=np.array(list(DEFAULT_SFH_PDF_MAINSEQ_PARAMS.values())),
    R_model_params_Q=np.array(list(DEFAULT_R_QUENCH_PARAMS.values())),
    R_model_params_MS=np.array(list(DEFAULT_R_MAINSEQ_PARAMS.values())),
):
    """
    Generate Monte Carlo realization of the star formation histories of
    a mixed population of quenched and main sequence galaxies
    for a single halo mass bin.

    There is correlation with p50.

    Parameters
    ----------
    t_table : ndarray of shape (n_times, )
        Cosmic time array in Gyr.
    logmh : float
        Base-10 log of present-day halo mass of the halo population
    mah_params : ndarray of shape (n_mah_haloes x n_mah_params)
        Array with the diffmah parameters that will be marginalized over. Could
        be either individual fits of n_mah_haloes haloes, or be n_mah_haloes
        samples from a population model. They are chosen at random n_halos times.
    p50 : mah_params : ndarray of shape (n_mah_haloes, )
        Formation time percentile of each halo conditioned on halo mass.
    ran_key : ndarray of shape 2
        JAX random key.
    pdf_model_params_Q : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the quenched population.
        Default is DEFAULT_SFH_PDF_QUENCH_PARAMS.
    pdf_model_params_MS : ndarray of shape (n_pdf, )
        Array containing the Diffstarpop parameters for the main sequence population.
        Default is DEFAULT_SFH_PDF_MAINSEQ_PARAMS.
    R_model_params_Q: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the quenched population.
    R_model_params_MS: ndarray of shape (n_R, )
        Array containing the Diffstarpop parameters for the correlation between
        diffstar and diffmah parameters for the main sequence population.

    Returns
    -------
    sfr : ndarray of shape (n_histories, n_times)
        Stores star formation rate history in units of Msun/yr.
    """
    logmh = jnp.atleast_1d(mah_params[0])

    (quench_key, mainseq_key, ran_key) = jran.split(ran_key, 3)

    _res = get_smah_means_and_covs_mainseq(logmh, *pdf_parameters_MS)
    means_mainseq, covs_mainseq = _res
    means_mainseq = means_mainseq[0]
    covs_mainseq = covs_mainseq[0]

    R_vals_mainseq = _get_slopes_mainseq(logmh, *R_model_params_MS)
    R_vals_mainseq = jnp.array(R_vals_mainseq)[:, 0]
    shifts_mainseq = R_vals_mainseq * (p50 - 0.5)

    _res = get_smah_means_and_covs_quench(logmh, *pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = frac_quench[0]
    means_quench = means_quench[0]
    covs_quench = covs_quench[0]

    _res = _get_slopes_quench(logmh, *R_model_params_Q)
    R_Fquench, R_vals_quench = _res[0], _res[1:]
    R_vals_quench = jnp.array(R_vals_quench)[:, 0]
    shifts_quench = R_vals_quench * (p50 - 0.5)
    shifts_Fquench = R_Fquench * (p50 - 0.5)
    fquench_x0 = pdf_parameters_Q[0] + shifts_Fquench
    frac_quench = frac_quench_vs_lgm0(logmh, fquench_x0, *pdf_parameters_Q[1:4])

    sfh_params_Q = jran.multivariate_normal(
        quench_key, means_quench, covs_quench, shape=(1,)
    )
    sfh_params_Q = sfh_params_Q + shifts_quench

    sfr_params_Q = sfh_params_Q[:, 0:4]
    q_params_Q = sfh_params_Q[:, 4:8]

    sfr_params_MS = jran.multivariate_normal(
        mainseq_key, means_mainseq, covs_mainseq, shape=(1,)
    )
    sfr_params_MS = sfr_params_MS + shifts_mainseq
    # q_params_MS = jnp.ones_like(q_params_Q) * 10.0

    sfr_params_Q = sfr_params_Q[0]
    q_params_Q = q_params_Q[0]
    sfr_params_MS = sfr_params_MS[0]

    sfr_Q_insitu = sfr_history_diffstar_scan(
        t_table,
        mah_params,
        sfr_params_Q,
        q_params_Q,
    )
    sfr_MS_insitu = sfr_history_diffstar_scan_MS(
        t_table,
        mah_params,
        sfr_params_MS,
    )

    sfr_Q = sfr_Q_insitu
    sfr_MS = sfr_MS_insitu

    sfr = jnp.array([sfr_Q, sfr_MS])

    frac_quench = frac_quench[0]
    weight = jnp.array([frac_quench, (1.0 - frac_quench)])

    return sfr, weight
