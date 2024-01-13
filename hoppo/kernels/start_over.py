"""
"""
from collections import OrderedDict

from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS, _get_chol_params_mainseq
from .pdf_mainseq import _get_cov_scalar as _get_cov_scalar_ms
from .pdf_mainseq import _get_mean_smah_params_mainseq
from .pdf_model_assembly_bias_shifts import (
    DEFAULT_R_MAINSEQ_PARAMS,
    DEFAULT_R_QUENCH_PARAMS,
    _get_slopes_mainseq,
    _get_slopes_quench,
)
from .pdf_quenched import DEFAULT_SFH_PDF_QUENCH_PARAMS, _get_chol_params_quench
from .pdf_quenched import _get_cov_scalar as _get_cov_scalar_q
from .pdf_quenched import _get_mean_smah_params_quench


@jjit
def mc_diffstar_u_params_singlegal(
    mah_params,
    p50,
    ran_key,
    pdf_pdict_Q=DEFAULT_SFH_PDF_QUENCH_PARAMS,
    pdf_pdict_MS=DEFAULT_SFH_PDF_MAINSEQ_PARAMS,
    R_model_pdict_Q=DEFAULT_R_QUENCH_PARAMS,
    R_model_pdict_MS=DEFAULT_R_MAINSEQ_PARAMS,
):
    lgm0 = mah_params[0]

    # main sequence
    means_mainseq_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_MS.items() if "mean_" in key]
    )
    cov_mainseq_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_MS.items() if "chol_" in key]
    )
    mu_ms = _get_mean_smah_params_mainseq(lgm0, **means_mainseq_pdict)
    chol_params_ms = _get_chol_params_mainseq(lgm0, **cov_mainseq_pdict)
    cov_ms = _get_cov_scalar_ms(*chol_params_ms)

    R_model_params_ms = jnp.array(list(R_model_pdict_MS.values()))
    R_vals_ms = _get_slopes_mainseq(lgm0, *R_model_params_ms)
    shifts_ms = jnp.array(R_vals_ms) * (p50 - 0.5)

    # Quenched sequence
    means_q_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_Q.items() if "mean_" in key]
    )
    cov_q_pdict = OrderedDict(
        [(key, val) for (key, val) in pdf_pdict_Q.items() if "chol_" in key]
    )

    mu_q = _get_mean_smah_params_quench(lgm0, **means_q_pdict)
    chol_params_q = _get_chol_params_quench(lgm0, **cov_q_pdict)
    cov_q = _get_cov_scalar_q(*chol_params_q)

    R_model_params_q = jnp.array(list(R_model_pdict_Q.values()))
    R_vals_q = _get_slopes_quench(lgm0, *R_model_params_q)
    shifts_q = jnp.array(R_vals_q) * (p50 - 0.5)

    # This part is not implemented yet either, but we need a scalar value of F_q
    # Compute quenched fraction
    # frac_quench = frac_quench_vs_lgm0(lgm0, fquench_x0, *pdf_parameters_Q[1:4])

    # Monte Carlo draws
    ms_key, q_key, frac_q_key = jran.split(ran_key, 3)

    u_params_q = jran.multivariate_normal(q_key, jnp.array(mu_q), cov_q, shape=())
    u_params_ms = jran.multivariate_normal(q_key, jnp.array(mu_ms), cov_ms, shape=())

    u_params_ms = u_params_ms + shifts_ms

    # Next line fails because shifts_q.shape=(9,) u_params_q.shape=(8,)
    # u_params_q = u_params_q + shifts_q

    uran = jran.uniform(frac_q_key, minval=0, maxval=1, shape=())

    # Finally need to implement this properly
    # should use the scalar value of F_q calculated above
    # Need to take care of u_params_q and u_params_ms having different dimension
    # u_params = jnp.where(uran < frac_q, u_params_q, u_params_ms)

    return u_params_q, u_params_ms, shifts_q
