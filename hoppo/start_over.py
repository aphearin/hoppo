"""
"""
from collections import OrderedDict

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp

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
def draw_single_sfh_MIX_with_exsitu(
    t_table,
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
    mean_ms = _get_mean_smah_params_mainseq(lgm0, **means_mainseq_pdict)
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

    mean_q = _get_mean_smah_params_quench(lgm0, **means_q_pdict)
    chol_params_q = _get_chol_params_quench(lgm0, **cov_q_pdict)
    cov_q = _get_cov_scalar_q(*chol_params_q)

    R_model_params_q = jnp.array(list(R_model_pdict_Q.values()))
    R_vals_q = _get_slopes_quench(lgm0, *R_model_params_q)
    shifts_q = jnp.array(R_vals_q) * (p50 - 0.5)

    return mean_ms, cov_ms, shifts_ms, mean_q, cov_q, shifts_q
