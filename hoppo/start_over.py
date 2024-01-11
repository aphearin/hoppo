"""
"""
import numpy as np
from jax import jit as jjit

from .pdf_mainseq import DEFAULT_SFH_PDF_MAINSEQ_PARAMS, _get_mean_smah_params_mainseq
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
    lgm0 = mah_params[0]
    _res = _get_mean_smah_params_mainseq(lgm0)
    ulgm, ulgy, ul, utau = _res
