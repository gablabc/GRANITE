"""
Functions to compute feature effects using the anchored functional
decomposition as a starting point. All public functions use a
decomposition as inputs, while private functions (starting with `_`)
take another form of input which is more optimized.
"""

from typing import Dict, List, Tuple, Callable
import numpy as np

from .utils import decomposition_to_R

###################
#      Local
###################


# Public function to compute PDP and M-Plots from decompositions, binned_features,
# and a single feature_idx
def get_marginal_conditional_effects(
    binned_features: List[np.ndarray],
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    feature_idx: int,
):
    R_i = decomposition[(feature_idx,)] + decomposition[()]  # (N, N)
    binned_feature = binned_features[feature_idx]
    return _get_marginal_conditional_effects(binned_feature, R_i)


# Private variant that directly takes the (N, N) R_i matrix.
# This function is used internally by the GRANITE loss functions.
def _get_marginal_conditional_effects(
    binned_feature: np.ndarray,
    R_i: np.ndarray,
):
    marginal_effect = R_i.mean(1)
    conditional_effect = np.zeros((binned_feature.shape[0],))
    for b in np.unique(binned_feature):
        inside_bin = np.where(binned_feature == b)[0]
        conditional_effect[inside_bin] = R_i[inside_bin[:, np.newaxis], inside_bin].mean(1)
    return marginal_effect, conditional_effect


###################
#     Variance
###################

# Private variant that directly takes the (N, N, d) H_tensor as input.
# This function is used internally by the GRANITE loss functions.
def _get_closed_total_sobol(H_tensor: np.ndarray):
    closed_sobol = H_tensor.mean(1).var(0)
    total_sobol = H_tensor.var(0).mean(0)
    return closed_sobol, total_sobol


# Public API recommended to users
def get_closed_total_sobol(decomposition: Dict[Tuple[int, ...], np.ndarray]):
    H_tensor = np.stack(
        [decomposition[u] for u in decomposition.keys() if len(u) == 1],
        axis=-1
    )
    return _get_closed_total_sobol(H_tensor)


# Public API that uses decompositions and binned_features
def get_marginal_conditional_sobol(
    binned_features: List[np.ndarray],
    decomposition: Dict[Tuple[int, ...], np.ndarray],
):
    R = decomposition_to_R(decomposition)
    return _get_marginal_conditional_sobol(binned_features, R)


# Private variant that directly takes the R dictionnary.
# This function is used internally by the GRANITE loss functions.
def _get_marginal_conditional_sobol(
    binned_features: List[np.ndarray],
    R: Dict[Tuple[int, ...], np.ndarray],
):
    n_features = len(binned_features)
    n_instances = len(binned_features[0])
    marginal_sobol = np.zeros(n_features)
    cond_sobol = np.zeros(n_features)
    for i in range(n_features):
        R_i = R[(i,)]
        marginal_sobol[i] = R_i.var(0).mean()
        for b in np.unique(binned_features[i]):
            inside_bin = np.where(binned_features[i] == b)[0]
            ratio = len(inside_bin) / n_instances
            cond_sobol[i] += ratio * R_i[inside_bin[:, np.newaxis], inside_bin].var(0).mean()
    return marginal_sobol, cond_sobol


###################
#      Risk
###################

# TODO

def get_pure_full_risk_effects(
    R: np.ndarray,
    preds: np.ndarray,
    y: np.ndarray,
    loss_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
):
    # nu(S) = loss_func( E[f(x_s, X_{-S}] , y )
    N = len(preds)
    # nu(i) - nu(empty)
    pure_effect = loss_func(R.mean(1), y) - loss_func(preds.mean()*np.ones(N), y)
    # nu(D) - nu(-i)
    full_effect = loss_func(preds, y) - loss_func(R.mean(0), y)
    return -1 * pure_effect, -1 * full_effect

def get_marginal_conditional_full_risk(binned_feature, R, y, risk_fn, risk_all_features=0.0):
    n_instances = len(binned_feature)
    marginal_risk = risk_fn(R.mean(0), y).mean(0)
    conditional_risk = 0.0
    for b in np.unique(binned_feature):
        inside_bin = np.where(binned_feature == b)[0][:, np.newaxis]
        ratio = len(inside_bin) / n_instances
        subset_R = R[inside_bin, inside_bin.T].mean(0)
        y_subset = y[inside_bin.ravel()]
        conditional_risk += ratio * risk_fn(subset_R, y_subset).mean(0)
    return float(marginal_risk) - risk_all_features, float(conditional_risk) - risk_all_features
