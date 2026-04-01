"""
loss functions that aggregate disagreements between explanation methods
caused by interaction or correlations.
"""

from typing import Dict, List, Tuple, Callable, Optional, Sequence
import numpy as np
from .utils import decomposition_to_R
from .experiments import (
    _get_closed_total_sobol,
    _get_marginal_conditional_effects,
    _get_marginal_conditional_sobol,
    _get_pure_full_risk_effects,
    _get_marginal_conditional_full_risk,
)

# marginal_pure_vs_full -> difference between pure and full interactions in a FD
# marginal_pure_vs_full_loss -> difference between pure nu(i) - nu(empty) and full nu(D) - nu(-i) loss (SAGE)
# marginal_pure_vs_full_variance -> difference between the marginal pure and full Sobol indices

def check_loss_args(
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    U: Sequence[Tuple[int, ...]],
    binned_features: Optional[List[np.ndarray]]=None,
):
    for i, u in enumerate(U):
        assert isinstance(u, tuple)
        assert isinstance(decomposition[u], np.ndarray)
        N_1, N_2 =  np.shape(decomposition[u])
        assert N_1 == N_2, "An Anchored decomposition with foreground=background is needed"
        if binned_features is not None:
            assert N_1 == len(binned_features[i])


def get_marginal_pure_vs_full_loss_fn(
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    U: Sequence[Tuple[int, ...]]
):
    """
    Return the loss_fn that computes disagreements between the pure and full local marginal effects.

    Parameters
    ----------
    decomposition : Dict[Tuple[int, ...], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : Sequence[Tuple[int, ...]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized.
        For example passing `U=[(0,), (1,), (2,)]` will minimize all interactions involving
        one of these three features.
        """
    check_loss_args(decomposition, U)
    H = np.stack([decomposition[u] for u in U], axis=-1)  # (N, N, |U|)
    len_u = np.array([len(u) for u in U])

    def loss_fn(instances_idx: np.ndarray):
        H_subset = H[instances_idx[:, np.newaxis], instances_idx]
        loss = np.sum(( H_subset.mean(0) - (-1)**len_u*H_subset.mean(1) )**2)
        return loss / len(U)
    return loss_fn



def get_marginal_pure_vs_full_risk_loss_fn(
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    U: Sequence[Tuple[int, ...]],
    y: np.ndarray,
    risk_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
):

    check_loss_args(decomposition, U)
    assert callable(risk_fn)
    assert len(y) == len(decomposition[()])
    R_dict = decomposition_to_R(decomposition)
    R_tensor = np.stack([R_dict[u] for u in U], axis=-1)  # (N, N, |U|)
    n_features = R_tensor.shape[2]
    predictions = decomposition[()]
    local_y = y
    local_risk_fn = risk_fn

    def get_loss(instances_idx: np.ndarray):
        """
        For each subset u in U, we report the pure-vs-full interaction
        disagreements and average them.
        """
        pred_subset = predictions[instances_idx]
        y_subset = local_y[instances_idx]
        R_subset = R_tensor[instances_idx[:, np.newaxis], instances_idx]
        pure_effect, full_effect = _get_pure_full_risk_effects(
                                                            R_subset,
                                                            pred_subset,
                                                            y_subset,
                                                            local_risk_fn
                                                        )
        return np.sum((full_effect - pure_effect)**2) / n_features
    return get_loss



def get_marginal_pure_vs_full_variance_loss_fn(
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    U: Sequence[Tuple[int, ...]],
):
    """
    Return the loss function that computes the error between Closed Sobol and Total Sobol i.e
    pure-vs-full marginal sensitivity methods.

    Parameters
    ----------
    decomposition : Dict[Tuple[int, ...], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e. `decomposition[(0,)].shape = (N, N)`.
    U : Sequence[Tuple[int, ...]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized.
        For example passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional
        effects accross these three features. This function only supports subsets of a single
        feature.
    """

    check_loss_args(decomposition, U)
    try:
        H = np.stack([decomposition[u] for u in U if len(u) == 1], axis=-1)  # (N, N, |U|)
    except ValueError:
        raise ValueError(
            """Closed and Total Sobol effects are only implemented for individual features.
            Thus, U must contain singleton sets e.g. U=[(0,), (1,)]."""
        )

    def loss_fn(instances_idx: np.ndarray):
        # Iterate over all splits
        H_subset = H[instances_idx[:, np.newaxis], instances_idx]
        closed_sobol, total_sobol = _get_closed_total_sobol(H_subset)
        loss = np.sum((closed_sobol - total_sobol)**2)
        return loss
    return loss_fn


# marginal_vs_conditional_pure -> difference between PDP and MPlot
# marginal_vs_conditional_full_variance -> difference marginal and conditional total Sobol indices
# marginal_vs_conditional_full_risk -> difference between PFI and cPFI


def get_marginal_vs_conditional_pure_loss_fn(
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    U: Sequence[Tuple[int, ...]],
    binned_features: List[np.ndarray]
):
    """
    Return the loss function that computes the error between PDP and MPlot for a subset of features

    Parameters
    ----------
    decomposition : Dict[Tuple[int, ...], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : Sequence[Tuple[int. ...]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional effects of these features.
    binned_features : List[np.ndarray]
        The ith element is an array of shape (N,) that stores the binned value of feature i.
        Used to compute conditional expectations.
    """

    check_loss_args(decomposition, U, binned_features)
    R_matrices = []
    for u in U:
        assert len(u) == 1, f"PDPvsMPlot only suppports main effects. Not {len(u)}-way interactions."
        R_matrices.append(decomposition[u] + decomposition[()])
    assert len(U) == len(binned_features), "A binned_features must be provided for each features in U."
    local_binned_features = binned_features

    def loss_fn(instances_idx: np.ndarray):
        # Iterate over all splits
        loss = 0
        for R, binned_feature in zip(R_matrices, local_binned_features):
            R_subset = R[instances_idx[:, np.newaxis], instances_idx]
            binned_feature_subset = binned_feature[instances_idx]
            marginal, conditional = _get_marginal_conditional_effects(binned_feature_subset, R_subset)
            loss += np.sum((marginal - conditional)**2)
        return loss
    return loss_fn




def get_marginal_conditional_sobol(binned_feature, R):
    n_instances = len(binned_feature)
    marginal_sobol = R.var(0).mean()
    conditional_sobol = 0.0
    for b in np.unique(binned_feature):
        inside_bin = np.where(binned_feature == b)[0][:, np.newaxis]
        ratio = len(inside_bin) / n_instances
        conditional_sobol += ratio * R[inside_bin, inside_bin.T].var(0).mean()
    return marginal_sobol, conditional_sobol



def get_marginal_vs_condition_full_variance_loss_fn(
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    U: List[Tuple[int, ...]],
    binned_features: List[np.ndarray],
):
    """
    Return the loss function that computes the error between Marginal and Conditional Sobol

    Parameters
    ----------
    decomposition : Dict[Tuple[int, ...], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : List[Tuple[int, ...]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional effects of these features.
    binned_features : List[np.ndarray]
        The ith element is an array of shape (N,) that stores the binned value of feature i.
        Used to compute conditional expectations.
    """

    check_loss_args(decomposition, U, binned_features)
    R_matrices = []
    for u in U:
        assert len(u) == 1, f"Marginal-vs-Conditional Sobol only suppports main effects. Not {len(u)}-way interactions."
        R_matrices.append(decomposition[u] + decomposition[()])
    assert len(U) == len(binned_features), "A binned_features must be provided for each features in U."
    local_binned_features = binned_features

    def loss_fn(instances_idx: np.ndarray):
        # Iterate over all splits
        loss = 0
        for R_i, binned_feature in zip(R_matrices, local_binned_features):
            R_i_region = R_i[instances_idx[:, np.newaxis], instances_idx]
            binned_feature_subset = binned_feature[instances_idx]
            marginal, conditional = _get_marginal_conditional_sobol(
                binned_feature_subset,
                R_i_region
            )
            loss += np.sum((marginal - conditional)**2)
        return loss
    return loss_fn




def get_marginal_vs_condition_full_risk_loss_fn(
    decomposition: Dict[Tuple[int, ...], np.ndarray],
    U: Sequence[Tuple[int, ...]],
    binned_features: List[np.ndarray],
    y: np.ndarray,
    risk_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
):
    """
    Return the loss function that computes the error between PFI and cPFI

    Parameters
    ----------
    decomposition : Dict[Tuple[int, ...], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : List[Tuple[int, ...]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional effects of these features.
    binned_features : List[np.ndarray]
        The ith element is an array of shape (N,) that stores the binned value of feature i.
        Used to compute conditional expectations.
    """

    check_loss_args(decomposition, U, binned_features)
    R_matrices = []
    for u in U:
        assert len(u) == 1, f"PFI-vs-cPFI only suppports main effects. Not {len(u)}-way interactions."
        R_matrices.append(decomposition[u] + decomposition[()])
    assert len(U) == len(binned_features), "A binned_features must be provided for each features in U."
    local_binned_features = binned_features
    local_y = y
    local_risk_fn = risk_fn

    def loss_fn(instances_idx: np.ndarray):
        # Iterate over all splits
        loss = 0
        for R_i, binned_feature in zip(R_matrices, local_binned_features):
            R_i_region = R_i[instances_idx[:, np.newaxis], instances_idx]
            y_region = local_y[instances_idx]
            binned_feature_subset = binned_feature[instances_idx]
            marginal, conditional = _get_marginal_conditional_full_risk(
                                        binned_feature_subset,
                                        R_i_region,
                                        y_region,
                                        local_risk_fn,
                                    )
            loss += np.sum((marginal - conditional)**2) * len(instances_idx)
        return loss
    return loss_fn
