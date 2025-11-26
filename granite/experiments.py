"""Class for minimizing the superset of a functional decomposition using a decision tree."""
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np

from .utils import decomposition_to_R


# marginal_pure_vs_full -> difference between pure and full interactions in a FD
# marginal_pure_vs_full_loss -> difference between pure nu(i) - nu(empty) and full nu(D) - nu(-i) loss (SAGE)
# marginal_pure_vs_full_variance -> difference between the marginal pure and full Sobol indices

def check_loss_args(decomposition, U, binned_features: Optional[list[np.ndarray]]=None):
    U = U
    assert isinstance(U, (list, tuple))
    assert isinstance(U[0], (list, tuple))
    for i, u in enumerate(U):
        N_1, N_2 =  np.shape(decomposition[u])
        assert N_1 == N_2, "An Anchored decomposition with foreground=background is needed"
        if binned_features is not None:
            assert N_1 == len(binned_features[i])



def get_marginal_pure_vs_full_loss_fn(
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]]
            ):
    """
    Return the loss_fn that computes disagreements between the pure and full effects of the marginal decomposition.

    Parameters
    ----------
    decomposition : dict{Tuple: np.ndarray}
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.  `decomposition[(0,)].shape = (N, N)`.
    U : List[Tuple[int]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all interactions involving one of these features.
        """
    check_loss_args(decomposition, U)
    H = np.stack([decomposition[u] for u in U], axis=-1)  # (N, N, |U|)
    len_u = np.array([len(u) for u in U])

    def loss_fn(instances_idx: np.ndarray[int]):
        instances_idx = instances_idx[:, np.newaxis]
        H_subset = H[instances_idx, instances_idx.T]
        loss = np.sum((H_subset.mean(0) - (-1)**len_u*H_subset.mean(1))**2)
        return loss / len(U)
    return loss_fn



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



def get_marginal_pure_vs_full_risk_loss_fn(
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]],
            y: np.ndarray,
            risk_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
            ):

    check_loss_args(decomposition, U)
    assert callable(risk_fn)
    assert len(y) == len(decomposition[()])
    R = decomposition_to_R(decomposition)
    R = np.stack([R[u] for u in U], axis=-1)  # (N, N, |U|)
    predictions = decomposition[()]
    local_y = y
    local_risk_fn = risk_fn


    def get_loss(instances_idx: np.ndarray):
        """
        For each subset u in U, we report the pure-vs-full interaction disagreements and average them.
        """
        pred_subset = predictions[instances_idx]
        y_subset = local_y[instances_idx]
        instances_idx = instances_idx[:, np.newaxis]
        R_subset = R[instances_idx, instances_idx.T]
        d = R.shape[2]
        pure_effect, full_effect = get_pure_full_risk_effects(
                                                            R_subset,
                                                            pred_subset,
                                                            y_subset,
                                                            local_risk_fn
                                                        )
        return np.sum((full_effect - pure_effect)**2) / d
    return get_loss



def get_closed_total_sobol(H: np.ndarray):
    closed_sobol = H.mean(1).var(0)
    total_sobol = H.var(0).mean(0)
    return closed_sobol, total_sobol



def get_marginal_pure_vs_full_variance_loss_fn(
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]],
            ):
    """
    Return the loss function that computes the error between Closed Sobol and Total Sobol.

    Parameters
    ----------
    decomposition : Dict[Tuple[int], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : List[Tuple[int]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional effects of these features.
    """

    check_loss_args(decomposition, U)
    H = []
    for u in U:
        assert len(u) == 1, f"Sobol-vs-TotalSobol only suppports main effects. Not {len(u)}-way interactions."
        H.append(decomposition[u])
    H = np.stack(H, axis=-1)

    def loss_fn(instances_idx: np.ndarray[int]):
        # Iterate over all splits
        instances_idx = instances_idx[:, np.newaxis]
        H_subset = H[instances_idx, instances_idx.T]
        closed_sobol, total_sobol = get_closed_total_sobol(H_subset)
        loss = np.sum((closed_sobol - total_sobol)**2)
        return loss
    return loss_fn


# marginal_vs_conditional_pure -> difference between PDP and MPlot
# marginal_vs_conditional_full_variance -> difference marginal and conditional total Sobol indices
# marginal_vs_conditional_full_risk -> difference between PFI and cPFI


def get_marginal_conditional_effects(binned_feature: np.ndarray[int], R: np.ndarray):
    marginal_effect = R.mean(1)
    conditional_effect = np.zeros((binned_feature.shape[0],))
    for b in np.unique(binned_feature):
        inside_bin = np.where(binned_feature == b)[0][:, np.newaxis]
        conditional_effect[inside_bin.ravel()] = R[inside_bin, inside_bin.T].mean(1)
    return marginal_effect, conditional_effect



def get_marginal_vs_conditional_pure_loss_fn(
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]],
            binned_features: List[np.ndarray[int]]
            ):
    """
    Return the loss function that computes the error between PDP and MPlot for a subset of features

    Parameters
    ----------
    decomposition : Dict[Tuple[int], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : List[Tuple[int]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional effects of these features.
    binned_features : List[np.ndarray[int]]
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


    def loss_fn(instances_idx: np.ndarray[int]):
        # Iterate over all splits
        instances_idx = instances_idx[:, np.newaxis]
        loss = 0
        for R, binned_feature in zip(R_matrices, local_binned_features):
            R_subset = R[instances_idx, instances_idx.T]
            binned_feature_subset = binned_feature[instances_idx.ravel()]
            marginal, conditional = get_marginal_conditional_effects(binned_feature_subset, R_subset)
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
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]],
            binned_features: List[np.ndarray[int]]
            ):
    """
    Return the loss function that computes the error between Marginal and Conditional Sobol

    Parameters
    ----------
    decomposition : Dict[Tuple[int], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : List[Tuple[int]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional effects of these features.
    binned_features : List[np.ndarray[int]]
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

    def loss_fn(instances_idx: np.ndarray[int]):
        # Iterate over all splits
        instances_idx = instances_idx[:, np.newaxis]
        loss = 0
        for R, binned_feature in zip(R_matrices, local_binned_features):
            R_subset = R[instances_idx, instances_idx.T]
            binned_feature_subset = binned_feature[instances_idx.ravel()]
            marginal, conditional = get_marginal_conditional_sobol(binned_feature_subset, R_subset)
            loss += np.sum((marginal - conditional)**2)
        return loss
    return loss_fn



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
    return float(marginal_risk) - risk_all_features, \
           float(conditional_risk) - risk_all_features



def get_marginal_vs_condition_full_risk_loss_fn(
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]],
            binned_features: List[np.ndarray[int]],
            y: np.ndarray,
            risk_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
            ):
    """
    Return the loss function that computes the error between PFI and cPFI

    Parameters
    ----------
    decomposition : Dict[Tuple[int], np.ndarray]
        The functional decomposition used to compute the objective.
        It needs to be anchored with foreground=background i.e.
        `decomposition[(0,)].shape = (N, N)`.
    U : List[Tuple[int]]
        A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
        passing `U=[(0,), (1,), (2,)]` will minimize all marginal-vs-conditional effects of these features.
    binned_features : List[np.ndarray[int]]
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
    risk_all_features = risk_fn(decomposition[()], y).mean()

    def loss_fn(instances_idx: np.ndarray[int]):
        # Iterate over all splits
        instances_idx = instances_idx[:, np.newaxis]
        loss = 0
        for R, binned_feature in zip(R_matrices, local_binned_features):
            R_subset = R[instances_idx, instances_idx.T]
            y_subset = local_y[instances_idx.ravel()]
            binned_feature_subset = binned_feature[instances_idx.ravel()]
            marginal, conditional = get_marginal_conditional_full_risk(
                                                    binned_feature_subset,
                                                    R_subset,
                                                    y_subset,
                                                    local_risk_fn,
                                                    risk_all_features
                                                )
            loss += np.sum((marginal - conditional)**2) * len(instances_idx)
        return loss
    return loss_fn
