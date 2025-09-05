"""Class for minimizing the superset of a functional decomposition using a decision tree."""
from typing import Dict, List, Tuple, Callable
import numpy as np

from .fd_trees import FDTree
from .utils import decomposition_to_R



class MinimizeSuperset(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(
            self,
            X: np.ndarray,
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]]
            ):
        """
        Fit the FDTree

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            The data on which to fit the tree. The ith column of `X` must be the ith feature
            in the Features object passed to the constructor.
        decomposition : dict{Tuple: np.ndarray}
            The functional decomposition used to compute the objective.
            It needs to be anchored with foreground=background i.e.  `decomposition[(0,)].shape = (N, N)`.
        U : List[Tuple[int]]
            A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
            passing `U=[(0,), (1,), (2,)]` will minimize all interactions involving one of these features.
            """
        super().fit(X)

        self.U = U
        assert isinstance(U, (list, tuple))
        assert isinstance(U[0], (list, tuple))
        assert np.shape(decomposition[U[0]]) == (self.N, self.N), "An Anchored decomposition with foreground=background is needed"
        self.H = np.stack([decomposition[u] for u in U], axis=-1)  # (N, N, |U|)
        self.len_u = np.array([len(u) for u in self.U])
        self.loss_factor = 1 / self.H.mean(1).var(0).mean()
        self.n_regions = 0
        # Loss function is the difference between pure and full  effects
        loss = self.get_loss(np.arange(self.N)) / self.N
        # Start recursive tree growth
        self.root, self.final_objective, self.n_regions = self._tree_builder(np.arange(self.N), depth=0, loss=loss)
        self.final_loss = self.final_objective - self.alpha * self.n_regions
        return self


    def get_loss(self, instances_idx: np.ndarray):
        """
        For each subset u in U, we report the pure-vs-full interaction disagreements and average them.
        """
        instances_idx = instances_idx[:, np.newaxis]
        H_subset = self.H[instances_idx, instances_idx.T]
        loss = np.sum((H_subset.mean(0) - (-1)**self.len_u*H_subset.mean(1))**2)
        return loss / len(self.U)




def get_pure_full_risk_effects(R, preds, y, loss_func):
    # nu(S) = loss_func( E[f(x_s, X_{-S}] - y )
    N = len(preds)
    # nu(i) - nu(empty)
    pure_effect = loss_func(R.mean(1), y) - loss_func(preds.mean()*np.ones(N), y)
    # nu(D) - nu(-i)
    full_effect = loss_func(preds, y) - loss_func(R.mean(0), y)
    return pure_effect, full_effect



class MinimizeLossGameSuperset(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]],
            loss_func: Callable[[np.ndarray], np.ndarray]
            ):
        """
        Fit the FDTree

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            The data on which to fit the tree. The ith column of `X` must be the ith feature
            in the Features object passed to the constructor.
        y : (N,) np.ndarray
            The label associated with each datapoint.
        decomposition : dict{Tuple: np.ndarray}
            The functional decomposition used to compute the objective.
            It needs to be anchored with foreground=background i.e.  `decomposition[(0,)].shape = (N, N)`.
        U : List[Tuple[int]]
            A subset of the powerset whose sum of pure-vs-full interactions are minimized. For example
            passing `U=[(0,), (1,), (2,)]` will minimize all interactions involving one of these features.
        loss_func : Callable
            Function that returns the point-wise loss l(y_hat, y), where y_hat could be a (N,) or (N, n_preds) array.
            See `utils.py` for common definitions of such functions.
        """
        super().fit(X)

        self.U = U
        assert isinstance(U, (list, tuple))
        assert isinstance(U[0], (list, tuple))
        assert np.shape(decomposition[U[0]]) == (self.N, self.N), "An Anchored decomposition with foreground=background is needed"
        R = []
        for u in U:
            assert len(u) == 1, f"MinimizeLossGameSuperset only suppports main effects. Not {len(u)}-way interactions."
            R.append(decomposition[u] + decomposition[()])
        self.R = np.stack(R, axis=-1)
        self.n_regions = 0
        # Loss function is the difference between pure and full effects
        self.predictions = decomposition[()]
        self.y = y
        self.loss_func = loss_func
        loss = self.get_loss(np.arange(self.N)) / self.N
        self.loss_factor = 1 / loss
        # Start recursive tree growth
        self.root, self.final_objective, self.n_regions = self._tree_builder(np.arange(self.N), depth=0, loss=loss)
        self.final_loss = self.final_objective - self.alpha * self.n_regions
        return self


    def get_loss(self, instances_idx: np.ndarray):
        """
        For each subset u in U, we report the pure-vs-full interaction disagreements and average them.
        """
        pred_subset = self.predictions[instances_idx]
        y_subset = self.y[instances_idx]
        instances_idx = instances_idx[:, np.newaxis]
        R_subset = self.R[instances_idx, instances_idx.T]
        pure_effect, full_effect = get_pure_full_risk_effects(
                                                            R_subset,
                                                            pred_subset,
                                                            y_subset,
                                                            self.loss_func
                                                        )
        return np.sum((full_effect - pure_effect)**2)



def get_marginal_conditional_effects(binned_feature, R):
    marginal_effect = R.mean(1)
    conditional_effect = np.zeros((binned_feature.shape[0],))
    for b in np.unique(binned_feature):
        inside_bin = np.where(binned_feature == b)[0][:, np.newaxis]
        conditional_effect[inside_bin.ravel()] = R[inside_bin, inside_bin.T].mean(1)
    return marginal_effect, conditional_effect



class PDPvsMPlot(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(
            self,
            X: np.ndarray,
            decomposition: Dict[Tuple[int], np.ndarray],
            U: List[Tuple[int]],
            binned_features: List[np.ndarray[int]]
            ):
        """
        Fit the FDTree

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            The data on which to fit the tree. The ith column of `X` must be the ith feature
            in the Features object passed to the constructor.
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
        super().fit(X)

        assert isinstance(U, (list, tuple))
        assert isinstance(U[0], (list, tuple))
        assert np.shape(decomposition[(0,)]) == (self.N, self.N), "An Anchored decomposition with foreground=background is needed"
        self.R = []
        self.loss_factor = 1
        for u in U:
            assert len(u) == 1, f"PDPvsMPlot only suppports main effects. Not {len(u)}-way interactions."
            self.R.append(decomposition[u] + decomposition[()])
            self.loss_factor += self.R[-1].var()
        self.loss_factor = 1 / self.loss_factor
        self.binned_features = binned_features
        self.n_regions = 0
        # The loss is the difference between the marginal and conditional effects
        loss = self.get_loss(np.arange(self.N)) / self.N
        # Start recursive tree growth
        self.root, self.final_objective, self.n_regions = self._tree_builder(np.arange(self.N), depth=0, loss=loss)
        self.final_loss = self.final_objective - self.alpha * self.n_regions
        return self


    def get_loss(self, instances_idx: np.ndarray):

        # Iterate over all splits
        instances_idx = instances_idx[:, np.newaxis]
        loss = 0
        for R, binned_feature in zip(self.R, self.binned_features):
            R_subset = R[instances_idx, instances_idx.T]
            binned_feature_subset = binned_feature[instances_idx.ravel()]
            marginal, conditional = get_marginal_conditional_effects(binned_feature_subset, R_subset)
            loss += np.sum((marginal - conditional)**2)
        return loss
