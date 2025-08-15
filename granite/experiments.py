"""Class for minimizing the superset of a functional decomposition using a decision tree."""
from typing import Dict, List, Tuple
import numpy as np

from .fd_trees import FDTree


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



def get_marginal_conditional_effects(binned_feature, R):
    marginal_effect = R.mean(1)
    conditional_effect = np.zeros((binned_feature.shape[0],))
    for b in np.unique(binned_feature):
        inside_bin = np.where(binned_feature == b)[0][:, np.newaxis]
        conditional_effect[inside_bin.ravel()] = R[inside_bin, inside_bin.T].mean(1)
    return marginal_effect, conditional_effect


# TODO improve the API of this class
class PDPvsMPlot(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, decomposition, i, binned_feature):
        """
        Fit the FDTree

        Parameters
        ----------
        X : (N, n_features) np.ndarray
            The data on which to fit the tree. The ith column of `X` must be the ith feature
            in the Features object passed to the constructor.
        decomposition : dict{Tuple: np.ndarray}
            The functional decomposition used to compute the objective.
            It needs to be anchored with foreground=background i.e.
            `decomposition[(0,)].shape = (N, N)`.
        i : int
            The index of the feature whose marginal and conditional effects are computed.
        binned_feature : np.ndarray[int]
            Binned values for feature i, used to compute conditional expectations
        """
        super().fit(X)

        assert np.shape(decomposition[(i,)]) == (self.N, self.N), "An Anchored decomposition with foreground=background is needed"
        self.R = decomposition[(i,)] + decomposition[()]
        self.loss_factor = 1 / self.R.var()
        self.binned_data = binned_feature
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
        R = self.R[instances_idx, instances_idx.T]
        marginal, conditional = get_marginal_conditional_effects(self.binned_data[instances_idx.ravel()], R)
        return np.sum((marginal - conditional)**2)
