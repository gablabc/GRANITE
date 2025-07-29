"""Class for minimizing the superset of a functional decomposition using a decision tree."""
import numpy as np

from .fd_trees import FDTree


class MinimizeSuperset(FDTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, decomposition, u):
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
        u : Tuple(int)
            The subset whose pure-full disagreement is minimized
        """

        self.X = X
        self.N, self.D = X.shape
        self.u = u
        assert np.shape(decomposition[u]) == (self.N, self.N), "An Anchored decomposition with foreground=background is needed"
        self.H = decomposition[u]
        self.loss_factor = 1 / self.H.mean(1).var()
        self.n_regions = 0
        # Loss function is the difference between full and partial effects
        loss = np.mean((self.H.mean(0) - (-1)**len(u)*self.H.mean(1))**2)
        # Start recursive tree growth
        self.root, self.final_objective, self.n_regions = self._tree_builder(np.arange(self.N), depth=0, loss=loss)
        self.final_loss = self.final_objective - self.alpha * self.n_regions
        return self


    def _get_objective_for_splits(self, instances_idx, feature):
        """
        Given the indices of the datapoints in the current leaf, compute
        the loss function for a variety of split candidates.
        """
        x_i = self.X[instances_idx, feature]

        splits = self._get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], []

        # Otherwise we optimize the objective
        loss_left = np.zeros(len(splits))
        loss_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split][:, np.newaxis]
            right = instances_idx[x_i > split][:, np.newaxis]
            H_left = self.H[left, left.T]
            H_right = self.H[right, right.T]
            to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
            loss_left[i]  = np.sum((H_left.mean(0) - (-1)**len(self.u)*H_left.mean(1))**2)
            loss_right[i]  = np.sum((H_right.mean(0) - (-1)**len(self.u)*H_right.mean(1))**2)

        return splits[to_keep], loss_left[to_keep], loss_right[to_keep]


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

        self.X = X
        self.N, self.D = X.shape
        assert np.shape(decomposition[(i,)]) == (self.N, self.N), "An Anchored decomposition with foreground=background is needed"
        self.R = decomposition[(i,)] + decomposition[()]
        self.loss_factor = 1 / self.R.var()
        self.binned_data = binned_feature
        self.n_regions = 0
        marginal_effect, conditional_effect = get_marginal_conditional_effects(binned_feature, self.R)
        # The loss is the difference between the marginal and conditional effects
        loss = np.mean((marginal_effect - conditional_effect)**2)
        # Start recursive tree growth
        self.root, self.final_objective, self.n_regions = self._tree_builder(np.arange(self.N), depth=0, loss=loss)
        self.final_loss = self.final_objective - self.alpha * self.n_regions
        return self


    def _get_objective_for_splits(self, instances_idx, feature):
        x_i = self.X[instances_idx, feature]

        splits = self._get_split_candidates(x_i, feature)

        # No split possible
        if len(splits) == 0:
            return [], [], []

        # Otherwise we optimize the objective
        loss_left = np.zeros(len(splits))
        loss_right = np.zeros(len(splits))
        to_keep = np.zeros((len(splits))).astype(bool)

        # Iterate over all splits
        for i, split in enumerate(splits):
            left = instances_idx[x_i <= split][:, np.newaxis]
            right = instances_idx[x_i > split][:, np.newaxis]
            R_left = self.R[left, left.T]
            R_right = self.R[right, right.T]
            to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
            marginal_left, conditional_left = get_marginal_conditional_effects(self.binned_data[left.ravel()], R_left)
            marginal_right, conditional_right = get_marginal_conditional_effects(self.binned_data[right.ravel()], R_right)
            loss_left[i]  = np.sum((marginal_left - conditional_left)**2)
            loss_right[i]  = np.sum((marginal_right - conditional_right)**2)

        return splits[to_keep], loss_left[to_keep], loss_right[to_keep]
