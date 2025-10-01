"""This module contains all tabular machine learning games."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from shapiq import Game

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray


class GlobalRiskGame(Game):
    """The Global Risk Cooperative Game.

    The GlobalExplanation game is a benchmark game for global explanation methods. It evaluates the
    worth of coalitions of features towards the model's performance. The players are individual
    features, and the worth of a coalition is the performance of the model on a random subset of the
    data where missing features are removed by setting the feature values to a random value from the
    background data. For more details, we highly recommend reading the SAGE paper [1]_ or the
    related blog post [2]_.

    Attributes:
        empty_loss: The model's prediction on an empty data point (all features missing).
        model: The model to explain as a callable function.
        loss_function: The loss function to use for the game.
        predictions: The model's predictions on the data.
        data: The background data used to fit the imputer.
        data_shuffled: The background data shuffled column wise.
        n_samples_eval: The number of background samples to use for each evaluation of the value
            function.

    References:
        .. [1] Covert, I., Lundberg, S., Lee, S.-L. (2020). Understanding Global Feature Contributions With Additive Importance Measures. https://arxiv.org/abs/2004.00668
        .. [2] https://iancovert.com/blog/understanding-shap-sage/
    """

    def __init__(
        self,
        *,
        data: np.ndarray,
        y_true: np.ndarray,
        model: Callable[[np.ndarray], np.ndarray],
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        n_expectation_rounds: int = 5000,
        n_samples_eval: int | None = None,
        random_state: int | None = 42,
        verbose: bool = False,
        bins: list[np.ndarray] | None = None,
        conditional_replacement: bool = False,
    ) -> None:
        """Initialize the GlobalExplanation game.

        Args:
            data: The background data used to fit the imputer. Should be a 2d matrix of shape
                ``(n_samples, n_features)``.

            y_true: The true values for the data. Should be a 1d vector of shape ``(n_samples,)``.

            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions. The input should be a 2d matrix of shape
                ``(n_samples, n_features)`` and the output a 1d vector of shape ``(n_samples,)``.

            loss_function: The loss function to use for the game as a callable function that takes the
                true values and the predictions as input and returns the loss.

            n_samples_eval: The number of background samples to use for each evaluation of the value
                function. The number of model evaluations is ``n_samples_eval * n_coalitions``. If
                ``None`` or greater than the number of samples in the background data, all samples
                are used. Defaults to ``None``.

            n_expectation_rounds: The number of random subsets to use for estimating the worth of
                a coalition. More rounds lead to a more accurate estimate of the worth, but also
                increase the computation time. Defaults to ``5000``.

            verbose: A flag to print information of the game. Defaults to ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.

            bins: A list of arrays specifying for each feature, which data points belong to which
                bin for conditional sampling. If ``None``, no conditional sampling is
                performed. Defaults to ``None``.

            conditional_replacement: A flag to indicate whether to use conditional replacement
                when replacing feature values. If ``True``, feature values are replaced with values
                from the same bin. If ``False``, feature values are replaced with random values
                from the entire column. Defaults to ``False``. This option is ignored if ``bins`` is
                ``None``.
        """
        self._rng = np.random.default_rng(random_state)

        # store a copy of the data and y_true
        self.data = copy.deepcopy(data)
        self.y_true = copy.deepcopy(y_true)

        # shuffle the data not column wise:
        shuffled_idx = self._rng.permutation(self.data.shape[0])
        self.data_shuffled = self.data[shuffled_idx]

        # specify the number of samples to evaluate for the coalitions
        if n_samples_eval is None:
            n_samples_eval = self.data_shuffled.shape[0]
        self.n_samples_eval = min(n_samples_eval, self.data_shuffled.shape[0])
        self.n_expectation_rounds = n_expectation_rounds

        # get the model, loss function, and y_true
        self.model = model
        self.loss_function = loss_function

        # if bins are provided, check that conditional_replacement is True
        # data_binned are of shape (n_samples, ) for each feature
        self.data_binned: list[np.ndarray] = bins
        if bins is None and conditional_replacement:
            if verbose:
                print(""
                  "Warning: bins is None, but conditional_replacement is True. Ignoring "
                  "conditional_replacement."
              )
            conditional_replacement = False
        self.conditional_replacement = conditional_replacement
        self.bins = [np.unique(bin) for bin in bins] if bins is not None else []

        # init the base game
        super().__init__(
            data.shape[1],
            normalize=False,
            normalization_value=0.0,
            verbose=verbose,
        )

        self.idx = self._rng.choice(self.data.shape[0], size=self.n_samples_eval, replace=False)

    def value_function(self, coalitions: NDArray[None, bool]) -> NDArray[None, float]:
        """Return the worth of the coalitions for the global explanation game.

        The worth of a coalition in the global explanation game is the performance of the model as
        measured by the loss function on a random subset of the data where the features not part of
        the coalition are replaced by shuffled values from the background data.

        Args:
            coalitions: The coalitions as a one-hot matrix for which the game is to be evaluated.

        Returns:
            The worth of the coalitions as a vector of length `n_coalitions`.

        """
        worth = np.zeros(coalitions.shape[0], dtype=float)
        for i, coalition in enumerate(coalitions):
            worth_coal = 0.0
            row_subset, y_true = self.data.copy(), self.y_true.copy()
            final_predictions = np.zeros(y_true.shape[0], dtype=float)
            for _ in range(self.n_expectation_rounds):
                idx = self._rng.choice(self.data.shape[0], size=self.n_samples_eval, replace=False)
                if not self.conditional_replacement:
                    # replace the features not part of the subset
                    row_subset[:, ~coalition] = self.data_shuffled[idx][:, ~coalition]
                else:
                    self._conditional_replace(row_subset, coalition, self.idx)
                # get the predictions of the model on the subset
                subset_predictions = self.model(row_subset)
                final_predictions += subset_predictions
            final_predictions /= self.n_expectation_rounds
            # get the loss of the model on the subset
            worth_coal += self.loss_function(y_true, final_predictions)
            worth[i] = worth_coal
        return worth

    def _conditional_replace(
        self,
        subset: NDArray[np.float64, np.dtype[np.float64]],
        coalition: NDArray[np.bool_, np.dtype[np.bool_]],
        idx: NDArray[np.int_, np.dtype[np.int_]],
    ) -> None:
        """Replace feature values conditionally based on bins.

        For each feature j that is NOT in the coalition, and for each row r in `subset`,
        we find the bin id of the *original* row `idx[r]` for feature j, then sample a
        donor row from the background data that shares the same bin for feature j, and
        copy that donor's value into `subset[r, j]`.
        """
        # safety checks
        if not self.data_binned or len(self.data_binned) != subset.shape[1]:
            raise ValueError(
                "Conditional replacement requires `bins` for each feature. "
                "Expected len(data_binned) == n_features."
            )

        for j in range(subset.shape[1]):
            if coalition[j]:
                continue  # keep original values for features inside the coalition

            # Bin ids for the selected rows (aligned with `subset` / `idx`)
            row_bin_ids = self.data_binned[j][idx]

            # For each unique bin id among these rows, replace values from same-bin donors
            unique_bin_ids = np.unique(row_bin_ids)
            for bin_id in unique_bin_ids:
                mask = (row_bin_ids == bin_id)  # which rows in `subset` share this bin
                if not np.any(mask):
                    continue

                # Candidate donor indices from the *entire* background set for this feature/bin
                donor_candidates = np.where(self.data_binned[j] == bin_id)[0]
                if donor_candidates.size == 0:
                    raise ValueError(
                        f"No donor candidates found for feature {j} and bin {bin_id}. "
                        "This should never happen."
                    )

                # Sample donors (with replacement) and copy the feature values
                donors = self._rng.choice(donor_candidates, size=mask.sum(), replace=True)
                subset[mask, j] = self.data[donors, j]


class LocalConditionalGame(Game):

    def __init__(
        self,
        x_explain: np.ndarray,
        model: Callable[[np.ndarray], np.ndarray],
        n_expectation_rounds: int = 5000,
        random_state: int | None = 42,
        bins: list[np.ndarray] | None = None,
    ) -> None:
        """Initialize the Local Conditional Game."""

        self._rng = np.random.default_rng(random_state)

        # store a copy of the data and y_true
        self.x_explain = copy.deepcopy(x_explain)

        # get the model, loss function, and y_true
        self.model = model

        self.n_expectation_rounds = n_expectation_rounds

        # if bins are provided, check that conditional_replacement is True
        # data_binned are of shape (n_samples, ) for each feature
        self.data_binned: list[np.ndarray] = bins
        if bins is None:
            raise ValueError("bins must be provided for LocalConditionalGame.")
        self.bins = [np.unique(bin) for bin in bins]

        # init the base game
        super().__init__(
            n_players=x_explain.shape[1],
            normalize=False,
            normalization_value=0.0,
            verbose=False,
        )

    def value_function(self, coalitions: NDArray[None, bool]) -> NDArray[None, float]:
        """Return the worth of the coalitions for the local conditional game.

        The worth of a coalition in the local conditional game is the prediction of the model on
        a random subset of the data point where the features not part of the coalition are replaced
        by values from the same bin.

        Args:
            coalitions: The coalitions as a one-hot matrix for which the game is to be evaluated.

        Returns:
            The worth of the coalitions as a vector of length `n_coalitions`.

        """
        worth = np.zeros(coalitions.shape[0], dtype=float)
        for i, coalition in enumerate(coalitions):
            worth_coal = 0.0
            row_subset = np.tile(self.x_explain, (self.n_expectation_rounds, 1))
            for _ in range(self.n_expectation_rounds):
                self._conditional_replace(row_subset, coalition)
                # get the predictions of the model on the subset
                subset_predictions = self.model(row_subset)
                worth_coal += float(np.mean(subset_predictions))
            worth_coal /= self.n_expectation_rounds
            worth[i] = worth_coal
        return worth

    def _conditional_replace(
        self,
        subset: NDArray[np.float64, np.dtype[np.float64]],
        coalition: NDArray[np.bool_, np.dtype[np.bool_]],
    ) -> None:
        """Replace feature values conditionally based on bins.

        For each feature j that is NOT in the coalition, and for each row r in `subset`,
        we find the bin id of the *original* row `idx[r]` for feature j, then sample a
        donor row from the background data that shares the same bin for feature j, and
        copy that donor's value into `subset[r, j]`.
        """
        # safety checks
        if not self.data_binned or len(self.data_binned) != subset.shape[1]:
            raise ValueError(
                "Conditional replacement requires `bins` for each feature. "
                "Expected len(data_binned) == n_features."
            )

        for j in range(subset.shape[1]):
            if coalition[j]:
                continue
            # Bin id for the explanation point
            row_bin_id = self.data_binned[j][0]
            # Candidate donor indices from the *entire* background set for this feature/bin
            donor_candidates = np.where(self.data_binned[j] == row_bin_id)[0]
            if donor_candidates.size == 0:
                raise ValueError(
                    f"No donor candidates found for feature {j} and bin {row_bin_id}. "
                    "This should never happen."
                )
            # Sample donors (with replacement) and copy the feature values
            donors = self._rng.choice(donor_candidates, size=subset.shape[0], replace=True)
            subset[:, j] = self.x_explain[donors, j]
