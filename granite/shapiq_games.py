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
        n_samples_eval: int | None = None,
        n_samples_empty: int | None = None,
        sampling_rounds: int = 1,
        normalize: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        """Initialize the GlobalExplanation game.

        Args:
            data: The background data used to fit the imputer. Should be a 2d matrix of shape
                ``(n_samples, n_features)``.

            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions. The input should be a 2d matrix of shape
                ``(n_samples, n_features)`` and the output a 1d vector of shape ``(n_samples,)``.

            loss_function: The loss function to use for the game as a callable function that takes the
                true values and the predictions as input and returns the loss.

            n_samples_eval: The number of background samples to use for each evaluation of the value
                function. The number of model evaluations is ``n_samples_eval * n_coalitions``. If
                ``None`` or greater than the number of samples in the background data, all samples
                are used. Defaults to ``None``.

            n_samples_empty: The number of samples to use for the empty subset of features. If
                ``None`` or greater than the number of samples in the background data, all samples
                are used. Defaults to ``None``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print information of the game. Defaults to ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        self._rng = np.random.default_rng(random_state)

        # store a copy of the data and y_true
        self.data = copy.deepcopy(data)
        self.y_true = copy.deepcopy(y_true)

        # shuffle the data not column wise:
        shuffled_idx = self._rng.permutation(self.data.shape[0])
        self.data_shuffled = self.data[shuffled_idx]
        self.sampling_rounds = sampling_rounds

        # specify the number of samples to evaluate for the coalitions
        if n_samples_eval is None:
            n_samples_eval = self.data_shuffled.shape[0]
        self.n_samples_eval = min(n_samples_eval, self.data_shuffled.shape[0])

        # get the model, loss function, and y_true
        self.model = model
        self.loss_function = loss_function

        # get empty prediction
        if n_samples_empty is None:
            n_samples_empty = self.data_shuffled.shape[0]
        n_samples_empty = min(n_samples_empty, self.data_shuffled.shape[0])
        idx = self._rng.choice(self.data_shuffled.shape[0], size=n_samples_empty, replace=False)
        empty_subset = self.data_shuffled[idx]
        empty_predictions = self.model(empty_subset)  # model call
        self.empty_loss: float = self.loss_function(y_true, empty_predictions)

        # init the base game
        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=self.empty_loss,
            verbose=verbose,
        )

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
            if not any(coalition):
                worth[i] = self.empty_loss
                continue
            # get the subset of the data
            for _ in range(self.sampling_rounds):
                idx = self._rng.choice(self.data.shape[0], size=self.n_samples_eval, replace=False)
                row_subset, y_true = self.data[idx].copy(), self.y_true[idx]
                # replace the features not part of the subset
                row_subset[:, ~coalition] = self.data_shuffled[idx][:, ~coalition]
                # get the predictions of the model on the subset
                subset_predictions = self.model(row_subset)
                # get the loss of the model on the subset
                worth_coal += self.loss_function(y_true, subset_predictions)
            worth[i] = worth_coal / self.sampling_rounds
        return worth
