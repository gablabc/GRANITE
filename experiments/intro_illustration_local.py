import copy
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapiq import InteractionValues
from shapiq import AgnosticExplainer, MarginalImputer

from granite.features import Features
from granite.beeswarm import beeswarm_plot
from granite.shapiq_games import LocalConditionalGame
from granite.utils import create_bins_for_data_partition_tree


def create_local_explanations(
    n_local_explanations: int = 5,
    random_state: int = 42,
    sample_size: int = 100,
    expecation_rounds: int = 10,
    n_data: int = 30_000
):
    """Makes a plot illustrating disagreement between different XAI methods."""

    def _make_storage_dict(
        region_name: str,
        x_i: np.ndarray | list,
        phi_i: np.ndarray | list,
        instance_id: int,
        method_name: str,
    ) -> dict[str, float | str]:
        """Turns information into a serializable dictionary."""
        x_i_arr = np.asarray(x_i).flatten()
        x_i_dict = {f"x_{j+1}": float(v) for j, v in enumerate(x_i_arr)}
        phi_i_arr = np.asarray(phi_i).flatten()
        phi_i_dict = {f"phi_{j+1}": float(v) for j, v in enumerate(phi_i_arr)}
        return {
            "region": region_name,
            "instance_id": instance_id,
            "method": method_name,
            **x_i_dict,
            **phi_i_dict,
        }

    def generate_problem(N, seed):
        # Generate input

        np.random.seed(seed)
        X_1 = np.random.normal(0, 1, size=(N,))
        X_2 = 2 * np.random.randint(0, 2, size=(N,)) - 1
        X_3 = 2 * np.random.randint(0, 2, size=(N,)) - 1
        X_4 = np.random.normal(X_3, 1, size=(N,))

        # Model to explain
        def f(X):
            return 3 * X[:, 0] * X[:, 1] + X[:, 2] + 2 * X[:, 3]

        return np.column_stack((X_1, X_2, X_3, X_4)), f

    def pred_diff(game) -> np.ndarray:
        """Computes prediction difference values."""
        explanation = []
        full_coalition = np.ones(game.n_players, dtype=bool)
        full_output = game(full_coalition)
        for i in range(len(features)):
            removed_coalition = full_coalition.copy()
            removed_coalition[i] = False
            removed_output = game(removed_coalition)
            explanation.append(full_output - removed_output)
        return np.asarray(explanation).flatten()

    def pure(game) -> np.ndarray:
        """Computes pure contribution values."""
        explanation = []
        empty_coalition = np.zeros(game.n_players, dtype=bool)
        empty_output = game(empty_coalition)
        for i in range(len(features)):
            added_coalition = empty_coalition.copy()
            added_coalition[i] = True
            added_output = game(added_coalition)
            explanation.append(added_output - empty_output)
        return np.asarray(explanation).flatten()

    # We generate the data and models. The data X is stored in a (N, d) numpy array
    X, model_function = generate_problem(n_data, random_state)
    features = Features(
        X,
        names=[f"x{i}" for i in range(1, 5)],
        types=["num", "num_int", "num_int", "num"]
    )
    print(features.summary())

    # create the regions manually similar to the previous plot
    regions = {
        "Full space": np.array([True] * X.shape[0]),
        "Region $X_2=1$": X[:, 1] == 1,
        "Region $X_3=1$": X[:, 2] == 1,
        "Region $X_2=1 \\wedge X_3=1$": (X[:, 1] == 1) & (X[:, 2] == 1)
    }

    explanations: list[dict[str, float | str]] = []
    for region_name, mask in regions.items():
        print(f"Region: {region_name}")
        rng = np.random.default_rng(random_state)
        X_region = copy.deepcopy(X[mask])
        shuffled_idx = rng.permutation(X_region.shape[0])
        X_region = X_region[shuffled_idx]
        _, binned_features = create_bins_for_data_partition_tree(
            X_region, cat_feature_indices=[1, 2], max_leaf_nodes=2
        )
        split_ratios = [float(np.sum(binned_features[i]) / n_data)  for i in range(len(binned_features))]
        print(f"Instances in Leaf 0: {split_ratios}%")

        for i in tqdm(range(n_local_explanations)):
            x_i = X_region[i:i+1]
            local_marginal_game = MarginalImputer(
                model=model_function,
                data=X_region,
                x=x_i,
                sample_size=sample_size,
                random_state=random_state,
            )

            local_marginal_game.verbose = False
            local_marginal_game.precompute()

            # m-SHAP
            shap = AgnosticExplainer(
                game=local_marginal_game,
                index="SV",
                max_order=1,
            )

            iv_region = shap.explain(budget=2 ** len(features))
            shap = np.asarray([abs(iv_region[(i,)]) for i in range(len(features))])
            explanations.append(_make_storage_dict(
                region_name=region_name, x_i=x_i, phi_i=shap, instance_id=i, method_name="m-SHAP"
            ))

            # pure marginal
            pure_val = pure(local_marginal_game)
            explanations.append(_make_storage_dict(
                region_name=region_name, x_i=x_i, phi_i=pure_val, instance_id=i, method_name="pure-marginal"
            ))

            # c-SHAP and pred-diff with local conditional game
            local_conditional_game = LocalConditionalGame(
                model=model_function,
                x_explain=x_i,
                sample_size=sample_size,
                n_expectation_rounds=expecation_rounds,
                random_state=random_state,
                bins=binned_features,
                data=X_region,
            )
            local_conditional_game.verbose = False
            local_conditional_game.precompute()

            # c-SHAP
            c_shap = AgnosticExplainer(
                game=local_conditional_game,
                index="SV",
                max_order=1,
            )
            iv_region = c_shap.explain(budget=2 ** len(features))
            cshap = np.asarray([abs(iv_region[(i,)]) for i in range(len(features))])
            explanations.append(_make_storage_dict(
                region_name=region_name, x_i=x_i, phi_i=cshap, instance_id=i, method_name="c-SHAP"
            ))

            # pred-diff
            pred_diff_val = pred_diff(local_conditional_game)
            explanations.append(_make_storage_dict(
                region_name=region_name, x_i=x_i, phi_i=pred_diff_val, instance_id=i, method_name="pred-diff"
            ))


    explanations_df = pd.DataFrame(explanations)
    explanations_df.to_csv("intro_illustration_local_explanations.csv", index=False)


def _arr_to_iv(phi: np.ndarray | list) -> InteractionValues:
    """Converts a numpy array or list to an InteractionValues object."""
    return InteractionValues(
        values=np.asarray(phi),
        interaction_lookup={(i,): i for i in range(len(phi))},
        index="SV",
        max_order=1,
        min_order=1,
        n_players=len(phi),
        baseline_value=0,
    )


def _region_name_to_subplot_idx(region_name: str) -> tuple[int, int]:
    """Maps region names to subplot indices."""
    mapping = {
        "Full space": (0, 0),
        "Region $X_2=1$": (0, 1),
        "Region $X_3=1$": (1, 0),
        "Region $X_2=1 \\wedge X_3=1$": (1, 1)
    }
    return mapping[region_name]


def plot_intro_illustration():
    """Plots the intro illustration."""
    data_df = pd.read_csv("intro_illustration_local_explanations.csv")

    region_names = data_df["region"].unique()
    method_names = data_df["method"].unique()

    n_methods = len(method_names)

    # create a new figure with four subplots
    fig, axes = plt.subplots(n_methods * 2, 2, figsize=(15, 10))

    for region_name in region_names:
        print(f"Region: {region_name}")
        for method_i, method_name in enumerate(method_names):
            print(f"Method: {method_name}")
            ax_idx = _region_name_to_subplot_idx(region_name)
            ax_row, ax_col = ax_idx[0] * n_methods + method_i, ax_idx[1]
            ax = axes[ax_row, ax_col]

            # get all explanations for this region and method
            subset_df = data_df[(data_df["region"] == region_name) & (data_df["method"] == method_name)]
            if subset_df.shape[0] == 0:
                continue
            phi_values = subset_df[[f"phi_{i+1}" for i in range(4)]].values

            var_phi = np.var(phi_values, axis=0)
            print(var_phi)

            data = subset_df[[f"x_{i+1}" for i in range(4)]].values
            assert len(data) == len(phi_values)
            ivs = [_arr_to_iv(phi) for phi in phi_values]
            beeswarm_plot(
                interaction_values_list=ivs,
                data=data,
                max_display=ivs[0].n_players,
                feature_names=[f"X_{i+1}" for i in range(4)],
                ax=ax,
                show=False,
                row_height=1,
                show_colormap=False,
                feature_order=[0, 1, 2, 3]
            )
            if method_i == 0:
                ax.set_title(region_name, fontsize=20)
            ax.set_xlabel("")  # remove xaxis label
            ax.set_ylabel(method_name)

    plt.tight_layout()
    plt.savefig(f"intro_illustration_local.pdf")
    plt.show()



if __name__ == "__main__":
    create_local_explanations(
        n_local_explanations=200,
        random_state=42,
        sample_size=10_000,
        expecation_rounds=20,
        n_data=30_000
    )
    plot_intro_illustration()
