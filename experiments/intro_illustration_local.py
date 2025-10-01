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


def create_local_explanations(n_local_explanations: int = 5, random_state: int = 42):
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
        pfi_values = []
        full = game(tuple(range(len(features))))
        for i in range(len(features)):
            removed_score = game(tuple(j for j in range(len(features)) if j != i))
            pfi_values.append(full - removed_score)
        return np.asarray(pfi_values).flatten()

    # We generate the data and models. The data X is stored in a (N, d) numpy array
    X, model_function = generate_problem(30_000, 42)
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

        print("Means in region:", np.mean(X_region, axis=0))

        for i in tqdm(range(n_local_explanations)):
            x_i = X_region[i:i+1]
            local_marginal_game = MarginalImputer(
                model=model_function,
                data=X_region,
                x=x_i,
                sample_size=250,
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

            _, binned_features = create_bins_for_data_partition_tree(
                X_region, cat_feature_indices=[1, 2], max_leaf_nodes=3
            )
            local_conditional_game = LocalConditionalGame(
                model=model_function,
                x_explain=x_i,
                n_expectation_rounds=250,
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
            pred_diff = pred_diff(local_conditional_game)
            explanations.append(_make_storage_dict(
                region_name=region_name, x_i=x_i, phi_i=pred_diff, instance_id=i, method_name="pred-diff"
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

    for method_i, method_name in enumerate(method_names):

        # create a new figure with four subplots
        fig, axes = plt.subplots(6, 2, figsize=(12, 10))

        for region_name in region_names:
            ax_idx = _region_name_to_subplot_idx(region_name)
            ax_row, ax_col = ax_idx[0] * 3 + method_i, ax_idx[1]
            ax = axes[ax_row, ax_col]

            # get all explanations for this region and method
            subset_df = data_df[(data_df["region"] == region_name) & (data_df["method"] == method_name)]
            if subset_df.shape[0] == 0:
                continue
            phi_values = subset_df[[f"phi_{i+1}" for i in range(4)]].values
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
                show_colormap=False
            )
            if method_i == 0:
                ax.set_title(region_name, fontsize=20)
            ax.set_xlabel("")  # remove xaxis label
            ax.set_ylabel(method_name)

        plt.tight_layout()
        plt.savefig(f"intro_illustration_local_{method_name.replace(' ', '_')}.pdf")
        plt.show()



if __name__ == "__main__":
    create_local_explanations(n_local_explanations=20, random_state=42)
    plot_intro_illustration()
