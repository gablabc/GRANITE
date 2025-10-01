import copy

import matplotlib.pyplot as plt
import numpy as np
from shapiq import AgnosticExplainer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from granite.features import Features
from granite.shapiq_games import GlobalRiskGame
from granite.utils import create_bins_for_data_partition_tree
from shapiq.utils.saving import lookup_and_values_to_dict


def plot_disagreement_xai_methods():
    """Makes a plot illustrating disagreement between different XAI methods."""

    def _print_explanation(expl: dict[str, dict[str, np.ndarray]]) -> None:
        """Helper function to print the explanations in a readable format."""
        for region, methods in expl.items():
            print(f"Region: {region}")
            for method, values in methods.items():
                print(f"  {method}: {values}")
            print()

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

    def pfi(game) -> np.ndarray:
        pfi_values = []
        full = game(tuple(range(len(features))))
        for i in range(len(features)):
            removed_score = game(tuple(j for j in range(len(features)) if j != i))
            pfi_values.append(removed_score - full)
        return np.asarray(pfi_values).flatten()

    # We generate the data and models. The data X is stored in a (N, d) numpy array
    X, model_function = generate_problem(20_000, 42)
    features = Features(
        X,
        names=[f"x{i}" for i in range(1, 5)],
        types=["num", "num_int", "num_int", "num"]
    )
    print(features.summary())

    # create the regions manually similar to the previous plot
    # TODO: regions
    regions = {
        "Full space": np.array([True] * X.shape[0]),
        "Region $X_2=1$": X[:, 1] == 1,
        "Region $X_3=1$": X[:, 2] == 1,
        "Region $X_2=1 \\wedge X_3=1$": (X[:, 1] == 1) & (X[:, 2] == 1)
    }

    explanations: dict[str, dict[str, np.ndarray]] = {}
    for region_name, mask in regions.items():
        print(f"Region: {region_name}")
        explanations[region_name] = {}
        X_region = copy.deepcopy(X[mask])
        print("Means in region:", np.mean(X_region, axis=0))

        marginal_game = GlobalRiskGame(
            data=X_region.copy(),
            y_true=model_function(X_region),
            model=model_function,
            loss_function=mean_squared_error,
            n_expectation_rounds=5000,
        )
        marginal_game.verbose = True
        marginal_game.precompute()
        dict_thing = lookup_and_values_to_dict(marginal_game.coalition_lookup, marginal_game.value_storage)
        for key, item in dict_thing.items():
            print(f"{key}: {float(item):.4f}")

        # SAGE
        explainer_sage = AgnosticExplainer(
            game=marginal_game,
            index="SV",
            max_order=1,
        )
        iv_region = explainer_sage.explain(budget=2 ** len(features))
        sage_values = np.asarray([abs(iv_region[(i,)]) for i in range(len(features))])
        explanations[region_name]["SAGE"] = sage_values

        # PFI (PFI is full minus removed one at a time)
        explanations[region_name]["PFI"] = pfi(marginal_game)

        # c-PFI (conditional PFI)

        bins, binned_features = create_bins_for_data_partition_tree(
            X_region, cat_feature_indices=[1, 2], max_leaf_nodes=10
        )

        conditional_game = GlobalRiskGame(
            data=X_region,
            y_true=model_function(X_region),
            model=model_function,
            loss_function=mean_squared_error,
            n_expectation_rounds=5000,
            bins=binned_features,
            conditional_replacement=True
        )

        explanations[region_name]["c-PFI"] = pfi(conditional_game)

    # Print explanations
    _print_explanation(explanations)

    # --- Plot: one subplot per region; each shows grouped horizontal bars per feature ---
    features_labels = [r"$X_1$", r"$X_2$", r"$X_3$", r"$X_4$"]
    methods = ["SAGE", "PFI", "c-PFI"]  # keep a stable order
    method_colors = {
        "SAGE": "#3F88C5",
        "PFI": "#57AA99",
        "c-PFI": "#E9724C",
        "Pure": "#C3423F",
    }

    n_regions = len(explanations)
    n_features = len(features_labels)
    n_methods = len(methods)

    # Create a 2x2 grid (you currently have 4 regions); extendable if you add more later
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    # Bar geometry (grouped horizontal bars)
    base_y = np.arange(n_features)
    total_height = 0.8  # total band height per feature
    bar_height = total_height / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_height

    # consistent axis limits across regions (optional; computed from all values)
    all_vals = []
    for reg_vals in explanations.values():
        for m in methods:
            all_vals.extend(np.asarray(reg_vals[m]).ravel())
    x_max = float(np.max(np.abs(all_vals))) if len(all_vals) else 1.0
    x_pad = 0.05 * x_max
    x_left, x_right = -x_pad, x_max + x_pad  # allow tiny padding on the left

    for ax_i, (region_name, method_dict) in enumerate(explanations.items()):
        ax = axes[ax_i]

        # plot each method as an offset horizontal bar
        for j, m in enumerate(methods):
            vals = np.asarray(method_dict[m]).ravel()
            ax.barh(
                base_y + offsets[j],
                vals,
                height=bar_height * 0.95,
                label=m if ax_i == 0 else None,  # only label once for shared legend
                color=method_colors.get(m, None),
                edgecolor="none",
            )

        # aesthetics
        ax.set_title(region_name)
        ax.set_yticks(base_y)
        ax.set_yticklabels(features_labels)
        ax.invert_yaxis()  # X1 at top
        ax.set_xlabel("Importance")
        ax.set_xlim(x_left, x_right)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # hide any unused axes (in case regions != 4)
    for k in range(ax_i + 1, len(axes)):
        axes[k].set_visible(False)

    # single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=len(methods), frameon=False,
                   bbox_to_anchor=(0.5, -0.01))

    plt.show()


if __name__ == "__main__":
    plot_disagreement_xai_methods()
