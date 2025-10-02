import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapiq import InteractionValues
from typing import Dict, List, Tuple

from granite.beeswarm import beeswarm_plot


# --- converter ---
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

# --- Region layout mapping (unchanged) ---
def _region_name_to_subplot_idx(region_name: str) -> Tuple[int, int]:
    """Maps region names to subplot indices."""
    mapping = {
        "Full space": (0, 0),
        "Region $X_2=1$": (0, 1),
        "Region $X_3=1$": (1, 0),
        "Region $X_2=1 \\wedge X_3=1$": (1, 1),
    }
    return mapping[region_name]


# --- Shared helpers ---
def _default_region_order() -> List[str]:
    """Stable region order aligned with the subplot mapping."""
    return [
        "Full space",
        "Region $X_2=1$",
        "Region $X_3=1$",
        "Region $X_2=1 \\wedge X_3=1$",
    ]


def _compute_agg_by_region_method(
    data_df: pd.DataFrame,
    *,
    feature_count: int = 4,
    region_order: List[str] | None = None,
    method_names: List[str] | None = None,
    agg: str = "var",  # "var" or "mae"
) -> Tuple[List[str], List[str], Dict[str, Dict[str, np.ndarray]]]:
    """
    Compute per-feature aggregates of phi across samples for each (region, method).

    agg:
        - "var": np.var across samples per feature
        - "mae": mean(abs(phi)) across samples per feature
    """
    if region_order is None:
        region_order = _default_region_order()
    if method_names is None:
        method_names = list(data_df["method"].unique())

    out: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in region_order}
    phi_cols = [f"phi_{i+1}" for i in range(feature_count)]

    for region_name in region_order:
        for method_name in method_names:
            subset_df = data_df[
                (data_df["region"] == region_name) & (data_df["method"] == method_name)
            ]
            if subset_df.empty:
                out[region_name][method_name] = np.zeros(feature_count, dtype=float)
                continue

            phi_values = subset_df[phi_cols].to_numpy()  # (n_samples, feature_count)
            if agg == "mae":
                out[region_name][method_name] = np.mean(np.abs(phi_values), axis=0)
            else:  # default: variance
                out[region_name][method_name] = np.var(phi_values, axis=0)

    return region_order, method_names, out


# --- Plot 1: Beeswarm panel (uses your existing utilities) ---
def plot_intro_beeswarm_panel(
    data_df: pd.DataFrame,
    *,
    feature_count: int = 4,
    figsize: Tuple[int, int] = (15, 10),
    savepath: str | None = "intro_illustration_beeswarm.pdf",
    show: bool = True,
) -> None:
    """Recreates your beeswarm grid using the region layout and method order in the data."""
    region_order = _default_region_order()
    method_names = list(data_df["method"].unique())
    n_methods = len(method_names)

    fig, axes = plt.subplots(n_methods * 2, 2, figsize=figsize)

    for region_name in region_order:
        for method_i, method_name in enumerate(method_names):
            ax_idx = _region_name_to_subplot_idx(region_name)
            ax_row, ax_col = ax_idx[0] * n_methods + method_i, ax_idx[1]
            ax = axes[ax_row, ax_col]

            subset_df = data_df[
                (data_df["region"] == region_name) & (data_df["method"] == method_name)
            ]
            if subset_df.empty:
                ax.set_visible(False)
                continue

            phi_cols = [f"phi_{i+1}" for i in range(feature_count)]
            x_cols = [f"x_{i+1}" for i in range(feature_count)]
            phi_values = subset_df[phi_cols].to_numpy()
            data = subset_df[x_cols].to_numpy()

            # your external utilities
            ivs = [_arr_to_iv(phi) for phi in phi_values]
            beeswarm_plot(
                interaction_values_list=ivs,
                data=data,
                max_display=ivs[0].n_players,
                feature_names=[f"X_{i+1}" for i in range(feature_count)],
                ax=ax,
                show=False,
                row_height=1,
                show_colormap=False,
                feature_order=list(range(feature_count)),
            )

            if method_i == 0:
                ax.set_title(region_name, fontsize=20)
            ax.set_xlabel("")
            ax.set_ylabel(method_name)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close()


# --- Plot 2: Variance panel (separate function) ---
def plot_variance_panel(
    data_df: pd.DataFrame,
    *,
    feature_count: int = 4,
    figsize: Tuple[int, int] = (14, 10),
    method_colors: Dict[str, str] | None = None,
    metric: str = "var",  # "var" (default) or "mae"
    savepath: str | None = "intro_illustration_variances.pdf",
    show: bool = True,
) -> None:
    """
    2x2 grid: one subplot per region.
    Horizontal grouped bars per feature (one bar per method), colored via `method_colors`.

    metric:
        - "var": plots Var(phi) across samples
        - "mae": plots mean(|phi|) across samples
    """
    # Normalize/validate metric
    metric_norm = metric.strip().lower()
    if metric_norm in {"mae", "mean_abs", "mean_absolute", "mean_absolute_error"}:
        agg = "mae"
        x_label = r"$\mathbb{E}[\,|\phi|\,]$"
        default_fname = "intro_illustration_phi_mae.pdf"
    else:
        agg = "var"
        x_label = r"Var($\phi$)"
        default_fname = "intro_illustration_variances.pdf"

    if savepath is None:
        savepath = default_fname

    region_order, method_names, agg_values = _compute_agg_by_region_method(
        data_df, feature_count=feature_count, agg=agg
    )

    # Colors per method (fallback to tab10 cycle if not provided)
    if method_colors is None:
        from matplotlib import cm
        palette = cm.get_cmap("tab10").colors
        method_colors = {m: palette[i % len(palette)] for i, m in enumerate(method_names)}
    else:
        from matplotlib import cm
        palette = cm.get_cmap("tab10").colors
        for i, m in enumerate(method_names):
            if m not in method_colors:
                method_colors[m] = palette[i % len(palette)]

    n_features = feature_count
    n_methods = len(method_names)

    # Layout
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = np.array(axes)

    # y positions for feature groups
    y = np.arange(n_features)
    total_height = 0.8
    bar_height = total_height / max(1, n_methods)
    offsets = (-total_height / 2) + (np.arange(n_methods) + 0.5) * bar_height

    # Global x limit across all regions/methods/features
    global_max = 0.0
    for region in region_order:
        for feat_idx in range(n_features):
            vals = [agg_values[region][m][feat_idx] for m in method_names]
            if vals:
                global_max = max(global_max, max(vals))
    x_max = global_max * 1.08 if global_max > 0 else 1.0

    for region_name in region_order:
        ax_row, ax_col = _region_name_to_subplot_idx(region_name)
        ax = axes[ax_row, ax_col]

        # bars per method grouped by feature
        for m_j, m in enumerate(method_names):
            x_vals = [agg_values[region_name][m][feat_idx] for feat_idx in range(n_features)]
            ax.barh(
                y + offsets[m_j],
                x_vals,
                height=bar_height,
                label=m if (ax_row == 0 and ax_col == 0) else None,
                color=method_colors[m],
            )

        ax.set_title(region_name, fontsize=14)
        ax.set_ylim(-0.5, n_features - 0.5)
        ax.set_xlim(0, x_max)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"$X_{i+1}$" for i in range(n_features)], fontsize=11)
        if ax_col == 0:
            ax.set_ylabel("Features")
        if ax_row == 1:
            ax.set_xlabel(x_label)

    # Shared legend on top
    handles = [plt.Rectangle((0, 0), 1, 1, color=method_colors[m]) for m in method_names]
    fig.legend(handles, method_names, loc="upper center", ncol=len(method_names),
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_phi_violins_panel(
    data_df: pd.DataFrame,
    *,
    feature_count: int = 4,
    figsize: Tuple[int, int] = (14, 10),
    method_colors: Dict[str, str] | None = None,
    savepath: str | None = "intro_illustration_phi_violins.pdf",
    show: bool = True,
) -> None:
    """
    2x2 grid: one subplot per region.
    Within each subplot, features X1..Xk on the y-axis; for each feature, plot
    grouped horizontal *violin* plots (one violin per method) colored via `method_colors`.
    """
    region_order = _default_region_order()
    method_names = list(data_df["method"].unique())

    # Colors per method (fallback to tab10 cycle if not provided)
    if method_colors is None:
        from matplotlib import cm
        palette = cm.get_cmap("tab10").colors
        method_colors = {m: palette[i % len(palette)] for i, m in enumerate(method_names)}
    else:
        from matplotlib import cm
        palette = cm.get_cmap("tab10").colors
        for i, m in enumerate(method_names):
            if m not in method_colors:
                method_colors[m] = palette[i % len(palette)]

    n_features = feature_count
    n_methods = len(method_names)

    # Pre-extract phi arrays per (region, method): shape (n_samples, feature_count)
    phi_cols = [f"phi_{i+1}" for i in range(feature_count)]
    phi_data: Dict[str, Dict[str, np.ndarray]] = {r: {} for r in region_order}
    for region_name in region_order:
        for method_name in method_names:
            subset_df = data_df[
                (data_df["region"] == region_name) & (data_df["method"] == method_name)
            ]
            if subset_df.empty:
                phi_data[region_name][method_name] = np.empty((0, feature_count))
            else:
                phi_data[region_name][method_name] = subset_df[phi_cols].to_numpy()

    # Compute symmetric x-limits across all violins
    global_abs_max = 0.0
    for region_name in region_order:
        for method_name in method_names:
            arr = phi_data[region_name][method_name]
            if arr.size:
                global_abs_max = max(global_abs_max, np.nanmax(np.abs(arr)))
    x_lim = global_abs_max * 1.1 if global_abs_max > 0 else 1.0

    # Layout: 2x2 fixed by mapping function
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    axes = np.array(axes)

    # y positions for feature groups
    y = np.arange(n_features)  # 0..k-1
    total_height = 0.8
    v_height = total_height / max(1, n_methods)
    offsets = (-total_height / 2) + (np.arange(n_methods) + 0.5) * v_height

    for region_name in region_order:
        ax_row, ax_col = _region_name_to_subplot_idx(region_name)
        ax = axes[ax_row, ax_col]

        # For each method, draw a violin per feature at y+offset
        for m_j, m in enumerate(method_names):
            arr = phi_data[region_name][m]  # (n_samples, n_features)
            # If no samples, skip
            if arr.size == 0:
                continue
            for feat_idx in range(n_features):
                data_1d = arr[:, feat_idx]
                # Skip if all-NaN or empty
                if data_1d.size == 0 or np.all(~np.isfinite(data_1d)):
                    continue
                v = ax.violinplot(
                    [data_1d],  # expects a sequence
                    positions=[y[feat_idx] + offsets[m_j]],
                    vert=False,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    widths=v_height * 0.95,
                )
                # Color the body
                for body in v["bodies"]:
                    body.set_facecolor(method_colors[m])
                    body.set_edgecolor("black")
                    body.set_linewidth(0.6)
                    body.set_alpha(0.9)

        # Styling
        ax.set_title(region_name, fontsize=14)
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-0.5, n_features - 0.5)
        ax.axvline(0.0, color="0.3", lw=0.8, alpha=0.6)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        ax.set_yticks(y)
        ax.set_yticklabels([f"$X_{i+1}$" for i in range(n_features)], fontsize=11)
        if ax_col == 0:
            ax.set_ylabel("Features")
        if ax_row == 1:
            ax.set_xlabel(r"$\phi$")

    # Shared legend (top)
    handles = [plt.Rectangle((0, 0), 1, 1, color=method_colors[m]) for m in method_names]
    fig.legend(handles, method_names, loc="upper center", ncol=len(method_names),
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# --- Convenience wrapper if you still want a single entry point ---
def plot_intro_illustration(
    figsize=(15, 10),
    plot_beeswarm: bool = False,
    plot_variance: bool = False,
    plot_phi_violins: bool = False,
    plot_mae: bool = False,
) -> None:
    """Plots the intro beeswarm panel, the variance panel, and the phi violin panel."""
    data_df = pd.read_csv("intro_illustration_local_explanations.csv")

    if plot_beeswarm:
        plot_intro_beeswarm_panel(
            data_df,
            feature_count=4,
            figsize=figsize,
            savepath="intro_illustration_beeswarm.pdf",
            show=True,
        )

    if plot_variance:
        plot_variance_panel(
            data_df,
            feature_count=4,
            figsize=figsize,
            savepath="intro_illustration_variances.pdf",
            show=True,
        )

    if plot_mae:
        plot_variance_panel(
            data_df,
            feature_count=4,
            figsize=figsize,
            metric="mae",
            savepath="intro_illustration_phi_mae.pdf",
            show=True,
        )

    if plot_phi_violins:
        plot_phi_violins_panel(
            data_df,
            feature_count=4,
            figsize=figsize,
            savepath="intro_illustration_phi_violins.pdf",
            show=True,
        )


if __name__ == '__main__':
    #plot_intro_illustration(figsize=(20, 20), plot_beeswarm=True, plot_variance=False)
    # plot_intro_illustration(figsize=(9, 7), plot_variance=True)
    plot_intro_illustration(figsize=(9, 7), plot_mae=True)
    #plot_intro_illustration(figsize=(9, 7), plot_phi_violins=True)