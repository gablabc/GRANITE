import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapiq import InteractionValues
from typing import Dict, List, Tuple, Literal

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
        "Full space": (1, 0),
        "Region $X_2=1$": (1, 1),
        "Region $X_3=1$": (0, 0),
        "Region $X_2=1 \\wedge X_3=1$": (0, 1),
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


def _title_for_region(region_name: str) -> str:
    """Returns a display title for a given region name."""
    titles = {
        "Full space": "Full space",
        "Region $X_2=1$": "Region: $X_2=1$",
        "Region $X_3=1$": "Region: $X_3=1$",
        "Region $X_2=1 \\wedge X_3=1$": "Region: $X_2=1$ & $X_3=1$",
    }
    return titles.get(region_name, region_name)


def _method_name(method_code: str) -> str:
    """Maps method codes to display names."""
    mapping = {
        "m-Shapley": "marginal Shapley",
        "c-Shapley": "conditional Shapley",
        "c-Full": "PredDiff",
    }
    return mapping.get(method_code, method_code)


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


def _parse_feature_selector(s: int | str, n_features: int) -> int:
    """Accepts 0-based int or 'X_i' (1-based); returns 0-based index."""
    if isinstance(s, int):
        idx = s
    else:
        s = str(s).strip()
        if s.startswith(("X_", "x_")):
            idx = int(s.split("_")[1]) - 1
        else:
            raise ValueError(f"Feature selector must be int or 'X_i', got {s}")
    if not (0 <= idx < n_features):
        raise IndexError(f"Feature index {idx} out of range [0, {n_features-1}]")
    return idx


def _normalize_highlight_map(
    highlight_map: Dict[str, Dict[int | str, str | tuple]] | None, n_features: int
) -> Dict[str, Dict[int, tuple[str, float]]]:
    """
    Convert user mapping to {region: {feat_idx: (color, alpha)}} with 0-based indices.
    Color may be 'moccasin' or ('moccasin', 0.2).
    """
    out: Dict[str, Dict[int, tuple[str, float]]] = {}
    if not highlight_map:
        return out
    for region, per_feat in highlight_map.items():
        out.setdefault(region, {})
        for feat_sel, spec in per_feat.items():
            idx = _parse_feature_selector(feat_sel, n_features)
            if isinstance(spec, tuple) and len(spec) == 2:
                color, alpha = spec
            else:
                color, alpha = spec, 0.12  # gentle default
            out[region][idx] = (color, float(alpha))
    return out


# --- Plot 2: Variance panel (separate function) ---
def plot_bar_panel(
    data_df: pd.DataFrame,
    *,
    feature_count: int = 4,
    figsize: Tuple[int, int] = (14, 10),
    method_colors: Dict[str, str] | None = None,
    metric: Literal["var", "mae"] = "var",  # "var" (default) or "mae"
    savepath: str | None = "intro_illustration_variances.pdf",
    show: bool = True,
    highlight_map: Dict[str, Dict[int | str, str | tuple]] | None = None,  # NEW
) -> None:
    """
    2x2 grid: one subplot per region.
    Horizontal grouped bars per feature (one bar per method), colored via `method_colors`.

    metric:
        - "var": plots Var(phi) across samples
        - "mae": plots mean(|phi|) across samples

    highlight_map:
        Dict[region_name, Dict[feature_selector, color_or_(color, alpha)]]
        - feature_selector: 0-based int or 'X_i' (1-based)
        - color_or_(color, alpha): e.g., 'moccasin' or ('moccasin', 0.2)
    """
    # Normalize/validate metric
    if metric == "mae":
        agg = "mae"
        x_label = "Regional mean of absolute feature effects"
        default_fname = "intro_illustration_phi_mae.pdf"
    elif metric == "var":
        agg = "var"
        x_label = r"Var($\phi$)"
        default_fname = "intro_illustration_variances.pdf"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if savepath is None:
        savepath = default_fname

    region_order, method_names, agg_values = _compute_agg_by_region_method(
        data_df, feature_count=feature_count, agg=agg
    )

    # Colors per method (fallback to tab10 cycle if not provided)
    if method_colors is None:
        from matplotlib import cm

        palette = cm.get_cmap("tab10").colors
        method_colors = {
            m: palette[i % len(palette)] for i, m in enumerate(method_names)
        }
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

    # Normalize highlight map once
    norm_highlights = _normalize_highlight_map(highlight_map, n_features)

    for region_name in region_order:
        ax_row, ax_col = _region_name_to_subplot_idx(region_name)
        ax = axes[ax_row, ax_col]

        # --- NEW: draw background bands for selected features in this region ---
        if region_name in norm_highlights:
            for feat_idx, (color, alpha) in norm_highlights[region_name].items():
                # span the full feature stripe; draw behind bars/grid
                ax.axhspan(
                    feat_idx - 0.5,
                    feat_idx + 0.5,
                    facecolor=color,
                    alpha=alpha,
                    zorder=0,
                    linewidth=0,
                )

        # bars per method grouped by feature
        for m_j, m in enumerate(method_names):
            x_vals = [
                agg_values[region_name][m][feat_idx] for feat_idx in range(n_features)
            ]
            ax.barh(
                y + offsets[m_j],
                x_vals,
                height=bar_height,
                label=_method_name(m) if (ax_row == 0 and ax_col == 0) else None,
                color=method_colors[m],
                zorder=2,  # ensure above spans/grid
                edgecolor="white",
                linewidth=0.75,
            )

        ax.set_title(_title_for_region(region_name), fontsize=14)
        ax.set_ylim(-0.5, n_features - 0.5)
        ax.set_xlim(0, x_max)
        # grid behind bars manually for styling
        # ax.grid(axis="x", color="grey", linestyle="--", linewidth=0.75, zorder=1, alpha=0.7)
        for x_pos in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            ax.axvline(
                x_pos, color="white", linestyle="-", linewidth=1, zorder=1, alpha=0.8
            )
            ax.axvline(
                x_pos, color="lightgrey", linestyle="dotted", linewidth=1, zorder=1
            )
        ax.set_yticks(y)
        ax.set_yticklabels([f"$X_{i+1}$" for i in range(n_features)], fontsize=11)
        if ax_col == 0:
            ax.set_ylabel("Features", fontsize=12)
        if ax_row == 1:
            ax.set_xlabel(x_label, fontsize=12)

    # Shared legend on top
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=method_colors[m]) for m in method_names
    ]
    fig.legend(
        handles,
        [_method_name(m) for m in method_names],
        loc="upper center",
        ncol=len(method_names),
        frameon=False,
        fontsize=13,
    )

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
    plot_mae: bool = False,
    method_selection: List[str] | None = None,
) -> None:
    """Plots the intro beeswarm panel, the variance panel, and the phi violin panel."""
    data_df = pd.read_csv("intro_illustration_local_explanations.csv")

    if method_selection is not None:
        data_df = data_df[data_df["method"].isin(method_selection)]

    if plot_beeswarm:
        plot_intro_beeswarm_panel(
            data_df,
            feature_count=4,
            figsize=figsize,
            savepath="intro_illustration_beeswarm.pdf",
            show=True,
        )

    # shades of orange
    method_color = {
        "m-Shapley": "#FFE569",
        "c-Shapley": "#EBC836",
        "c-Full": "#CAAB00",
    }

    alpha_highlight = 1.0
    highlight_map_example = {
        "Full space": {
            "X_1": ("#3F88C5", alpha_highlight),  # highlight feature X2 softly
            "X_2": ("#3F88C5", alpha_highlight),  # highlight feature X2 softly
            "X_3": ("#57AA99", alpha_highlight),  # highlight feature X2 softly
            "X_4": ("#57AA99", alpha_highlight),  # highlight feature X2 softly
        },
        "Region $X_2=1$": {
            "X_3": ("#57AA99", alpha_highlight),  # highlight feature X2 softly
            "X_4": ("#57AA99", alpha_highlight),  # highlight feature X2 softly
        },
        "Region $X_3=1$": {
            "X_1": ("#3F88C5", alpha_highlight),  # highlight feature X2 softly
            "X_2": ("#3F88C5", alpha_highlight),  # highlight feature X2 softly
        },
    }

    if plot_mae:
        plot_bar_panel(
            data_df,
            feature_count=4,
            figsize=figsize,
            metric="mae",
            savepath="intro_illustration_phi_mae.pdf",
            show=True,
            method_colors=method_color,
            highlight_map=highlight_map_example,
        )


if __name__ == "__main__":

    methods = ["m-Shapley", "c-Shapley", "c-Full"]
    plot_intro_illustration(figsize=(9, 7), plot_mae=True, method_selection=methods)
