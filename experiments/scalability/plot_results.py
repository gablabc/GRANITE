import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from granite.plots import setup_pyplot_font

setup_pyplot_font(size=15)

def main():

    results_df = pd.read_csv("runtimes.csv")
    results_df_filtered = results_df[results_df["N"] % 200 == 0]
    N_range = np.arange(100, 1100, 100)
    d_range = np.arange(5, 35, 5)

    # Plot time vs N
    plt.figure()
    sns.lineplot(data=results_df, x="N", y="decomposition_time", hue="d", style="d", markers=True, palette="tab10")
    plt.xticks(ticks=N_range)
    plt.yticks(np.arange(12))
    plt.ylabel("Time to compute R matrices (s)")
    plt.legend(framealpha=1, title="d")
    plt.grid("on")
    plt.savefig(os.path.join("figures", "decomposition_time_N.pdf"), bbox_inches="tight")

    plt.figure()
    sns.lineplot(data=results_df, x="N", y="partition_time", hue="d", style="d", markers=True, palette="tab10")
    plt.xticks(ticks=N_range)
    # plt.yticks(np.arange(0, , 25))
    plt.ylabel("Time to compute partition (s)")
    plt.legend(framealpha=1, title="d")
    plt.grid("on")
    plt.savefig(os.path.join("figures", "partition_time_N.pdf"), bbox_inches="tight")

    # Plot time vs d
    plt.figure()
    sns.lineplot(data=results_df_filtered, x="d", y="decomposition_time", hue="N", style="N", markers=True, palette="tab10")
    plt.xticks(ticks=d_range)
    plt.yticks(np.arange(12))
    plt.ylabel("Time to compute R matrices (s)")
    plt.legend(framealpha=1, title="N")
    plt.grid("on")
    plt.savefig(os.path.join("figures", "decomposition_time_d.pdf"), bbox_inches="tight")

    plt.figure()
    sns.lineplot(data=results_df_filtered, x="d", y="partition_time", hue="N", style="N", markers=True, palette="tab10")
    plt.xticks(ticks=d_range)
    # plt.yticks(np.arange(12))
    plt.ylabel("Time to compute partition (s)")
    plt.legend(framealpha=1, title="N")
    plt.grid("on")
    plt.savefig(os.path.join("figures", "partition_time_d.pdf"), bbox_inches="tight")

    # Plot time vs max-depth
    plt.figure()
    sns.boxplot(data=results_df, x="max-depth", y="partition_time", hue="d", palette="tab10")
    # plt.xticks(ticks=d_range)
    # plt.yticks(np.arange(12))
    plt.ylabel("Time to compute partition (s)")
    plt.legend(framealpha=1, title="d")
    plt.grid("on")
    plt.savefig(os.path.join("figures", "partition_time_max_depth.pdf"), bbox_inches="tight")

if __name__ == "__main__":
    main()
