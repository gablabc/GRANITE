import pandas as pd
import numpy as np
from time import time
from typing import Any

from utils import generate_data
from granite.decompositions import get_components_brute_force
from granite.fd_trees import FDTree
from granite.losses import get_marginal_pure_vs_full_loss_fn



def run_benchmark(N: int, d: int, max_depth: int, random_state: int=42) -> dict[str, Any]:
    print(f"N: {N}, d: {d}, max_depth: {max_depth}, random_state: {random_state}")

    np.random.seed(random_state)
    X, features, model = generate_data(N, d, max_depth, random_state)
    # model.fd_tree.print()

    # Computing the functional decomposition
    start = time()
    decomposition = get_components_brute_force(model, X, X, features)
    decomposition_time = time() - start

    # Computing the partition
    fd_tree = FDTree(features, max_depth)
    loss_fn = get_marginal_pure_vs_full_loss_fn(decomposition, U=[(i,) for i in range(d)])
    start = time()
    fd_tree.fit(X, loss_fn=loss_fn)
    partition_time = time() - start

    results = {
            "N": N,
            "d": d,
            "max-depth": max_depth,
            "random_state": random_state,
            "decomposition_time": decomposition_time,
            "partition_time": partition_time,
    }
    return results


def main():
    results_df = []
    for N in range(100, 1100, 100):
        for d in range(5, 35, 5):
            for max_depth in [1, 2, 3]:
                for random_state in range(5):
                    results_df.append(run_benchmark(N, d, max_depth, random_state))
    results_df = pd.DataFrame(results_df)
    results_df.to_csv("runtimes.csv")



if __name__ == "__main__":
    main()
