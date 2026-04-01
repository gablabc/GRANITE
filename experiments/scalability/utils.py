import numpy as np
from typing import Tuple, Callable

from granite.fd_trees import Node, FDTree
from granite.features import Features


def grow_random_tree(
    X: np.ndarray, tree: FDTree, instances_idx: np.ndarray, current_depth: int
):

    # Create a node
    curr_node = Node(instances_idx, current_depth, 0)

    if current_depth >= tree.max_depth:
        return curr_node

    # Choose a random feature for split
    d = X.shape[1]
    feature_to_split = np.random.choice(range(d), 1)[0]

    # Select instances of the chosen feature
    x_i = X[instances_idx, feature_to_split]
    split_value = np.median(x_i)

    # Update the node
    curr_node.update(feature_to_split, split_value)

    # Go left
    curr_node.child_left = grow_random_tree(
        X, tree, instances_idx[x_i <= split_value], current_depth=current_depth + 1
    )
    # Go right
    curr_node.child_right = grow_random_tree(
        X, tree, instances_idx[x_i > split_value], current_depth=current_depth + 1
    )

    return curr_node


class PiecewiseLinearModel(object):
    def __init__(self, fd_tree: FDTree, d: int):
        self.fd_tree = fd_tree
        self.n_regions = 2**fd_tree.max_depth
        self.weights = np.random.poisson(size=(self.n_regions, d))

    def __call__(self, X):
        regions = self.fd_tree.predict(X)
        output = np.zeros(X.shape[0])
        for r in range(self.n_regions):
            in_region = regions == r
            output[in_region] = np.sum(self.weights[r] * X[in_region], axis=1)
        return output


def generate_data(
    N: int, d: int, max_depth: int, random_seed: int = 42
) -> Tuple[np.ndarray, Features, Callable]:
    """Generate data from a isotropic gaussian and a random piece-wise linear model"""

    X = np.random.normal(0, 1, size=(N, d))
    features = Features(X, names=[f"x{i}" for i in range(d)], types=["num"] * d)
    tree = FDTree(features, max_depth)
    tree.root = grow_random_tree(X, tree, np.arange(N), 0)
    tree.final_loss = 0
    model = PiecewiseLinearModel(tree, d)
    return X, features, model
