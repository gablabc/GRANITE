from typing import List, Tuple, Dict, Optional, Callable
from copy import deepcopy
import numpy as np
from heapq import heappush, heappop

from .features import Features


SYMBOLS = { True :
            {"leq" : "$\,\leq\,$",
            "and_str" : "$)\,\,\land$\,\,(",
            "up" : "$\,>\,$",
            "low" : "$\,<\,$",
            "in_set" : "$\in$"},
            False :
            {"leq" : "<=",
            "and_str" : " & ",
            "up" : ">",
            "low": "<",
            "in_set" : "∈"}
        }



class Node(object):
    """ Node in a Decision Tree """
    def __init__(self, instances_idx: List[int], depth: int, loss: float):
        self.N_samples = len(instances_idx)
        self.depth = depth
        self.loss = loss
        # Placeholders
        self.feature = None
        self.threshold = None
        self.child_left = None
        self.child_right = None
        self.splits = []
        self.objectives = []


    def update(self, feature: int, threshold: float):
        self.feature = feature
        self.threshold = threshold




class FDTree(object):
    """ Train a binary tree to minimize : Loss + alpha |L| """

    def __init__(
            self,
            features: Features,
            max_depth: int=3,
            min_samples_leaf: int=20,
            branching_per_node: int=1,
            alpha: float=0.05,
            save_losses: bool=False
            ):
        """
        Parameters
        ----------
        features : Feature object
            Features along which to split the input space. Features cannot be grouped.
        max_depth : int, default=3
            Maximum depth of FDTrees
        min_samples_leaf : int, default=20
            The minimum number of samples allowed per leaf
        branching_per_node : int, default=1
            At each node, we consider `branching_per_node` different splits candidates.
            A value of `1` corresponds to greedy CART-like optimization, while larger values
            allow to try various splits and return a more optimal solution. Note that the training
            scales as `O(branching_per_node^max_depth)`
        alpha : float, default=0.05
            Objective regularization `Loss + alpha |L|` so splitting nodes increases the loss by
            `alpha` and reductions in `LoA` must be large enough to compensate.
        save_losses: bool, default=False
            Save the loss function for each split value of each internal node. Useful for debugging.
        """
        self.feature_objs = features.feature_objs
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.branching_per_node = branching_per_node
        self.alpha = alpha
        self.save_losses = save_losses
        self.D = len(features)
        self.root = None
        self.n_regions = 0
        self.loss_fn = None
        self.init_loss = None


    def fit(self, X: np.ndarray, loss_fn: Callable[[np.ndarray[int]], float]):
        """
        Fit the FDTree using foreground data and possibly other parameters.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (N, d) of features values to split upon.
        loss_fn: Callable[[np.ndarray[int]], float]
            A callable that takes an array of datum indices and returns a loss.
        """
        self.X = X
        self.N, D = X.shape
        N_range = np.arange(self.N)
        assert callable(loss_fn), "The loss_fn passed to fit must be a callable."
        self.loss_fn = loss_fn
        # The loss is relative to the loss when grouping all datapoints
        self.init_loss = self.loss_fn(N_range) / self.N
        # Start recursive tree growth
        self.root, self.final_objective, self.n_regions = self._tree_builder(
                                                                        N_range,
                                                                        depth=0,
                                                                        node_loss=1.0,
                                                                    )
        self.final_loss = self.final_objective - self.alpha * self.n_regions
        return self


    def _tree_builder(
            self,
            instances_idx: np.ndarray,
            depth: int,
            node_loss: float
            ) -> Tuple[Node, float, int]:
        """ Recursive calls to build the tree

        Parameters
        ----------
        instances_idx : np.ndarray
            `(N,)` array of integers representing the index of the instances at the node.
        depth: int
            The current depth in the tree traversal.
        node_loss: float
            The loss contribution of the current node.

        Returns
        -------
        curr_node: Node
            The current node.
        objective: float
            The objective contribution of the current node i.e. Loss + alpha * n_children
        n_children_leaves: int
            Number of child leaves. If terminal node then return 1.
        """

        # Create a node
        curr_node = Node(instances_idx, depth, node_loss)

        # Stop the tree growth if the maximum depth is attained,
        # or no further split can be justified given the regularization alpha,
        # or any further split will yield leaves with too few samples
        if depth >= self.max_depth or \
                node_loss < self.alpha or \
                len(instances_idx) < 2 * self.min_samples_leaf:

            return curr_node, node_loss + self.alpha, 1

        # Otherwise get a heapq of split candidates
        heapq = self._get_feature_splits_heapq(curr_node, instances_idx)

        # Stop the tree growth if no further splits are possible
        if len(heapq) == 0:
            return curr_node, node_loss + self.alpha, 1

        # If the next split is guaranteed to be a leaf, we do not need to branch
        if depth + 1 == self.max_depth:
            n_branching = 1
        else:
            n_branching = min(len(heapq), self.branching_per_node)
        subobjective_per_branch = np.zeros(n_branching)
        nodes_per_branch = [deepcopy(curr_node) for _ in range(n_branching)]
        n_leaves_per_branch = np.zeros(n_branching, dtype=np.int32)
        for branch in range(n_branching):

            _, feature_split, split_value, loss_left, loss_right = heappop(heapq)

            # Select instances of the chosen feature
            x_i = self.X[instances_idx, feature_split]

            # Update the node
            nodes_per_branch[branch].update(feature_split, split_value)

            # Go left
            nodes_per_branch[branch].child_left, subobjective_left, n_leaves_left = \
                                self._tree_builder(instances_idx[x_i <= split_value],
                                                    depth=depth+1, node_loss=loss_left)
            # Go right
            nodes_per_branch[branch].child_right, subobjective_right, n_leaves_right = \
                                self._tree_builder(instances_idx[x_i > split_value],
                                                    depth=depth+1, node_loss=loss_right)
            n_leaves_per_branch[branch] = n_leaves_left + n_leaves_right
            subobjective_per_branch[branch] = subobjective_left + subobjective_right

        # Identify the best branch from the current node
        best_branch = np.argmin(subobjective_per_branch)

        # The best solution resulting from branching has introduced many leaves. Hence, if the reduction in loss is
        # not sufficient, it is best not to split and define the current node as a leaf
        if node_loss + self.alpha <= subobjective_per_branch[best_branch]:
            return curr_node, node_loss + self.alpha, 1
        else:
            # Branching lead to a better solution and so we go up in the recursion
            return nodes_per_branch[best_branch], subobjective_per_branch[best_branch], n_leaves_per_branch[best_branch]


    def _get_feature_splits_heapq(
            self,
            curr_node: Node,
            instances_idx: np.ndarray
            ):
        """ Compute a heapqueue for splits along each feature """
        heapq = []
        for feature in range(self.D):

            x_i = self.X[instances_idx, feature]
            splits = self._get_split_candidates(x_i, feature)

            # No split possible
            if len(splits) == 0:
                continue

            # Otherwise we optimize the objective
            loss_left = np.zeros(len(splits))
            loss_right = np.zeros(len(splits))
            to_keep = np.zeros((len(splits))).astype(bool)

            # Iterate over all splits
            for i, split in enumerate(splits):
                left = instances_idx[x_i <= split]
                right = instances_idx[x_i > split]
                to_keep[i] = min(len(left), len(right)) >= self.min_samples_leaf
                loss_left[i] = self.loss_fn(left) / (self.init_loss * self.N)
                loss_right[i] = self.loss_fn(right) / (self.init_loss * self.N)

            splits = splits[to_keep]
            loss_left = loss_left[to_keep]
            loss_right = loss_right[to_keep]

            # No split was conducted
            if len(splits) == 0:
                continue

            # Otherwise search for the best split
            loss = loss_right + loss_left
            if self.save_losses:
                curr_node.splits.append(splits)
                curr_node.objectives.append(loss)

            best_split_idx = np.argmin(loss)
            # The heap contains (obj, feature, split_value, obj_left, obj_right)
            heappush(
                    heapq,
                    (
                        loss[best_split_idx],
                        feature,
                        splits[best_split_idx],
                        loss_left[best_split_idx],
                        loss_right[best_split_idx]
                    )
                 )
        return heapq


    def _get_split_candidates(
            self,
            x_i: np.ndarray,
            i: int
            ) -> np.ndarray:
        """ Return a list of split candiates along feature i """

        if self.feature_objs[i].type == "num":
            x_i_unique = np.unique(x_i)
            if len(x_i_unique) < 40:
                splits = np.sort(x_i_unique)[:-1:]
            else:
                splits = np.quantile(x_i, np.arange(1, 40) / 40)
                # It is possible that quantiles equal the last element when there are
                # duplications. Hence we remove those splits to avoid leaves with no data
                splits = splits[~np.isclose(splits, np.max(x_i))]
        elif self.feature_objs[i].type == "sparse_num":
            is_nonzero = np.where(x_i > 0)[0]
            if len(is_nonzero) == 0:
                splits = []
            else:
                x_i_nonzero = x_i[is_nonzero]
                x_i_non_zero_unique = np.unique(x_i_nonzero)
                if len(x_i_non_zero_unique) < 50:
                    splits = np.sort(x_i_non_zero_unique)[::-1]
                else:
                    splits = np.append(0, np.quantile(x_i_nonzero, np.arange(1, 50) / 50))
                # It is possible that quantiles equal the last element when there are
                # duplications. Hence we remove those splits to avoid leaves with no data
                splits = splits[~np.isclose(splits, np.max(x_i))]
        # Integers we take the values directly
        elif self.feature_objs[i].type in ["ordinal", "num_int"]:
            splits = np.sort(np.unique(x_i))[:-1]
        elif self.feature_objs[i].type == "bool":
            x_i = np.unique(x_i)
            if len(x_i) == 1:
                splits = []
            else:
                splits = np.array([0])
        elif ":" in self.feature_objs[i].type:
            raise Exception("Cannot group upon features that are grouped")
        else:
            raise Exception("Only `num`, `sparse_num`, `ordinal`, and `num_int` features can be split")

        return splits


    def print(
            self,
            verbose: bool=False,
            return_string: bool=False
            ) -> Optional[str]:
        """
        Print the FDTree

        Parameters
        ----------
        verbose : bool, default=False
            To be extra verbose.
        return_strong : bool, default=False
            Whether to print the tree or the return the string.
        """
        tree_strings = []
        self.region_idx = 0
        if self.root is None:
            raise Exception("Cannot print a tree before calling `.fit`")
        self._recurse_print_tree_str(self.root, verbose=verbose, tree_strings=tree_strings)
        tree_strings.append(f"Final Loss {self.final_loss:.4f}")
        if return_string:
            return "\n".join(tree_strings)
        else:
            print("\n".join(tree_strings))


    def _recurse_print_tree_str(
            self,
            node: Node,
            verbose: bool=False,
            tree_strings: List[str]=[]
            ):
        if verbose:
            tree_strings.append("|   " * node.depth + f"Loss {node.loss:.4f}")
            tree_strings.append("|   " * node.depth + f"Samples {node.N_samples:d}")
        # Leaf
        if node.child_left is None:
            tree_strings.append("|   " * node.depth + f"Region {self.region_idx}")
            self.region_idx += 1
        # Internal node
        else:
            curr_feature_name = self.feature_objs[node.feature].name
            tree_strings.append("|   " * node.depth + f"If {curr_feature_name} ≤ {node.threshold:.4f}:")
            self._recurse_print_tree_str(node=node.child_left, verbose=verbose, tree_strings=tree_strings)
            tree_strings.append("|   " * node.depth + "else:")
            self._recurse_print_tree_str(node=node.child_right, verbose=verbose, tree_strings=tree_strings)


    def predict(
            self,
            X_new: np.ndarray
            ) -> np.ndarray[np.int32]:
        """
        Compute the region index of each instancec

        Parameters
        ----------
        X_new : (N, d) np.ndarray
            The data to assign to each lead (region) of the FDTree. The ith column of `X_new` must be the
            ith feature in the Features object passed to the constructor.

        Returns
        -------
        regions : (N,) np.ndarray
            The region index of each datum.
        """
        if self.root is None:
            raise Exception("Cannot predict before calling `.fit`")
        regions = np.zeros(X_new.shape[0], dtype=np.int32)
        self.region_idx = 0
        if self.n_regions == 1:
            return regions
        else:
            self._tree_traversal_predict(self.root, np.arange(X_new.shape[0]), X_new, regions)
            return regions


    def _tree_traversal_predict(
            self,
            node: Node,
            instances_idx: np.ndarray,
            X_new: np.ndarray,
            regions: np.ndarray[np.int32]
            ):
        """ Modifies `regions` in-place """

        if node.child_left is None:
            # Label the instances at the leaf
            regions[instances_idx] = self.region_idx
            self.region_idx += 1
        else:
            x_i = X_new[instances_idx, node.feature]
            # Go left
            self._tree_traversal_predict(node.child_left,
                                         instances_idx[x_i <= node.threshold],
                                         X_new, regions)

            # Go right
            self._tree_traversal_predict(node.child_right,
                                         instances_idx[x_i > node.threshold],
                                         X_new, regions)


    def rules(
            self,
            use_latex:bool=False
            ) -> Dict[int, str]:
        """ Return the rule for each leaf """

        if self.root is None:
            raise Exception("Cannot compute rules before calling `.fit`")

        self.region_idx  = 0
        if self.n_regions == 1:
            return "all"
        else:
            rules = {}
            curr_rule = []
            self._tree_traversal_rules(self.root, rules, curr_rule, use_latex)
            return rules


    def _tree_traversal_rules(
            self,
            node: Node,
            rules: Dict[int, str],
            curr_rule: List[str],
            use_latex: bool
            ):
        """ Modifies rules in-place """
        if node.child_left is None:
            if len(curr_rule) > 1:
                # Simplify long rule lists if possible
                curr_rule_copy = self._postprocess_numerical_rules(curr_rule, use_latex)
                curr_rule_copy = self._postprocess_categorical_rules(curr_rule_copy, use_latex)
                if len(curr_rule_copy) > 1:
                    rules[self.region_idx] = "(" + SYMBOLS[use_latex]["and_str"].join(curr_rule_copy) + ")"
                else:
                    rules[self.region_idx] = curr_rule_copy[0]
            else:
                rules[self.region_idx] = curr_rule[0]
            self.region_idx += 1
        else:

            feature_obj = self.feature_objs[node.feature]
            feature_name = feature_obj.name
            feature_type = feature_obj.type

            # Boolean
            if feature_type == "bool":
                assert np.isclose(node.threshold, 0)
                curr_rule.append(f"not {feature_name}")
            # Ordinal
            elif feature_type == "ordinal":
                categories = np.array(feature_obj.cats)
                cats_left = categories[:int(node.threshold)+1]
                if len(cats_left) == 1:
                    curr_rule.append(f"{feature_name}={cats_left[0]}")
                else:
                    curr_rule.append(f"{feature_name} " + SYMBOLS[use_latex]['in_set']
                                     + " {" + ",".join(cats_left)+"}")
            # Numerical
            else:
                curr_rule.append(feature_name + SYMBOLS[use_latex]['leq'] + f"{node.threshold:.2f}")


            # Go left
            self._tree_traversal_rules(node.child_left, rules, curr_rule, use_latex)
            curr_rule.pop()


            # Boolean
            if feature_type == "bool":
                curr_rule.append(f"{feature_name}")
            # Ordinal
            elif feature_type == "ordinal":
                cats_right = categories[int(node.threshold)+1:]
                if len(cats_right) == 1:
                    curr_rule.append(f"{feature_name}={cats_right[0]}")
                else:
                    curr_rule.append(f"{feature_name} " + SYMBOLS[use_latex]['in_set']
                                     + " {" + ",".join(cats_right) +"}")
            # Numerical
            else:
                curr_rule.append(feature_name + SYMBOLS[use_latex]['up'] + f"{node.threshold:.2f}")

            # Go right
            self._tree_traversal_rules(node.child_right, rules, curr_rule, use_latex)
            curr_rule.pop()


    def _postprocess_numerical_rules(
            self,
            curr_rule: List[str],
            use_latex: bool
            ) -> List[str]:
        """
        Simplify numerical rules
        - Remove redundancy x1>3 and x1>5 becomes x1>5
        - Intervals x1>3 and x1<5 becomes 3<x1<5
        """

        curr_rule_copy = deepcopy(curr_rule)
        separators = [SYMBOLS[use_latex]["leq"], SYMBOLS[use_latex]["up"]]
        select_rules_0 = [rule for rule in curr_rule_copy if separators[0] in rule]
        splits_0 = [rule.split(separators[0])+[0] for rule in select_rules_0]
        select_rules_1 = [rule for rule in curr_rule_copy if separators[1] in rule]
        splits_1 = [rule.split(separators[1])+[1] for rule in select_rules_1]
        # There are splits
        if splits_0 or splits_1:
            splits = np.array(splits_0 + splits_1)
            select_features, inv, counts = np.unique(splits[:, 0], return_inverse=True, return_counts=True)
            # There is redundancy
            if np.any(counts >= 2):
                # Iterate over redundant features
                for i in np.where(counts >= 2)[0]:
                    select_feature = select_features[i]
                    idxs = np.where(inv == i)[0]
                    # Remove the redundant rules
                    for idx in idxs:
                        curr_rule_copy.remove(select_feature+ separators[int(splits[idx, 2])]+ splits[idx, 1])
                    # Sort the rules in ascending order of threshold
                    argsort = idxs[np.argsort(splits[idxs, 1].astype(float))]
                    thresholds = splits[argsort, 1]
                    directions = splits[argsort, 2]
                    threshold_left = None
                    threshold_right = None
                    # We go from left to right and define the rule
                    for threshold, direction in zip(thresholds, directions):
                        # We stop at the first leq
                        if direction == '0':
                            threshold_left = threshold
                            break
                        if direction == '1':
                            threshold_right = threshold
                    # print the new rule
                    if threshold_right is None:
                        new_rule = select_feature+SYMBOLS[use_latex]["leq"]+threshold_left
                    elif threshold_left is None:
                        new_rule = select_feature+SYMBOLS[use_latex]["up"]+threshold_right
                    else:
                        new_rule = threshold_right + SYMBOLS[use_latex]["low"] +\
                            select_feature + SYMBOLS[use_latex]["leq"] + threshold_left
                    # Add the new rule
                    curr_rule_copy.append(new_rule)

        return curr_rule_copy



    def _postprocess_categorical_rules(
            self,
            curr_rule: List[str],
            use_latex: bool
            ) -> List[str]:
        """
        Simplify categorical rules
        - x in {0, 1, 2} and x in {1, 2, 3} becomes x in {1, 2}
        """

        curr_rule_copy = deepcopy(curr_rule)
        separator = SYMBOLS[use_latex]["in_set"]
        features_split = [rule.split(separator) for rule in curr_rule_copy if separator in rule]
        # There are splits
        if features_split:
            splits = np.array(features_split)
            select_features, inv, counts = np.unique(splits[:, 0], return_inverse=True, return_counts=True)
            # There is redundancy
            if np.any(counts >= 2):
                # Iterate over redundant features
                for i in np.where(counts >= 2)[0]:
                    select_feature = select_features[i]
                    idxs = np.where(inv == i)[0]
                    new_set = None
                    # Remove the redundant rules
                    for idx in idxs:
                        curr_rule_copy.remove(select_feature + separator + splits[idx, 1])
                        current_set = splits[idx, 1][2:-1].split(",")
                        if new_set is None:
                            new_set = current_set
                        else:
                            new_set = [j for j in current_set if j in new_set]
                    new_rule = select_feature + SYMBOLS[use_latex]["in_set"] + " {" + ",".join(new_set) + "}"
                    # Add the new rule
                    curr_rule_copy.append(new_rule)

        return curr_rule_copy




# class CoE_Tree(FDTree):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def fit(self, X, decomposition):
#         """
#         Fit the Coe_Tree

#         Parameters
#         ----------
#         X : (N, n_features) np.ndarray
#             The data on which to fit the tree. The ith column of `X` must be the ith feature
#             in the Features object passed to the constructor.
#         decomposition : dict{Tuple: np.ndarray}
#             The functional decomposition used to compute the CoE objective.
#             It needs to be anchored with foreground=background i.e.
#             `decomposition[(0,)].shape = (N, N)`.
#         """
#         super().fit(X)

#         assert np.shape(decomposition[(0,)]) == (self.N, self.N), "An Anchored decomposition with foreground=background is needed"
#         self.H_add = get_h_add(decomposition)
#         self.h = decomposition[()]
#         self.loss_factor = 1 / self.h.var() # To have an loss [0, 1]
#         self.n_regions = 0
#         loss = np.mean((self.h - self.H_add.mean(1))**2)
#         # Start recursive tree growth
#         self.root, self.final_objective, self.n_regions = self._tree_builder(np.arange(self.N), depth=0, loss=loss)
#         self.final_loss = self.final_objective - self.alpha * self.n_regions
#         return self


#     def get_loss(self, instances_idx):
#         h = self.h[instances_idx]
#         instances_idx = instances_idx[:, np.newaxis]
#         return np.sum((h - self.H_add[instances_idx, instances_idx.T].mean(-1))**2)

