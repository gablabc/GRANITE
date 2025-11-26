import os
from typing import Any, Literal
import numpy as np
from joblib import load, dump

import json
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


from granite.data import get_data_bike, get_data_kin8nm, Features
from granite.decompositions import get_components_tree, get_components_brute_force
from granite.experiments import get_marginal_pure_vs_full_loss_fn, get_marginal_vs_conditional_pure_loss_fn
from granite.experiments import get_marginal_pure_vs_full_risk_loss_fn, get_marginal_vs_condition_full_risk_loss_fn
from granite.utils import create_bins_for_data, create_bins_for_data_partition_tree, pointwise_squared_risk



DATASETS = {
    "bike": get_data_bike,
    "kin8nm": get_data_kin8nm,
}

TASKS = {
    "bike": "regression",
    "kin8nm": "regression",
}

CAT_FEATURES_INDICES = {
    "bike" : [0, 1, 3, 4, 5, 6],
    "kin8nm" : [],
}


##### Dataclasses for parsing arguments ####

@dataclass
class Granite_Config:
    loss_fn: Literal['PDP_vs_ICE', 'PureRisk_vs_FullRisk', 'PDP_vs_Mplot', 'PFI_vs_CFI']
    max_depth: int
    alpha: float
    train_background_size: int = 1000
    test_background_size: int = 2000



@dataclass
class Search_Config:
    n_splits: int = 5  # Number of train/valid splits
    split_type: str = "K-Fold" # Type of cross-valid "Shuffle" "K-fold"
    n_repetitions: int = 20  # Number of hyper-param candidates to evaluate
    n_jobs: int = 1          # Number of parallel processes to run jobs

############################## Data Utilities ##############################


def setup_data(name):
    X, y, features = DATASETS[name]()
    task = TASKS[name]

    return X, y, features, task

def subsample_data(x: np.ndarray, y: np.ndarray, subsample_size: int, random_state: 42):
    np.random.seed(random_state)
    N = x.shape[0]
    if subsample_size > N:
        return x, y
    idx_choose = np.random.choice(range(N), subsample_size, replace=False)
    return x[idx_choose], y[idx_choose]

# Custom train/test split for reproducability (random_state is always 42 !!!)
def get_train_test_split(X, y, task, random_state=42):
    ratio = 0.2
    if task == "regression":
        return train_test_split(X, y, test_size=ratio, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=ratio, random_state=random_state, stratify=y)


def get_scoring(task):
    if task == "regression":
        scoring = "neg_root_mean_squared_error"
    else:
        scoring = "accuracy"
    return scoring


############################## Model Utilities ##############################



MODELS = {
         "rf" : {
             "regression": RandomForestRegressor(),
             "classification": RandomForestClassifier()
         },
         "gbt" : {
             "regression": HistGradientBoostingRegressor(),
             "classification": HistGradientBoostingClassifier()
         },
         "mlp" : {
             "regression": MLPRegressor(max_iter=1000),
             "classification": MLPClassifier(max_iter=1000)
         }
        }


def get_hp_grid(dataset: str, model_name: str) -> dict[str, Any]:
    filename = os.path.join(
        "models", "sweeps", f"{model_name}_{dataset}_grid.json"
    )
    def to_eval(string):
        if type(string) is str:
            split = string.split("_")
            if len(split) == 2:
                return split[1]
            else:
                return None
        else:
            return None

    hp_dict = json.load(open(filename, "r"))
    for key, value in hp_dict.items():
        # Iterate over list
        if type(value) is list:
            for i, element in enumerate(value):
                str_to_eval = to_eval(element)
                if str_to_eval is not None:
                    value[i] = eval(str_to_eval)
        # Must be evaluated
        if type(value) is str:
            str_to_eval = to_eval(value)
            if str_to_eval is not None:
                hp_dict[key] = eval(str_to_eval)
    return hp_dict



def get_cross_validator(k: int, task: str, random_state: int, split_type: str):
    # Train / Test split and cross-validator. Dont look at the test yet...
    if task == "regression":
        if split_type == "Shuffle":
            cross_validator = ShuffleSplit(n_splits=k, test_size=0.1, random_state=random_state)
        elif split_type == "K-Fold":
            cross_validator = KFold(n_splits=k, shuffle=True, random_state=random_state)
        else:
            raise ValueError("Wrong type of cross-validator")

    # Binary Classification
    else:
        if split_type == "Shuffle":
            cross_validator = StratifiedShuffleSplit(n_splits=k, test_size=0.1, random_state=random_state)
        elif split_type == "K-Fold":
            cross_validator = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        else:
            raise ValueError("Wrong type of cross-validator")

    return cross_validator



def load_model(dataset: str, model_name: str, random_state: int):
    # Random state used for fitting
    state = str(random_state)
    file_path = os.path.join("models", dataset, model_name+"_"+str(state))

    # Pickle model
    model = load(os.path.join(file_path, "model.joblib"))
    perf = pd.read_csv(os.path.join(file_path, "performance.csv")).to_numpy()
    return model, perf


def save_model(model: Any, dataset: str, model_name: str, random_state: int, perf_df: pd.DataFrame):

    # Make folder models/dataset/models
    folder_path = os.path.join("models", dataset)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, f"{model_name}_{random_state}")
    # Make folder for models/dataset/models/model_name/
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # Pickle model
    filename = "model.joblib"
    dump(model, os.path.join(file_path, filename))

    # Save performance in CSV file
    perf_df.to_csv(os.path.join(file_path, "performance.csv"), index=False)

    # Save model hyperparameters
    json.dump(model.get_params(), open(os.path.join(file_path, "hparams.json"), "w"), indent=4)



######################## GRANITE Utils ############################get#


def get_functional_decomposition(
        background: np.ndarray,
        model: Any,
        model_name: str,
        features: Features,
        ) -> dict[tuple[int, ...], np.ndarray]:
    """ This function computes the FD and caches it locally. It already cached, then it load the decomposition """

    # Use the partitioning tree
    if model_name in ["gbt", "rf"]:
        decomposition = get_components_tree(model, background, background, features=features, anchored=True)
    else:
        decomposition = get_components_brute_force(model.predict, background, background, features=features, anchored=True)
    return decomposition


def get_granite_loss_functions(
        decomposition,
        background: np.ndarray,
        targets: np.ndarray,
        task: str,
        cat_features_indices=list[int]
    ):
    N, d = background.shape
    U = [(i,) for i in range(d)]
    # TODO genralize to classification
    risk_fn = pointwise_squared_risk
    loss_functions = {}


    # Report the error between PDP and full-Marginal
    loss_functions['PDP_vs_ICE'] =\
        get_marginal_pure_vs_full_loss_fn(
            decomposition=decomposition,
            U=U
        )

    # Bin along each feature to do Mplots
    _, binned_data = create_bins_for_data(
        background,
        cat_features_indices,
        n_bins_numerical=5
    )
    loss_functions['PDP_vs_Mplot'] =\
        get_marginal_vs_conditional_pure_loss_fn(
            decomposition,
            U,
            binned_data
        )

    loss_functions['PureRisk_vs_FullRisk'] =\
        get_marginal_pure_vs_full_risk_loss_fn(
            decomposition,
            U,
            y=targets,
            risk_fn=risk_fn,
        )

    # Bin data using C-Trees
    _, binned_data = create_bins_for_data_partition_tree(
        background, cat_feature_indices=cat_features_indices, max_leaf_nodes=5
    )
    loss_functions['PFI_vs_CFI'] =\
        get_marginal_vs_condition_full_risk_loss_fn(
            decomposition,
            U,
            binned_data,
            y=targets,
            risk_fn=risk_fn
        )
    return loss_functions



def save_GRANITE_disagreements(disagreements: dict, dataset: str, model_name: str, random_state: int, cfg: Granite_Config):
    # Add elements to the dict
    disagreements["loss_fn_minimized"] = cfg.loss_fn
    disagreements["dataset"] = dataset
    disagreements["model_name"] = model_name
    disagreements["random_state"] = random_state
    disagreements["max_depth"] = cfg.max_depth
    disagreements["alpha"] = cfg.alpha

    dataframe = pd.DataFrame(disagreements, index=[0])

    filepath = os.path.join("models", dataset, f"{model_name}_{random_state}", "results.csv")
    if os.path.exists(filepath):
        existing_dataframe = pd.read_csv(filepath)
        dataframe = pd.concat((existing_dataframe, dataframe), axis=0, ignore_index=True)

    dataframe.to_csv(filepath, index=False)


# def save_GRANITE(tree, dataset: str, model_name: str, random_state: int, cfg: Granite_Config):

#     # Make folder for dataset models
#     folder_path = os.path.join("models", dataset)
#     file_path = os.path.join(folder_path, f"{model_name}_{random_state}")

#     # Pickle model
#     filename = f"{cfg.loss_fn}_{cfg.max_depth}_{cfg.alpha}_{cfg.train_background_size}"
#     dump(tree, os.path.join(file_path, filename + ".joblib"))

#     # Save the print
#     tree_string = tree.print(return_string=True, verbose=True)
#     with open(os.path.join(file_path, filename + ".txt"), "w") as f:
#         f.write(tree_string)



# def load_GRANITE(dataset, model_name, random_state, cfg):

#     # Make folder for dataset models
#     folder_path = os.path.join("models", dataset)
#     file_path = os.path.join(folder_path, f"{model_name}_ {random_state}")

#     # Pickle model
#     filename = f"{cfg.loss_fn}_{cfg.max_depth}_{cfg.alpha}_{cfg.train_background_size}"
#     return load(os.path.join(file_path, filename + ".joblib"))
