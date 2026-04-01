import os
import numpy as np
from joblib import load, dump
from utils import (
    setup_data,
    get_train_test_split,
    load_model,
    subsample_data,
    CAT_FEATURES_INDICES,
)
from utils import get_functional_decomposition, Granite_Config, get_granite_loss_functions
from utils import save_GRANITE_disagreements

from granite.fd_trees import FDTree

if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Granite_Config, "granite")
    parser.add_argument(
        "--model_name", type=str, default="gbt", help="Type of model: gbt, rf, mlp"
    )
    parser.add_argument(
        "--dataset", type=str, default="bike", help="Dataset to use: bike, "
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Seed that controls every non-deterministic op",
    )
    args, unknown = parser.parse_known_args()
    print(args)

    # Load data and model
    X, y, features, task = setup_data(args.dataset)
    x_train, x_test, y_train, y_test = get_train_test_split(
        X, y, task, random_state=args.random_state
    )
    # Load models
    model, perfs = load_model(args.dataset, args.model_name, args.random_state)

    # We train GRANITE on train data, and compute disagreements on test set
    background, y_background = subsample_data(
        x_train, y_train, args.granite.train_background_size, args.random_state
    )
    filename = os.path.join(
        "models",
        args.dataset,
        f"{args.model_name}_{args.random_state}",
        f"decomposition_{args.granite.train_background_size}.joblib",
    )
    if os.path.exists(filename):
        decomposition = load(filename)
    else:
        decomposition = get_functional_decomposition(
            background, model, args.model_name, features=features
        )
        dump(decomposition, filename)
    loss_fn = get_granite_loss_functions(
        decomposition,
        background,
        y_background,
        task,
        CAT_FEATURES_INDICES[args.dataset],
    )

    tree = FDTree(
        max_depth=args.granite.max_depth, features=features, alpha=args.granite.alpha
    )
    tree.fit(background, loss_fn=loss_fn[args.granite.loss_fn])
    tree.print(verbose=True)

    # On test set, compute four losses regionally and aggregate
    test_background, y_background = subsample_data(
        x_test, y_test, args.granite.test_background_size, args.random_state
    )
    N = len(test_background)
    regions = tree.predict(test_background)
    _, weights = np.unique(regions, return_counts=True)
    weights = weights / weights.sum()

    disagreements = {
        "PDP_vs_ICE": 0.0,
        "PDP_vs_Mplot": 0.0,
        "PureRisk_vs_FullRisk": 0.0,
        "PFI_vs_CFI": 0.0,
    }
    for r in range(tree.n_regions):
        region_subset = np.where(regions == r)[0]
        N_r = len(region_subset)
        regional_background = test_background[region_subset]
        regional_y = y_background[region_subset]
        regional_decomposition = get_functional_decomposition(
            regional_background, model, args.model_name, features=features
        )
        loss_fn = get_granite_loss_functions(
            regional_decomposition,
            regional_background,
            regional_y,
            task,
            CAT_FEATURES_INDICES[args.dataset],
        )
        for metric in disagreements.keys():
            disagreements[metric] += weights[r] * loss_fn[metric](np.arange(N_r)) / N
    save_GRANITE_disagreements(
        disagreements, args.dataset, args.model_name, args.random_state, args.granite
    )
