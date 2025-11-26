import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


# Local imports
from utils import setup_data, get_train_test_split, get_cross_validator, get_scoring
from utils import MODELS, Search_Config, save_model, get_hp_grid


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    # Parse arguments
    parser = ArgumentParser()
    parser.add_arguments(Search_Config, "search")
    parser.add_argument("--model_name", type=str, default="gbt", help="Type of model: gbt, rf, mlp")
    parser.add_argument("--dataset", type=str, default="bike", help="Dataset to use: bike, ")
    parser.add_argument("--random_state", type=int, default=0, help="Seed that controls every non-deterministic op")

    args, unknown = parser.parse_known_args()
    print(args)

    # Load data
    X, y, features, task = setup_data(args.dataset)
    x_train, x_test, y_train, y_test = get_train_test_split(X, y, task, random_state=args.random_state)

    # Load model
    model = MODELS[args.model_name][task]
    model.set_params(random_state=args.random_state)

    cross_validator = get_cross_validator(
        k=args.search.n_splits,
        task=task,
        random_state=args.random_state,
        split_type=args.search.split_type,
    )
    hp_dict = get_hp_grid(args.dataset, args.model_name)
    scoring = get_scoring(task)

    cv_search = RandomizedSearchCV(
        model,
        hp_dict,
        scoring=scoring,
        n_iter=args.search.n_repetitions,
        n_jobs=args.search.n_jobs,
        cv=cross_validator,
        verbose=2,
        random_state=args.random_state
    ).fit(x_train, y_train)
    model = cv_search.best_estimator_

    # Assess train/test performance
    if task == "regression":
        perf_train = np.sqrt(np.mean((model.predict(x_train) - y_train) ** 2))
        perf_test  = np.sqrt(np.mean((model.predict(x_test) - y_test) ** 2))
    else:
        perf_train = np.mean(model.predict(x_train) == y_train)
        perf_test  = np.mean(model.predict(x_test) == y_test)

    perf_df = pd.DataFrame([[perf_train, perf_test]], columns=["Train", "Test"])
    print(perf_df)
    print("Saving Results")
    save_model(model, args.dataset, args.model_name, args.random_state, perf_df)
