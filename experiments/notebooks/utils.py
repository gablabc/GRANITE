
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV



def train_gbt_bikesharing(X_train, y_train):
    os.makedirs("models", exist_ok=True)
    model = HistGradientBoostingRegressor(random_state=0)
    model_path = os.path.join("models", "bike_gbt.joblib")

    if os.path.exists(model_path):
        print("Model already exists. Loading from disk.")
        model = load(model_path)
    else:
        print("Training the model...")
        grid = {"learning_rate": np.logspace(-2, 0, 10),
                "max_depth" : [3, 4, 5, 6, 7],
                "max_iter" : [50, 100, 150, 200],
                'min_samples_leaf' : [1, 20, 40, 60, 80, 100]}

        search = RandomizedSearchCV(
            model,
            cv=KFold(),
            scoring='neg_root_mean_squared_error',
            param_distributions=grid,
            verbose=2,
            n_iter=20,
            random_state=42
        )
        search.fit(X_train, y_train)
        print(search)

        # get the model with the best hyperparameters
        model = search.best_estimator_
        dump(model, model_path)

        # visualize the results
        res = search.cv_results_
        cv_perf = np.nan_to_num(-res['mean_test_score'], nan=1e10)
        best_idx = np.argmin(cv_perf)
        plt.scatter(res['param_learning_rate'], cv_perf, c='b', alpha=0.75)
        plt.plot(res['param_learning_rate'][best_idx], cv_perf[best_idx], 'r*', markersize=10, markeredgecolor='k')
        plt.xlabel("Learning Rate")
        plt.ylabel("Cross-Validated RMSE")
        plt.xscale('log')
        plt.show()

    return model

def train_mlp_kin8nm(X_train, y_train):
    os.makedirs("models", exist_ok=True)
    model = MLPRegressor(random_state=42, max_iter=500)

    model_path = os.path.join("models", "kin8nm_mlp.joblib")

    if os.path.exists(model_path):
        print("Model already exists. Loading from disk.")
        model = load(model_path)
    else:
        print("Training the model...")
        # For MLP
        grid = {"hidden_layer_sizes": [(10, 5), (20, 10), (30, 15)],
                "learning_rate_init": np.logspace(-4, -1, 10),
                "alpha": np.logspace(-4, -1, 10),
                "momentum": [0.75, 0.9, 0.99, 0.999]
        }

        search = RandomizedSearchCV(
            model,
            cv=KFold(),
            scoring='neg_root_mean_squared_error',
            param_distributions=grid,
            verbose=2,
            n_iter=20,
            random_state=42
        )
        search.fit(X_train, y_train)
        print(search)

        model = search.best_estimator_
        dump(model, os.path.join("models", "kin8nm_mlp.joblib"))

        res = search.cv_results_
        cv_perf = np.nan_to_num(-res['mean_test_score'], nan=1e10)
        best_idx = np.argmin(cv_perf)
        plt.figure(figsize=(4, 4))
        plt.scatter(res['param_learning_rate_init'], cv_perf, c='b', alpha=0.75)
        plt.plot(res['param_learning_rate_init'][best_idx], cv_perf[best_idx], 'r*', markersize=10, markeredgecolor='k')
        plt.xlabel("Learning Rate")
        plt.ylabel("Cross-Validated RMSE")
        plt.xscale('log')
        plt.show()
    
    return model

def train_gbt_diabetes(X_train, y_train):
    os.makedirs("models", exist_ok=True)
    model = HistGradientBoostingRegressor(random_state=0)
    model_path = os.path.join("models", "diabetes_gbt.joblib")

    if os.path.exists(model_path):
        print("Model already exists. Loading from disk.")
        model = load(model_path)
    else:
        print("Training the model...")
        grid = {
            "learning_rate": np.logspace(-2, 0, 10),
            "max_depth" : [3, 4, 5, 6, 7],
            "max_iter" : [50, 100, 150, 200],
            'min_samples_leaf' : [1, 20, 40, 60, 80, 100]
        }

        search = RandomizedSearchCV(
            model,
            cv=KFold(),
            scoring='neg_root_mean_squared_error',
            param_distributions=grid,
            verbose=2,
            n_iter=20,
            random_state=42
        )
        search.fit(X_train, y_train)
        print(search)

        # get the model with the best hyperparameters
        model = search.best_estimator_
        dump(model, model_path)

        # visualize the results
        res = search.cv_results_
        cv_perf = np.nan_to_num(-res['mean_test_score'], nan=1e10)
        best_idx = np.argmin(cv_perf)
        plt.scatter(res['param_learning_rate'], cv_perf, c='b', alpha=0.75)
        plt.plot(res['param_learning_rate'][best_idx], cv_perf[best_idx], 'r*', markersize=10, markeredgecolor='k')
        plt.xlabel("Learning Rate")
        plt.ylabel("Cross-Validated RMSE")
        plt.xscale('log')
        plt.show()

    return model