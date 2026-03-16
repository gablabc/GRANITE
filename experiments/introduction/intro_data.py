import copy
import numpy as np
import pandas as pd
from shapiq import ExactComputer, Game


def generate_problem(N, seed):
    # Generate input

    np.random.seed(seed)
    X_1 = np.random.normal(0, 1, size=(N,))
    X_2 = 2 * np.random.randint(0, 2, size=(N,)) - 1
    X_3 = 2 * np.random.randint(0, 2, size=(N,)) - 1
    X_4 = np.random.normal(X_3, 1, size=(N,))

    # Model to explain
    def f(X):
        return 3 * X[:, 0] * X[:, 1] + X[:, 2] + 2 * X[:, 3]

    return np.column_stack((X_1, X_2, X_3, X_4)), f


class conditionalGame(Game):
    def __init__(
        self,
        model,
        data,
        x_explain,
        background_data,
        n_players=4,
        seed=42,
        normalize=False,
    ):
        self.x_explain = x_explain
        super().__init__(n_players=n_players, normalize=normalize)
        self.background_data = copy.copy(background_data)
        self.model = model
        self.original_data = data

        # np.random.seed(seed)
        # self.replacement_x1 = np.random.normal(0, 1, size=(self.n_background_data,))
        # self.replacement_x2 = 2 * np.random.randint(0, 2, size=(self.n_background_data,)) - 1
        # self.replacement_x3 = 2 * np.random.randint(0, 2, size=(self.n_background_data,)) - 1
        # # sample conditional on X_3
        # self.replacement_x4_joint = np.random.normal(self.replacement_x3, 1, size=(self.n_background_data,))
        # self.replacement_x4_cond_x3 = np.random.normal(self.x_explain[2], 1, size=(self.n_background_data,))
        # #self.replacement_x4_joint2 = np.random.normal(X_3, 1, size=(self.n_background_data,))

    def conditional_data_gen(self, coalition):
        conditional_data = copy.deepcopy(self.background_data)
        for i, conditional_flag in enumerate(coalition):
            if conditional_flag:
                # keep the values of x_explain
                conditional_data[:, i] = self.x_explain[i]
            else:
                if i == 3 and coalition[2]:
                    conditional_data = conditional_data[
                        self.background_data[:, 2] == self.x_explain[2], :
                    ]
        return conditional_data

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        rslt = np.zeros(np.shape(coalitions)[0])
        for i, subset in enumerate(coalitions):
            data_to_predict = self.conditional_data_gen(subset)
            rslt[i] = np.mean(self.model(data_to_predict))
        return rslt


class marginalGame(Game):
    def __init__(self, model, x_explain, background_data, n_players=4, normalize=False):
        self.x_explain = x_explain
        super().__init__(n_players=n_players, normalize=normalize)
        self.n_background_data = np.shape(background_data)[0]
        self.shuffled_data = copy.deepcopy(background_data)
        self.shuffled_data = self.shuffled_data[
            np.random.permutation(self.n_background_data)
        ]
        self.model = model

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        rslt = np.zeros(np.shape(coalitions)[0])
        for i, subset in enumerate(coalitions):
            # repeat self.x_explain n_background_data time
            data_to_predict = np.tile(self.x_explain, (self.n_background_data, 1))
            data_to_predict[:, ~subset] = self.shuffled_data[:, ~subset]
            rslt[i] = np.mean(self.model(data_to_predict))
        return rslt


def compute_full(game):
    rslt = np.zeros(game.n_players)
    all_players = np.ones(game.n_players, dtype=bool)
    for i in range(game.n_players):
        removed_i = np.ones(game.n_players, dtype=bool)
        removed_i[i] = False
        rslt[i] = game(all_players)[0] - game(removed_i)[0]
    return rslt


def compute_pure(game):
    rslt = np.zeros(game.n_players)
    none = np.zeros(game.n_players, dtype=bool)
    for i in range(game.n_players):
        only_i = np.zeros(game.n_players, dtype=bool)
        only_i[i] = True
        rslt[i] = game(only_i)[0] - game(none)[0]
    return rslt


def compute_explanations(region, model, data, x):
    regional_marginal_game = marginalGame(model, x, region)
    exact_computer = ExactComputer(regional_marginal_game, 4)
    regional_shapley_marginal = exact_computer(index="SV").values[1:]
    regional_full_marginal = compute_full(regional_marginal_game)
    regional_pure_marginal = compute_pure(regional_marginal_game)

    regional_conditional_game = conditionalGame(model, data, x, background_data=region)
    exact_computer = ExactComputer(regional_conditional_game, 4)
    regional_shapley_conditional = exact_computer(index="SV").values[1:]
    regional_full_conditional = compute_full(regional_conditional_game)
    regional_pure_conditional = compute_pure(regional_conditional_game)

    return (
        regional_shapley_marginal,
        regional_full_marginal,
        regional_pure_marginal,
        regional_shapley_conditional,
        regional_full_conditional,
        regional_pure_conditional,
    )


def _make_storage_dict(
    region_name: str,
    x_i: np.ndarray | list,
    phi_i: np.ndarray | list,
    instance_id: int,
    method_name: str,
) -> dict[str, float | str]:
    """Turns information into a serializable dictionary."""
    x_i_arr = np.asarray(x_i).flatten()
    x_i_dict = {f"x_{j+1}": float(v) for j, v in enumerate(x_i_arr)}
    phi_i_arr = np.asarray(phi_i).flatten()
    phi_i_dict = {f"phi_{j+1}": float(v) for j, v in enumerate(phi_i_arr)}
    return {
        "region": region_name,
        "instance_id": instance_id,
        "method": method_name,
        **x_i_dict,
        **phi_i_dict,
    }


def create_explanations(
    n_phi=100, N=10000, random_seed=42, *, print_result: bool = False
):
    data, model = generate_problem(N, random_seed)
    # y_true = model(data)
    region_x2_1 = data[data[:, 1] == 1, :]
    region_x3_1 = data[data[:, 2] == 1, :]
    region_x2_1_x3_1 = data[(data[:, 1] == 1) & (data[:, 2] == 1), :]

    explanations = []

    REGIONS = {
        "Full space": data,
        "Region $X_2=1$": region_x2_1,
        "Region $X_3=1$": region_x3_1,
        "Region $X_2=1 \\wedge X_3=1$": region_x2_1_x3_1,
    }
    for name, region in REGIONS.items():
        print("------------REGION: ", name, "--------------")
        for i, x in enumerate(region):
            (
                regional_shapley_marginal,
                regional_full_marginal,
                regional_pure_marginal,
                regional_shapley_conditional,
                regional_full_conditional,
                regional_pure_conditional,
            ) = compute_explanations(region, model, data, x)

            # marginal ----------
            explanations.append(
                _make_storage_dict(
                    region_name=name,
                    x_i=x,
                    phi_i=regional_pure_marginal,
                    instance_id=i,
                    method_name="m-Pure",
                )
            )
            explanations.append(
                _make_storage_dict(
                    region_name=name,
                    x_i=x,
                    phi_i=regional_shapley_marginal,
                    instance_id=i,
                    method_name="m-Shapley",
                )
            )
            explanations.append(
                _make_storage_dict(
                    region_name=name,
                    x_i=x,
                    phi_i=regional_full_marginal,
                    instance_id=i,
                    method_name="m-Full",
                )
            )

            # conditional ----------
            explanations.append(
                _make_storage_dict(
                    region_name=name,
                    x_i=x,
                    phi_i=regional_pure_conditional,
                    instance_id=i,
                    method_name="c-Pure",
                )
            )
            explanations.append(
                _make_storage_dict(
                    region_name=name,
                    x_i=x,
                    phi_i=regional_shapley_conditional,
                    instance_id=i,
                    method_name="c-Shapley",
                )
            )
            explanations.append(
                _make_storage_dict(
                    region_name=name,
                    x_i=x,
                    phi_i=regional_full_conditional,
                    instance_id=i,
                    method_name="c-Full",
                )
            )

            if print_result:
                print("---------OBSERVATION ", i, "----------")
                print("----------MARGINAL-----------")
                print("Pure: ", regional_pure_marginal)
                print("Shapley: ", regional_shapley_marginal)
                print("Full: ", regional_full_marginal)
                print("----------CONDITIONAL-----------")
                print("Pure: ", regional_pure_conditional)
                print("Shapley: ", regional_shapley_conditional)
                print("Full: ", regional_full_conditional)
            if i > n_phi:
                break

    explanations_df = pd.DataFrame(explanations)
    explanations_df.to_csv("intro_illustration_local_explanations.csv", index=False)


if __name__ == "__main__":
    create_explanations(1_000, 10000, 42, print_result=False)
