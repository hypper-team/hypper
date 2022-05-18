from sklearn.model_selection import ParameterGrid

from hypper.feature_selection import CDWFS, RandomWalkFS
from hypper.data import read_german_data, read_spect_heart, read_sample_data

DS = [read_german_data(), read_spect_heart(), read_sample_data()]


def test_CDWFS(
    dfs=DS,
    weighting_iterations_list=[1, 2, 5],
    weighting_normalization_strategy_list=["max", "l1", "l2"],
    feature_values=[True, False],
):
    pg = ParameterGrid(
        {
            "wi": weighting_iterations_list,
            "wns": weighting_normalization_strategy_list,
            "fv": feature_values,
            "dfs": dfs,
        }
    )
    for pgi in pg:
        cdwfs = CDWFS(
            weighting_iterations=pgi["wi"],
            weighting_normalization_strategy=pgi["wns"],
            feature_values=feature_values,
            random_seed=42,
        )
        out = cdwfs.fit_transform(pgi["dfs"][0], label_column=pgi["dfs"][1])
        assert out.size != 0


def test_RWFS(
    dfs=DS,
    iterations=[1, 10],
    walk_length=[2, 3],
    scoring_variant=["v1_1", "v1_2", "v1_3", "v1_4"],
    feature_values=[True, False],
):
    pg = ParameterGrid(
        {
            "i": iterations,
            "wl": walk_length,
            "dfs": dfs,
            "sv": scoring_variant,
            "fv": feature_values,
        }
    )
    for pgi in pg:
        rwfs = RandomWalkFS(
            iterations=pgi["i"],
            walk_length=pgi["wl"],
            scoring_variant=pgi["sv"],
            feature_values=pgi["fv"],
        )
        out = rwfs.fit_transform(pgi["dfs"][0], label_column=pgi["dfs"][1])
        assert out.size != 0
