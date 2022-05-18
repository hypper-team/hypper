import pandas as pd

from sklearn.model_selection import ParameterGrid
from pandas.testing import assert_frame_equal
from pathlib import Path

from hypper.undersampling import CDWU
from hypper.data import read_german_data, read_spect_heart, read_sample_data

DS = [read_german_data(), read_spect_heart(), read_sample_data()]

TEST_RES = pd.read_csv(
    Path(__file__).parent / "tests_data" / "undersampling_german.csv", index_col=0
)


def test_CDWU(
    dfs=DS,
    weighting_iterations_list=[1, 2, 5],
    weighting_normalization_strategy_list=["max", "l1", "l2"],
    majority_left_threshold=[0.0, 0.1],
):
    pg = ParameterGrid(
        {
            "wi": weighting_iterations_list,
            "wns": weighting_normalization_strategy_list,
            "mlt": majority_left_threshold,
            "dfs": dfs,
        }
    )
    for pgi in pg:
        cdwu = CDWU(
            weighting_iterations=pgi["wi"],
            weighting_normalization_strategy=pgi["wns"],
            majority_left_threshold=pgi["mlt"],
            random_seed=42,
        )
        out = cdwu.fit_transform(pgi["dfs"][0], label_column=pgi["dfs"][1])
        assert out.size != 0


def test_deterministic_run():
    cdwu = CDWU(
        weighting_iterations=2,
        weighting_normalization_strategy="max",
        majority_left_threshold=0.5,
        random_seed=42,
    )
    assert_frame_equal(TEST_RES, cdwu.fit_transform(*DS[0][:2]))
