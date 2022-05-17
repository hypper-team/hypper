from sklearn.model_selection import ParameterGrid

from hypper.classification import CDWC
from hypper.data_reader import read_german_data, read_spect_heart, read_sample_data

DS = [read_german_data(), read_spect_heart(), read_sample_data()]


def test_CDWC(
    dfs=DS,
    weighting_iterations_list=[1, 2, 5],
    weighting_normalization_strategy_list=["max", "l1", "l2"],
):
    pg = ParameterGrid(
        {
            "wi": weighting_iterations_list,
            "wns": weighting_normalization_strategy_list,
            "dfs": dfs,
        }
    )
    for pgi in pg:
        cdwc = CDWC(
            weighting_iterations=pgi["wi"],
            weighting_normalization_strategy=pgi["wns"],
            random_seed=42,
            verbosity=False,
        )
        cdwc.fit(pgi["dfs"][0], label_column=pgi["dfs"][1])
        y_pred = cdwc.predict(pgi["dfs"][0].drop(pgi["dfs"][1], axis=1).head(3))
        y_pred_prob = cdwc.predict_proba(
            pgi["dfs"][0].drop(pgi["dfs"][1], axis=1).head(3)
        )
        assert y_pred.size != 0
        assert y_pred_prob.size != 0
