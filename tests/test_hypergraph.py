import time

from sklearn.model_selection import ParameterGrid

from hypper.hypergraph import HyperGraph
from hypper.data import read_german_data, read_spect_heart, read_sample_data

DS = [read_german_data(), read_spect_heart(), read_sample_data()]


def test_weighting(
    dfs=DS,
    weighting_iterations_list=[1, 2, 5],
    weighting_normalization_strategy_list=["max", "l1", "l2"],
    iterations_history=[True, False],
):
    start = time.time()
    pg = ParameterGrid(
        {
            "wi": weighting_iterations_list,
            "wns": weighting_normalization_strategy_list,
            "dfs": dfs,
            "ih": iterations_history,
        }
    )
    for pgi in pg:
        hg = HyperGraph(input_data=pgi["dfs"][0], label=pgi["dfs"][1])
        hg.calculate_weights(
            pgi["wi"], normalization_strategy=pgi["wns"], iter_history=pgi["ih"]
        )
        assert hg.edges_weights.size != 0
        assert hg.vertices_weights.size != 0
    end = time.time()
    print(f"Weighting testing time: {end - start}")
