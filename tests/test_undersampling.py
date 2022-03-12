from sklearn.model_selection import ParameterGrid

from hypper.undersampling import CDWU
from hypper.data_reader import read_german_data, read_spect_heart, read_sample_data

DS = [
    read_german_data(),
    read_spect_heart(),
    read_sample_data()
]

def test_CDWU(dfs=DS, weighting_iterations_list = [1,2,5], weighting_normalization_strategy_list=['max','l1','l2'], majority_left_threshold=[0.0, 0.1]):
    pg = ParameterGrid({
        'wi': weighting_iterations_list,
        'wns': weighting_normalization_strategy_list,
        'mlt': majority_left_threshold,
        'dfs': dfs
    })
    for pgi in pg:
        cdwu = CDWU(weighting_iterations=pgi['wi'], weighting_normalization_strategy=pgi['wns'], majority_left_threshold=pgi['mlt'], random_seed=42, verbosity=False)
        out = cdwu.fit_transform(pgi['dfs'][0], label_column=pgi['dfs'][1])
        if out.size != 0: continue
    return True