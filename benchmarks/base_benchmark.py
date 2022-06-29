from pathlib import Path
from typing import List

import hypper.data as hd
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder


class BaseBenchmark:
    def __init__(
        self,
        param_grid: dict,
        savefile: str,
        random_seed: int = 9090,
        verbosity: bool = False,
    ) -> None:
        self.param_grid = ParameterGrid(param_grid)
        self.savefile = savefile
        self.random_seed = random_seed
        self.verbosity = verbosity

    def run_benchmark(self):
        raise NotImplementedError("Benchmark class requires `run_benchmark` method.")

    def load_datasets(self):
        return [
            getattr(hd, read_function)
            for read_function in self.config["read_functions"]
        ]

    def save(self, df):
        Path(self.savefile).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(self.savefile))

    def _cat_oe(self, data: pd.DataFrame, cat_cols: List[str]):
        for col in cat_cols:
            data[col] = LabelEncoder().fit_transform(data[col])
        return data

    def _cat_le(self, data: pd.DataFrame):
        return LabelEncoder().fit_transform(data)

    def _load_config(self, config_filepath: str):
        filepath = Path(__file__).parent / "configs" / config_filepath
        with open(filepath, "r") as yaml_file:
            self.config = yaml.safe_load(yaml_file)
