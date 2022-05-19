import logging
from copy import copy
from random import random, seed
from typing import Optional

import numpy as np
import pandas as pd

from ..base import Base, TransformerMixin
from ..utils import BASE_LOGGING_LEVEL


class CDWU(Base, TransformerMixin):
    """Hypergraph-based undersampling method.

    Method resamples dataset based on class-dependent weighting algorithm.
    """

    def __init__(
        self,
        weighting_iterations: int,
        weighting_normalization_strategy: Optional[str] = "max",
        weighting_history: Optional[bool] = True,
        version: Optional[int] = 1,
        majority_left_threshold: Optional[float] = 0.0,
        randomize_A: Optional[float] = 0.0,
        random_seed: Optional[int] = 42,
        verbosity: Optional[logging.LogRecord] = BASE_LOGGING_LEVEL,
    ) -> None:
        """_summary_

        Args:
            weighting_iterations (int): Number of weighting iterations during hypergraph class-dependent weighting method.
            weighting_normalization_strategy (str, optional): Type of normalization during hypergraph class-dependent weighting method. Defaults to 'max'. Options: 'max', 'l1', 'l2'.
            weighting_history (bool, optional): If True use hyperedges and vertices weights from every weighting iteration to calculate final scores. Defaults to True.
            version (int, optional): Sample score calculation version. Defaults to 1. Options:
                * 1 - `majority_class_sample_weight - minority_class_sample_weight` - promotes samples close to majority class (with repsect to minority class);
                * 2 - `majority_class_sample_weight + minority_class_sample_weight` - promotes samples close to both classes;
                * 3 - `minority_class_sample_weight` - promotes samples close to minority class;
                * 4 - `minority_class_sample_weight - majority_class_sample_weight` - promotes samples close to minority class (with respect to majority class);
                * 5 - `abs(majority_class_sample_weight - minority_class_sample_weight)` - promotes samples close to any class;
                * 6 - `majority_class_sample_weight` - promotes samples close to majority class.
            majority_left_threshold (float, optional): Parameter controlling precentage of additional samples from previously rejected. E.g. for binary class distribution 20-80 and `majority_left_threshold = 0.5`, majority class will be reduced to 50 samples `(20 + (80-20)*0.5)`.  Defaults to 0.0.
            randomize_A (float, optional): Randomization strength of the final output. 0.0 means lack of randomization. Defaults to 0.0.
            random_seed (int, optional): Random seed. Defaults to 42.
            verbosity (logging.LogRecord, optional): Specifies the lowest-severity log message a logger will handle. Defaults to logging.WARNING.

        Raises:
            ValueError: `majority_left_threshold` is out of expected range <0.0, 1.0).
            SyntaxError: Unrecognized undersampling `version` selected.
        """
        self.weighting_iterations = weighting_iterations
        self.weighting_normalization_strategy = weighting_normalization_strategy
        self.weighting_history = weighting_history
        if not (0.0 <= majority_left_threshold < 1.0):
            raise ValueError(
                f"Expected value in range <0.0, 1.0), while majority_left_threshold = {majority_left_threshold}"
            )
        self.majority_left_threshold = majority_left_threshold
        self.version = version
        self.randomize_A = randomize_A

        self.random_seed = random_seed
        self.verbosity = verbosity
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)

    def _undersampling(
        self,
        weighting_iterations: int,
        weighting_normalization_strategy: str,
        majority_left_threshold: float,
        weighting_history: bool,
        version: int,
        randomize_A: float,
    ) -> pd.DataFrame:
        # Find majority class
        initial_class_distribution = copy(self.hg.vertices_weights)
        distribution = initial_class_distribution.sum(axis=0)
        size_diff_between_classes = int(np.abs(np.diff(distribution)[0, 0]))
        minority_class = distribution.argmin()
        if minority_class == 0:
            majority_class = 1
        elif minority_class == 1:
            majority_class = 0
        else:
            Warning("Multiclass classification problems are not supported.")
        # Calculates class-dependent vertices weights
        self.hg.calculate_weights(
            iterations=weighting_iterations,
            normalization_strategy=weighting_normalization_strategy,
            iter_history=True,
        )
        # Reduce vertex weight to single value
        if version == 1:
            if weighting_history:
                subtracted_weights = (
                    self.hg.weighting_iteration_history[0][1][
                        :, majority_class
                    ].todense()
                    - self.hg.weighting_iteration_history[0][1][
                        :, minority_class
                    ].todense()
                )
                for _, vertices_weights in self.hg.weighting_iteration_history[1:]:
                    subtracted_weights += (
                        vertices_weights[:, majority_class].todense()
                        - vertices_weights[:, minority_class].todense()
                    )
            else:
                subtracted_weights = (
                    self.hg.vertices_weights[:, majority_class].todense()
                    - self.hg.vertices_weights[:, minority_class].todense()
                )
        elif version == 2:
            if weighting_history:
                subtracted_weights = (
                    self.hg.weighting_iteration_history[0][1][
                        :, majority_class
                    ].todense()
                    + self.hg.weighting_iteration_history[0][1][
                        :, minority_class
                    ].todense()
                )
                for _, vertices_weights in self.hg.weighting_iteration_history[1:]:
                    subtracted_weights += (
                        vertices_weights[:, majority_class].todense()
                        + vertices_weights[:, minority_class].todense()
                    )
            else:
                subtracted_weights = (
                    self.hg.vertices_weights[:, majority_class].todense()
                    + self.hg.vertices_weights[:, minority_class].todense()
                )
        elif version == 3:
            if weighting_history:
                subtracted_weights = self.hg.weighting_iteration_history[0][1][
                    :, minority_class
                ].todense()
                for _, vertices_weights in self.hg.weighting_iteration_history[1:]:
                    subtracted_weights += vertices_weights[:, minority_class].todense()
            else:
                subtracted_weights = self.hg.vertices_weights[
                    :, minority_class
                ].todense()
        elif version == 4:
            if weighting_history:
                subtracted_weights = (
                    self.hg.weighting_iteration_history[0][1][
                        :, minority_class
                    ].todense()
                    - self.hg.weighting_iteration_history[0][1][
                        :, majority_class
                    ].todense()
                )
                for _, vertices_weights in self.hg.weighting_iteration_history[1:]:
                    subtracted_weights += (
                        vertices_weights[:, minority_class].todense()
                        - vertices_weights[:, majority_class].todense()
                    )
            else:
                subtracted_weights = (
                    self.hg.vertices_weights[:, minority_class].todense()
                    - self.hg.vertices_weights[:, majority_class].todense()
                )
        elif version == 5:
            if weighting_history:
                subtracted_weights = np.abs(
                    self.hg.weighting_iteration_history[0][1][:, 0].todense()
                    - self.hg.weighting_iteration_history[0][1][:, 1].todense()
                )
                for _, vertices_weights in self.hg.weighting_iteration_history[1:]:
                    subtracted_weights += np.abs(
                        vertices_weights[:, 0].todense()
                        - vertices_weights[:, 1].todense()
                    )
            else:
                subtracted_weights = (
                    self.hg.vertices_weights[:, 0].todense()
                    - self.hg.vertices_weights[:, 1].todense()
                )
        elif version == 6:
            if weighting_history:
                subtracted_weights = self.hg.weighting_iteration_history[0][1][
                    :, majority_class
                ].todense()
                for _, vertices_weights in self.hg.weighting_iteration_history[1:]:
                    subtracted_weights += vertices_weights[:, majority_class].todense()
            else:
                subtracted_weights = self.hg.vertices_weights[
                    :, majority_class
                ].todense()
        else:
            raise SyntaxError(f"Unknown scoring version: {version}")
        # Add random selection
        if randomize_A > 0.0:
            # num_to_remove = int(np.max(distribution)/50)
            # low_weight_indxs = np.argsort(subtracted_weights, axis=None)[:,:num_to_remove].tolist()[0]
            temp_max_weight = np.amax(subtracted_weights)
            seed(self.random_seed)
            subtracted_weights = np.apply_along_axis(
                lambda x: x + randomize_A * temp_max_weight * random(),
                1,
                subtracted_weights,
            ).reshape(-1, 1)
            # subtracted_weights[np.r_[low_weight_indxs]] = subtracted_weights.min()-1

        # Replace minority class with max values (can't be removed)
        subtracted_weights[
            initial_class_distribution[:, minority_class].todense() == 1
        ] = (subtracted_weights.max() + 1)
        # Find vertices to drop
        num_of_idxs_to_drop = int(
            min(
                size_diff_between_classes,
                size_diff_between_classes
                - np.ceil(size_diff_between_classes * majority_left_threshold),
            )
        )
        indxs_to_drop = [
            self.hg.vertices.inverse[i]
            for i in subtracted_weights.argsort(axis=None)[
                :, :num_of_idxs_to_drop
            ].tolist()[0]
        ]
        return pd.concat(
            [self.hg.X.drop(indxs_to_drop), self.hg.y.drop(indxs_to_drop)], axis=1
        )

    def fit(self, data: pd.DataFrame, label_column: str):
        """Fit data.

        Args:
            data (pd.DataFrame): Input data.
            label_column (str): Label column name.

        Returns:
            pd.DataFrame: Resampled DataFrame with reduced majority class.
        """
        return super().fit(data, label_column)

    def fit_transform(self, data: pd.DataFrame, label_column: str, **fit_params):
        """Fit and transform data.

        Args:
            data (pd.DataFrame): Input data.
            label_column (str): Label column name.

        Returns:
            pd.DataFrame: Resampled DataFrame with reduced majority class.
        """
        return super().fit_transform(data, label_column, **fit_params)

    def transform(self):
        """Transform fitted data and return resampled dataset.

        Returns:
            pd.DataFrame: Resampled DataFrame with reduced majority class.
        """
        return self._undersampling(
            weighting_iterations=self.weighting_iterations,
            weighting_normalization_strategy=self.weighting_normalization_strategy,
            majority_left_threshold=self.majority_left_threshold,
            weighting_history=self.weighting_history,
            version=self.version,
            randomize_A=self.randomize_A,
        )
