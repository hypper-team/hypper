import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..base import Base, PredictorMixin
from ..utils import BASE_LOGGING_LEVEL


class CDWC(Base, PredictorMixin):
    """Hypergraph-based binary classifier.

    Classifier is based on the hypergraph class-dependent weighting method.

    """

    def __init__(
        self,
        weighting_iterations: int,
        weighting_normalization_strategy: str = "max",
        random_seed: Optional[int] = 42,
        verbosity: Optional[logging.LogRecord] = BASE_LOGGING_LEVEL,
    ) -> None:
        """
        Args:
            weighting_iterations (int): Number of weighting iterations during hypergraph class-dependent weighting method.
            weighting_normalization_strategy (str, optional): Type of normalization during hypergraph class-dependent weighting method. Defaults to 'max'. Options: 'max', 'l1', 'l2'.
            random_seed (int, optional): Random seed. Defaults to 42.
            verbosity (logging.LogRecord, optional): Specifies the lowest-severity log message a logger will handle. Defaults to logging.WARNING.
        """
        self.weighting_iterations = weighting_iterations
        self.weighting_normalization_strategy = weighting_normalization_strategy

        self.random_seed = random_seed
        self.verbosity = verbosity
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)

    def fit(self, data: pd.DataFrame, label_column: str):
        """Fit data into classifier.

        Args:
            data (pd.DataFrame): Input data.
            label_column (str): Label column name.

        Returns:
            self
        """
        super().fit(data, label_column)
        self.hg.calculate_weights(
            iterations=self.weighting_iterations,
            normalization_strategy=self.weighting_normalization_strategy,
        )
        return self

    def _modify_fi(self, fi: pd.DataFrame) -> pd.DataFrame:
        """Method aggregates feature value pairs to features only with summed importances.

        Args:
            fi (pd.DataFrame): Pandas DataFrame where indexes are feature-value pairs and the only columns consists of feature importances.

        Returns:
            pd.DataFrame: Pandas DataFrame witha aggregated features.
        """
        # Known bug - columns starting with the same name and underscore are aggregated together
        prefixes = self.hg.X.columns
        grouper = [next(p for p in prefixes if p in c) for c in list(fi.index.values)]
        return fi.groupby(grouper, axis=0).sum()

    def classifier(self, X: pd.DataFrame) -> np.ndarray:
        """Classification method based on the class-dependent hypergraph weighting.

        Args:
            X (pd.DataFrame): Input data.

        Raises:
            TypeError: Input data unknown type error.
            ValueError: Extracting feature names error.

        Returns:
            np.ndarray: Array with predictions.
        """
        if type(X) == np.ndarray:
            X = pd.DataFrame(data=X, columns=self.hg.X.columns)
        elif type(X) != pd.DataFrame:
            raise TypeError(f"Unknown data type: {type(X)}")

        X = X.astype(str)

        # Calculate aggregated features
        feature_value_weights = pd.DataFrame(
            data=self.hg.edges_weights.todense().T,
            columns=list(range(len(self.hg.edges_labels))),
            index=self.hg.edges.keys(),
        )
        feature_weights = self._modify_fi(feature_value_weights)

        def calculate_score(row):
            try:
                feature_value_pairs = (
                    OneHotEncoder(drop=None, sparse=True, handle_unknown="error")
                    .fit(row.values.reshape(1, -1))
                    .get_feature_names_out(self.hg.X.columns)
                )
            except ValueError as ve:
                raise ValueError(
                    "Number of features does not match input data"
                ).with_traceback(ve.__traceback__)

            score_per_class = np.zeros((len(self.hg.edges_labels),))

            for fv in feature_value_pairs:
                try:
                    score_per_class += (
                        self.hg.edges_weights.getcol(self.hg.edges[fv]).todense().A1
                    )
                except KeyError:
                    feature = next(p for p in self.hg.X.columns if p in fv)
                    score_per_class += feature_weights.loc[
                        feature_weights.index == feature
                    ].values[0]

            return score_per_class

        return X.apply(calculate_score, axis=1, result_type="expand").values

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Method returns class predictions.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Array with predicted classes.
        """
        return np.argmax(self.classifier(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Method returns class probabilities predictions.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: Array with predicted probabilities for classes.
        """
        X = self.classifier(X)
        return X / X.sum(axis=1)[:, None]
