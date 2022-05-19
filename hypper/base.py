import pandas as pd

from abc import ABC, abstractmethod

from .hypergraph import HyperGraph


class Base:
    """Base class implements fit function for all hypergraph-based methods."""

    def fit(self, data: pd.DataFrame, label_column: str):
        self.hg = HyperGraph(
            input_data=data,
            label=label_column,
            random_seed=self.random_seed,
            verbosity=self.verbosity,
        )
        return self


class TransformerMixin(ABC):
    """Mixin class for all transformers in Hypper.

    Raises:
        NotImplementedError: Raised when `transform` function was not implemented.

    Returns:
        np.ndarray: Transformed array.
    """

    @abstractmethod
    def transform(self):
        raise NotImplementedError("transform() method must be implemented")

    def fit_transform(self, data: pd.DataFrame, label_column: str, **fit_params):
        return self.fit(data, label_column, **fit_params).transform()


class PredictorMixin(ABC):
    """Mixin class for all predictors in Hypper.

    Returns:
        np.ndarray: Predictions array.
    """

    @abstractmethod
    def predict(self):
        """Method returns class predictions.

        Raises:
            NotImplementedError: Raised when `predict` function was not implemented.
        """
        raise NotImplementedError("predict() method must be implemented")

    @abstractmethod
    def predict_proba(self):
        """Method returns class probabilities predictions.

        Raises:
            NotImplementedError: Raised when `predict_proba` function was not implemented.
        """
        raise NotImplementedError("predict_proba() method must be implemented")
