import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from ..base import Base, TransformerMixin
from ..utils import BASE_LOGGING_LEVEL


class BaseFS:
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


class CDWFS(BaseFS, Base, TransformerMixin):
    """Hypergraph-based feature selection method.

    Method generates feature importances for feature-value pairs (or aggregates results per feature) based on class-dependent weighting algorithm.

    """

    def __init__(
        self,
        weighting_iterations: int,
        weighting_normalization_strategy: Optional[str] = "max",
        feature_values: Optional[bool] = True,
        random_seed: Optional[int] = 42,
        verbosity: Optional[logging.LogRecord] = BASE_LOGGING_LEVEL,
    ) -> None:
        """
        Args:
            weighting_iterations (int): Number of weighting iterations during hypergraph class-dependent weighting method.
            weighting_normalization_strategy (str, optional): Type of normalization during hypergraph class-dependent weighting method. Defaults to 'max'. Options: 'max', 'l1', 'l2'.
            feature_values (bool, optional): If method should return feature-value pairs or features (aggregated) importances. Defaults to True for feature-value pairs.
            random_seed (int, optional): Random seed. Defaults to 42.
            verbosity (logging.LogRecord, optional): Specifies the lowest-severity log message a logger will handle. Defaults to logging.WARNING.
        """
        self.weighting_iterations = weighting_iterations
        self.weighting_normalization_strategy = weighting_normalization_strategy
        self.feature_values = feature_values

        self.random_seed = random_seed
        self.verbosity = verbosity
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)

    def _weight_based_fi(
        self,
        weighting_iterations: int,
        weighting_normalization_strategy: str,
        feature_values: bool,
    ) -> pd.DataFrame:
        self.logger.info("Calculating feature importances ...")

        self.hg.calculate_weights(
            iterations=weighting_iterations,
            normalization_strategy=weighting_normalization_strategy,
            iter_history=True,
        )

        fis = csr_matrix((1, self.hg.weighting_iteration_history[0][0].shape[1]))
        for edges_weights, _ in self.hg.weighting_iteration_history:
            fis += np.abs(edges_weights[0, :] - edges_weights[1, :])

        # Return feature-value pairs importances
        feature_value_importances = pd.DataFrame(
            {
                "feature_value": self.hg.edges.keys(),
                "feature_importance": fis.toarray()[0],
            }
        ).set_index("feature_value")

        self.logger.info("Feature importances calculated.")

        if feature_values:
            return feature_value_importances
        else:
            # Return features importances
            feature_importances = self._modify_fi(feature_value_importances)
            return feature_importances

    def fit(self, data: pd.DataFrame, label_column: str):
        """Fit data.

        Args:
            data (pd.DataFrame): Input data.
            label_column (str): Label column name.

        Returns:
            self
        """
        return super().fit(data, label_column)

    def fit_transform(self, data: pd.DataFrame, label_column: str, **fit_params):
        """Fit and transform data.

        Args:
            data (pd.DataFrame): Input data.
            label_column (str): Label column name.

        Returns:
            pd.DataFrame: DataFrame with feature names and feature importances columns.
        """
        return super().fit_transform(data, label_column, **fit_params)

    def transform(self) -> pd.DataFrame:
        """Transform fitted data and return feature importances.

        Returns:
            pd.DataFrame: DataFrame with feature names and feature importances columns.
        """
        return self._weight_based_fi(
            weighting_iterations=self.weighting_iterations,
            weighting_normalization_strategy=self.weighting_normalization_strategy,
            feature_values=self.feature_values,
        )


class RandomWalkFS(BaseFS, Base, TransformerMixin):
    """Hypergraph-based feature selection method.

    Method generates feature importances for feature-value pairs (or aggregates results per feature) based on random walk algorithm.
    """

    def __init__(
        self,
        iterations: int,
        walk_length: int,
        scoring_variant: Optional[str] = "v1_3",
        feature_values: Optional[bool] = True,
        random_seed: Optional[int] = 42,
        verbosity: Optional[logging.LogRecord] = BASE_LOGGING_LEVEL,
    ) -> None:
        """_summary_

        Args:
            iterations (int): Number of random walks iterations.
            walk_length (int): Random walks maximum length.
            scoring_variant (str, optional): Formula used to calculate score. Defaults to 'v1_3'. Options: 'v1_1', 'v1_2', 'v1_3' and 'v1_4'. Defaults to 'v1_3'.
            feature_values (bool, optional): _description_. Defaults to True.
            random_seed (int, optional): _description_. Defaults to 42.
            verbosity (logging.LogRecord, optional): Specifies the lowest-severity log message a logger will handle. Defaults to logging.WARNING.

        Raises:
            SyntaxError: Raised when unknown scoring variant is passed to a function.
        """
        self.iterations = iterations
        self.walk_length = walk_length
        self.scoring_variant = scoring_variant
        self.feature_values = feature_values

        self.random_seed = random_seed
        self.verbosity = verbosity
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)

    def _random_walk_based_fi(
        self,
        iterations: int,
        walk_length: int,
        scoring_variant: str,
        feature_values: bool,
    ) -> pd.DataFrame:
        self.logger.info("Calculating feature importances ...")

        ohe = OneHotEncoder(drop=None, sparse=True, handle_unknown="error")
        vertices_labels = ohe.fit_transform(self.hg.y.values.reshape(-1, 1))

        all_walks = defaultdict(list)

        # Get all available vertices per class (or for the whole dataset)
        vertices_main = np.array(range(vertices_labels.shape[0]))
        vertices_class1 = vertices_labels[:, 0].nonzero()[0]
        vertices_class2 = vertices_labels[:, 1].nonzero()[0]

        for _ in range(iterations):
            # Get random vertex for all samples (for both labels)
            vertex_idxs = [np.random.choice(vertices_main)]
            # Get random vertex for samples from the first and second class
            for vc in [vertices_class1, vertices_class2]:
                vertex_idxs.append(np.random.choice(vc))
            walk_history = [[], [], []]
            for iter_ in range(walk_length):
                # Algorithm should avoid visiting the same edges during a walk
                edge_idxs = []
                for i, (vi, wh) in enumerate(zip(vertex_idxs, walk_history)):
                    while True:
                        # Select edge
                        pos_temp = self._rw_possible_steps("edge", vi)
                        edge_temp = np.random.choice(pos_temp)
                        if edge_temp in wh:
                            if all(i in wh for i in pos_temp):
                                # Edge is added but the final score is ommited for duplicates
                                edge_idxs.append(edge_temp)
                                break  # When walk will be shorter than expected due to lack of unexplored connections
                            else:
                                continue
                        else:
                            edge_idxs.append(edge_temp)
                            break
                for j, vc in enumerate(
                    [vertices_main, vertices_class1, vertices_class2]
                ):
                    # Update visited edges
                    walk_history[j].append(edge_idxs[j])
                    # Select next vertex based on visited edge
                    if (
                        iter_ != walk_length - 1
                    ):  # No need to select next vertex for the last hyperedge in a path
                        vertex_idxs[j] = np.random.choice(
                            [
                                i
                                for i in self._rw_possible_steps("vertex", edge_idxs[j])
                                if i in vc
                            ]
                        )
            all_walks["main"].append(tuple(sorted(walk_history[0])))
            all_walks["class1"].append(tuple(sorted(walk_history[1])))
            all_walks["class2"].append(tuple(sorted(walk_history[2])))

        # Aggregate results

        # Join aggregated (count) results into Pandas DataFrame
        dfs = []
        for im_type, paths in all_walks.items():
            cc = Counter(paths)
            dfs.append(pd.DataFrame.from_dict(cc, orient="index", columns=[im_type]))
        df_cc = pd.concat(dfs, axis=1)

        # If combination doesnt exist replace nans with zeroes
        df_cc.fillna(0.0, inplace=True)

        # Calculate score
        if scoring_variant == "v1_1":
            df_cc["score"] = (df_cc["class1"] - df_cc["class2"]).abs() / df_cc[
                ["class1", "class2"]
            ].max(axis=1)
        elif scoring_variant == "v1_2":
            df_cc["score"] = (
                (df_cc["class1"] - df_cc["class2"]).abs()
                * df_cc.iloc[:, 0]
                / df_cc[["class1", "class2"]].max(axis=1)
            )
        elif scoring_variant == "v1_3":
            df_cc["score"] = (df_cc["class1"] - df_cc["class2"]).abs() / df_cc[
                ["class1", "class2"]
            ].max(axis=1).apply(np.sqrt)
        elif scoring_variant == "v1_4":
            df_cc["score"] = (
                (df_cc["class1"] - df_cc["class2"]).abs()
                * df_cc.sum(axis=1)
                / df_cc[["class1", "class2"]].max(axis=1).apply(np.sqrt)
            )
        else:
            raise SyntaxError(f"Unknown scoring variant: {scoring_variant}")

        # Aggregate score
        agg_score = {}
        for index, row in df_cc.iterrows():
            for single_feature in set(index):
                if single_feature in agg_score.keys():
                    agg_score[self.hg.edges.inverse[single_feature]] += row["score"]
                else:
                    agg_score[self.hg.edges.inverse[single_feature]] = row["score"]

        # Return feature-value pairs importances
        feature_value_importances = pd.DataFrame(
            {
                "feature_value": agg_score.keys(),
                "feature_importance": agg_score.values(),
            }
        ).set_index("feature_value")

        self.logger.info("Feature importances calculated.")

        if feature_values:
            return feature_value_importances
        else:
            # Return features importances
            feature_importances = self._modify_fi(feature_value_importances)
            return feature_importances

    def _rw_possible_steps(self, path_element, idx):
        if path_element == "edge":
            possible_pe = self.hg.incidence_matrix[idx, :]
            return possible_pe.nonzero()[1]
        elif path_element == "vertex":
            possible_pe = self.hg.incidence_matrix[:, idx]
            return possible_pe.nonzero()[0]
        else:
            raise SyntaxError(f"Unrecognized path_element: {path_element}")

    def fit(self, data: pd.DataFrame, label_column: str):
        """Fit data.

        Args:
            data (pd.DataFrame): Input data.
            label_column (str): Label column name.

        Returns:
            self
        """
        return super().fit(data, label_column)

    def fit_transform(self, data: pd.DataFrame, label_column: str, **fit_params):
        """Fit and transform data.

        Args:
            data (pd.DataFrame): Input data.
            label_column (str): Label column name.

        Returns:
            pd.DataFrame: DataFrame with feature names and feature importances columns.
        """
        return super().fit_transform(data, label_column, **fit_params)

    def transform(self):
        """Transform fitted data and return feature importances.

        Returns:
            pd.DataFrame: DataFrame with feature names and feature importances columns.
        """
        return self._random_walk_based_fi(
            iterations=self.iterations,
            walk_length=self.walk_length,
            scoring_variant=self.scoring_variant,
            feature_values=self.feature_values,
        )
