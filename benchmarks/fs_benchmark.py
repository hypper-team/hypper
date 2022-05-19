import math
import random
import warnings
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from hypper.feature_selection import CDWFS, RandomWalkFS
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from base_benchmark import BaseBenchmark


class FeatureSelectionBenchmark(BaseBenchmark):
    def __init__(self, config_filepath: str) -> None:
        self._load_config(config_filepath)
        super().__init__(
            param_grid=self.config["param_grid"],
            savefile=Path(__file__).parent / "results_df" / self.config["savefile"],
        )
        self.feature_values = self.config["feature_values"]
        self.kfold_splits = self.config["kfold_splits"]
        self.skfold = StratifiedKFold(
            n_splits=self.kfold_splits, shuffle=True, random_state=self.random_seed
        )
        self.eval_algorithms = []
        for ea in self.config["eval_algorithms"]:
            if ea == "LogisticRegression":
                clf = LogisticRegression(
                    random_state=self.random_seed, class_weight="balanced"
                )
            elif ea == "XGBClassifier":
                clf = XGBClassifier(
                    random_state=self.random_seed,
                    eval_metric="logloss",
                    use_label_encoder=False,
                )
            elif ea == "MLPClassifier":
                clf = MLPClassifier(random_state=self.random_seed)
            elif ea == "CatBoostClassifier":
                clf = CatBoostClassifier(random_seed=self.random_seed, verbose=False)
            elif ea == "LGBMClassifier":
                clf = LGBMClassifier(random_state=self.random_seed)
            else:
                raise ValueError(f"Not recognized classifier: {ea}")
            self.eval_algorithms.append((ea, clf))
        self.percent_of_features = self.config["percent_of_features"]

    def run_benchmark(self):
        # Benchmark results structure
        df_out = pd.DataFrame(
            columns=[
                "dataset",
                "fs_method",
                "eval_algorithm",
                "parameters",
                "po_features",
                "n_features",
                "accuracy_val",
                "balanced_accuracy_val",
                "roc_auc_val",
                "average_precision_val",
                "precision_val",
                "recall_val",
                "f1_val",
                "accuracy_test",
                "balanced_accuracy_test",
                "roc_auc_test",
                "average_precision_test",
                "precision_test",
                "recall_test",
                "f1_test",
            ]
        )

        # For every dataset
        for load_function in self.load_datasets():
            print(f"Processing datset with load function: {load_function}")
            dataset, label, cat_cols = load_function()
            # Preprocess dataset
            dataset = self._cat_oe(dataset, cat_cols)  # Transform categorical features
            X, y = dataset.drop(label, axis=1).values, dataset[label].values
            y = self._cat_le(y)
            # For every fold in KFolds
            for split_train, split_test in tqdm(
                self.skfold.split(X, y), total=self.kfold_splits
            ):
                # Prepare data for particular fold
                np.random.seed(self.random_seed)
                np.random.shuffle(split_test)
                val_amount = int(0.5 * split_test.shape[0])
                split_validation, split_test = (
                    split_test[:val_amount],
                    split_test[val_amount:],
                )
                X_train, X_test, X_val = (
                    X[split_train],
                    X[split_test],
                    X[split_validation],
                )
                y_train, y_test, y_val = (
                    y[split_train],
                    y[split_test],
                    y[split_validation],
                )
                X_test = np.nan_to_num(X_test)
                X_val = np.nan_to_num(X_val)
                if not self.feature_values:
                    X_train = X_train.astype(float)
                    X_train = np.nan_to_num(X_train)
                    # X_train[np.isnan(X_train)] = 0 # Simplification for other algos | Hypper handles NaNs
                # X_test[np.isnan(X_test)] = 0 # Check if Hypper methods should have separate evaluation with NaNs
                # Calculate DF
                X_train_columns = list(dataset.columns)
                X_train_columns.append(
                    X_train_columns.pop(X_train_columns.index(label))
                )
                # Hypper doesn't need dataset previously transformed with OHE
                df_train = pd.DataFrame(
                    np.hstack((X_train, np.expand_dims(y_train, axis=1))),
                    columns=X_train_columns,
                )
                X_train_columns = X_train_columns[:-1]  # Without label
                # Apply OHE
                if self.feature_values:
                    ohe = OneHotEncoder(drop=None, sparse=True, handle_unknown="ignore")
                    # ohe.fit(np.vstack((X_train, X_test)))
                    ohe.fit(X_train)
                    X_train = ohe.transform(X_train)
                    X_test = ohe.transform(X_test)
                    X_val = ohe.transform(X_val)
                    X_train_columns = ohe.get_feature_names_out(X_train_columns)
                # For every parameter set
                for params in self.param_grid:
                    if self.verbosity:
                        print(f"{load_function.__name__} | {params}")
                    # Find feature importances train data
                    if params["method"] == "hypper_weights":
                        df_fs_results = self._hypper_weights(
                            df_train,
                            label,
                            params["weighting_iterations"],
                            params["weighting_normalization_strategy"],
                            params["feature_values"],
                        )
                    elif params["method"] == "hypper_rw":
                        df_fs_results = self._hypper_rw(
                            df_train,
                            label,
                            params["iterations"],
                            params["walk_length"],
                            params["scoring_variant"],
                            params["feature_values"],
                        )
                    elif params["method"] == "random":
                        df_fs_results = self._random(X_train, X_train_columns)
                    elif params["method"] == "random_forest":
                        df_fs_results = self._random_forest(
                            X_train, y_train, X_train_columns
                        )
                    elif params["method"] == "logistic_regression":
                        df_fs_results = self._logistic_regression(
                            X_train, y_train, X_train_columns
                        )
                    else:
                        raise Exception(f"Unrecognized method name: {params['method']}")
                    # For n features
                    for pof in self.percent_of_features:
                        nof = math.ceil(pof * len(df_fs_results.index))
                        df_fs_results_n_features = df_fs_results.head(nof)
                        # Preprocess
                        if not self.feature_values:
                            ## Test
                            X_test = X_test.astype(float)
                            X_test = np.nan_to_num(X_test)
                            X_test = sparse.csr_matrix(X_test)
                            ## Val
                            X_val = X_val.astype(float)
                            X_val = np.nan_to_num(X_val)
                            X_val = sparse.csr_matrix(X_val)
                        # Create dataset with selected features
                        ## Test
                        df_test = pd.DataFrame.sparse.from_spmatrix(
                            X_test, columns=X_train_columns
                        )  # When X_test will be modified with OHE data leak occures
                        # (with X_train, otherwise non recognized values occur)
                        n_features = df_fs_results_n_features.index.tolist()
                        X_test_n_features = copy(df_test[n_features])
                        ## Val
                        df_val = pd.DataFrame.sparse.from_spmatrix(
                            X_val, columns=X_train_columns
                        )
                        X_val_n_features = copy(df_val[n_features])
                        # Eval on all predefined classifiers
                        for classifier_name, clf in self.eval_algorithms:
                            if self.feature_values:
                                temp_cat_cols = n_features
                            else:
                                temp_cat_cols = [i for i in cat_cols if i in n_features]
                            for col in temp_cat_cols:
                                X_test_n_features[col] = X_test_n_features[col].astype(
                                    int, copy=False
                                )
                                X_val_n_features[col] = X_val_n_features[col].astype(
                                    int, copy=False
                                )
                            # Train classifier - test
                            if classifier_name == "catboost":
                                clf.fit(
                                    X_test_n_features,
                                    y_test,
                                    cat_features=temp_cat_cols,
                                )
                            elif classifier_name == "lightgbm":
                                clf.fit(
                                    X_test_n_features,
                                    y_test,
                                    categorical_feature=temp_cat_cols,
                                )
                            else:
                                clf.fit(X_test_n_features.values, y_test)
                            # Evaluate results
                            y_pred_test = clf.predict(X_test_n_features)
                            y_pred_prob_test = clf.predict_proba(X_test_n_features)[
                                :, 1
                            ]
                            y_true_test = y_test
                            # Train classifier - val
                            if classifier_name == "catboost":
                                clf.fit(
                                    X_val_n_features, y_val, cat_features=temp_cat_cols
                                )
                            elif classifier_name == "lightgbm":
                                clf.fit(
                                    X_val_n_features,
                                    y_val,
                                    categorical_feature=temp_cat_cols,
                                )
                            else:
                                clf.fit(X_val_n_features.values, y_val)
                            # Evaluate results
                            y_pred_val = clf.predict(X_val_n_features)
                            y_pred_prob_val = clf.predict_proba(X_val_n_features)[:, 1]
                            y_true_val = y_val
                            # Append results
                            df_out = pd.concat(
                                [
                                    df_out,
                                    pd.DataFrame(
                                        [
                                            [
                                                load_function.__name__,
                                                params["method"],
                                                classifier_name,
                                                params,
                                                pof,
                                                df_fs_results_n_features.shape[0],
                                                accuracy_score(y_true_val, y_pred_val),
                                                balanced_accuracy_score(
                                                    y_true_val, y_pred_val
                                                ),
                                                roc_auc_score(
                                                    y_true_val, y_pred_prob_val
                                                ),
                                                precision_score(y_true_val, y_pred_val),
                                                average_precision_score(
                                                    y_true_val, y_pred_prob_val
                                                ),
                                                recall_score(y_true_val, y_pred_val),
                                                f1_score(y_true_val, y_pred_val),
                                                accuracy_score(
                                                    y_true_test, y_pred_test
                                                ),
                                                balanced_accuracy_score(
                                                    y_true_test, y_pred_test
                                                ),
                                                roc_auc_score(
                                                    y_true_test, y_pred_prob_test
                                                ),
                                                precision_score(
                                                    y_true_test, y_pred_test
                                                ),
                                                average_precision_score(
                                                    y_true_test, y_pred_prob_test
                                                ),
                                                recall_score(y_true_test, y_pred_test),
                                                f1_score(y_true_test, y_pred_test),
                                            ]
                                        ],
                                        columns=df_out.columns,
                                    ),
                                ],
                                ignore_index=True,
                            )
        self.save(df_out)

    def _logistic_regression(self, X, y, X_cols):
        clf = LogisticRegression(n_jobs=-1, random_state=self.random_seed)
        clf.fit(X, y)
        return pd.Series(np.abs(clf.coef_[0]), index=X_cols).sort_values(
            ascending=False
        )

    def _random_forest(self, X, y, X_cols):
        clf = RandomForestClassifier(n_jobs=-1, random_state=self.random_seed)
        clf.fit(X, y)
        return pd.Series(clf.feature_importances_, index=X_cols).sort_values(
            ascending=False
        )

    def _random(self, X, cols):
        random.Random(self.random_seed).shuffle(cols)
        return pd.Series(np.ones(len(cols)), index=cols)

    def _hypper_rw(
        self, df, label, iterations, walk_length, scoring_variant, feature_values
    ):
        hrw = RandomWalkFS(
            iterations=iterations,
            walk_length=walk_length,
            scoring_variant=scoring_variant,
            feature_values=feature_values,
            random_seed=self.random_seed,
        )
        return hrw.fit_transform(df, label).squeeze().sort_values(ascending=False)

    def _hypper_weights(
        self,
        df,
        label,
        weighting_iteration,
        weighting_normalization_strategy,
        feature_values,
    ):
        hw = CDWFS(
            weighting_iterations=weighting_iteration,
            weighting_normalization_strategy=weighting_normalization_strategy,
            feature_values=feature_values,
        )
        return hw.fit_transform(df, label).squeeze().sort_values(ascending=False)


def main():
    ub = FeatureSelectionBenchmark(config_filepath="fvs.yaml")
    ub.run_benchmark()


if __name__ == "__main__":
    main()
