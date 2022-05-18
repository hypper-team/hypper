import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from hypper.undersampling import CDWU
from imblearn.under_sampling import (
    EditedNearestNeighbours,
    NearMiss,
    RandomUnderSampler,
    TomekLinks,
)
from lightgbm import LGBMClassifier
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
from tqdm import tqdm
from xgboost import XGBClassifier

from base_benchmark import BaseBenchmark

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class UndersamplingBenchmark(BaseBenchmark):
    def __init__(self, config_filepath: str) -> None:
        self._load_config(config_filepath)
        super().__init__(
            param_grid=self.config["param_grid"],
            savefile=Path(__file__).parent / "results_df" / self.config["savefile"],
            verbosity=self.config["verbosity"],
        )
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

    def run_benchmark(self):
        # Benchmark results structure
        df_out = pd.DataFrame(
            columns=[
                "dataset",
                "undersampling_method",
                "eval_algorithm",
                "parameters",
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
                # X_train[np.isnan(X_train)] = 0
                # X_test[np.isnan(X_test)] = 0
                # For every parameter set
                for params in self.param_grid:
                    if self.verbosity:
                        print(f"{load_function.__name__} | {params}")
                    # Resample train data
                    if params["method"] == "hypper":
                        new_columns = list(dataset.columns)
                        new_columns.append(new_columns.pop(new_columns.index(label)))
                        X_resampled, y_resampled = self._hypper(
                            pd.DataFrame(
                                np.hstack((X_train, np.expand_dims(y_train, axis=1))),
                                columns=new_columns,
                            ),
                            label,
                            weighting_iteration=params["weighting_iteration"],
                            weighting_normalization_strategy=params[
                                "weighting_normalization_strategy"
                            ],
                            majority_left_threshold=params["majority_left_threshold"],
                            weighting_history=params["weighting_history"],
                            version=params["version"],
                            randomize_A=params["randomize_A"],
                        )
                    else:
                        try:
                            X_train = np.nan_to_num(X_train)  # Default to 0.0
                            if params["method"] == "without_undersampling":
                                X_resampled, y_resampled = self._without_undersampling(
                                    X_train, y_train
                                )
                            elif params["method"] == "random_undersampling":
                                X_resampled, y_resampled = self._random_undersampling(
                                    X_train, y_train
                                )
                            elif params["method"] == "tomek_links":
                                X_resampled, y_resampled = self._tomek_links(
                                    X_train, y_train
                                )
                            elif params["method"] == "near_miss":
                                X_resampled, y_resampled = self._near_miss(
                                    X_train, y_train, params["version"]
                                )
                            elif params["method"] == "edited_nearest_neighbours":
                                (
                                    X_resampled,
                                    y_resampled,
                                ) = self._edited_nearest_neighbours(X_train, y_train)
                            else:
                                raise Exception(
                                    f"Unrecognized method name: {params['method']}"
                                )
                        except MemoryError as me:
                            warnings.warn(
                                f"{params['method']} could not handle data size - {me}"
                            )
                            continue
                    # Eval on all predefined classifiers
                    for classifier_name, clf in self.eval_algorithms:
                        # Train classifier
                        if classifier_name == "catboost":
                            X_resampled = pd.DataFrame(
                                X_resampled,
                                columns=[
                                    col for col in dataset.columns if col != label
                                ],
                            )
                            clf.fit(X_resampled, y_resampled, cat_features=cat_cols)
                        elif classifier_name == "lightgbm":
                            X_resampled = pd.DataFrame(
                                X_resampled,
                                columns=[
                                    col for col in dataset.columns if col != label
                                ],
                            )
                            clf.fit(
                                X_resampled, y_resampled, categorical_feature=cat_cols
                            )
                        else:
                            clf.fit(X_resampled, y_resampled)
                        # Evaluate results
                        ## Test
                        y_pred_test = clf.predict(X_test)
                        y_pred_prob_test = clf.predict_proba(X_test)[:, 1]
                        y_true_test = y_test
                        # Val
                        y_pred_val = clf.predict(X_val)
                        y_pred_prob_val = clf.predict_proba(X_val)[:, 1]
                        y_true_val = y_val
                        if self.verbosity:
                            print(
                                f"{classifier_name}: {roc_auc_score(y_true_test, y_pred_prob_test)}"
                            )
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
                                            accuracy_score(y_true_val, y_pred_val),
                                            balanced_accuracy_score(
                                                y_true_val, y_pred_val
                                            ),
                                            roc_auc_score(y_true_val, y_pred_prob_val),
                                            precision_score(y_true_val, y_pred_val),
                                            average_precision_score(
                                                y_true_val, y_pred_prob_val
                                            ),
                                            recall_score(y_true_val, y_pred_val),
                                            f1_score(y_true_val, y_pred_val),
                                            accuracy_score(y_true_test, y_pred_test),
                                            balanced_accuracy_score(
                                                y_true_test, y_pred_test
                                            ),
                                            roc_auc_score(
                                                y_true_test, y_pred_prob_test
                                            ),
                                            precision_score(y_true_test, y_pred_test),
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

    def _hypper(
        self,
        df,
        label_col,
        weighting_iteration,
        weighting_normalization_strategy,
        majority_left_threshold,
        weighting_history,
        version,
        randomize_A,
    ):
        cdwu = CDWU(
            weighting_iterations=weighting_iteration,
            weighting_normalization_strategy=weighting_normalization_strategy,
            majority_left_threshold=majority_left_threshold,
            weighting_history=weighting_history,
            version=version,
            randomize_A=randomize_A,
            random_seed=self.random_seed,
        )
        df_out = cdwu.fit_transform(df, label_col)
        df_out.fillna(0, inplace=True)
        return df_out.drop(label_col, axis=1).to_numpy(), df_out[label_col].to_numpy()

    def _without_undersampling(self, X_train, y_train):
        return X_train, y_train

    def _random_undersampling(self, X_train, y_train):
        return RandomUnderSampler(
            random_state=self.random_seed, sampling_strategy=1.0
        ).fit_resample(X_train, y_train)

    def _tomek_links(self, X_train, y_train):
        return TomekLinks(n_jobs=-1).fit_resample(X_train, y_train)

    def _near_miss(self, X_train, y_train, version):
        return NearMiss(version=version).fit_resample(X_train, y_train)

    def _edited_nearest_neighbours(self, X_train, y_train):
        return EditedNearestNeighbours(n_jobs=-1).fit_resample(X_train, y_train)


def main():
    ub = UndersamplingBenchmark(config_filepath="undersampling.yaml")
    ub.run_benchmark()


if __name__ == "__main__":
    main()
