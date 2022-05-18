import io
import logging
from os import read
import sys
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATAFILES_PREFIX = Path.home() / ".hypper"
LOGGER = logging.getLogger(__name__)


def _download_file(filepath: str, url: str) -> str:
    filepath = Path(DATAFILES_PREFIX / filepath)
    if not filepath.exists():
        # Create directory
        Path(DATAFILES_PREFIX).mkdir(parents=True, exist_ok=True)
        # Download and unpack
        zipfile = ZipFile(io.BytesIO(urlopen(url).read()))
        zipfile.extractall(path=Path(DATAFILES_PREFIX))
    # Read dataset
    return filepath


def _accept_download(question: str, default: str = "yes") -> bool:
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    while True:
        sys.stdout.write(question + " [Y/n] ")
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def read_sample_data() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads custom sample dataset.

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    DF_TEST_STRUCTURE = {
        "x1": [
            "alpha",
            "alpha",
            "alpha",
            "beta",
            "beta",
            "gamma",
            "gamma",
            "gamma",
            "gamma",
        ],
        "x2": ["y", "x", "y", "x", "y", "x", "y", "x", "x"],
        "index": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        "y": [1, 1, 1, 1, 0, 0, 0, 0, 0],
    }
    return (
        pd.DataFrame.from_dict(DF_TEST_STRUCTURE).set_index("index"),
        "y",
        ["x1", "x2"],
    )


def read_german_data() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `German Credit Data`.

    **Original Dataset**: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

    **Modified Dataset**: https://www.kaggle.com/kabure/german-credit-data-with-risk/download

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    filepath = _download_file(
        filepath="german/german_credit_data.csv",
        url="https://github.com/hypper-team/hypper/raw/main/data/german.zip",
    )
    df = pd.read_csv(filepath, index_col=0)
    # df.drop(['Credit amount'], axis=1, inplace=True) # Only for feature importance purposes
    categorical_cols = [
        "Sex",
        "Job",
        "Housing",
        "Saving accounts",
        "Checking account",
        "Purpose",
    ]
    return df, "Risk", categorical_cols


def _read_abdominal_pain_data() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Acute Inflammations Data Set`.

    **Dataset**: https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    df = pd.read_csv(
        "data/abdominal_pain/diagnosis.data",
        names=[
            "patient_temperature",
            "occurance_of_nausea",
            "lumbar_pain",
            "urine_pushing",
            "mictorition_pain",
            "burning_of_urethra",
            "inflamation_of_urinary_bladder",  # class 1
            "nephritis_of_renal_pelvis_origin",  # class 2
        ],
    )
    # Combining 2 classes in single column
    df["label"] = np.select(
        [
            (df["inflamation_of_urinary_bladder"] == "yes")
            & (df["nephritis_of_renal_pelvis_origin"] == "yes"),
            (df["inflamation_of_urinary_bladder"] == "yes")
            & (df["nephritis_of_renal_pelvis_origin"] == "no"),
            (df["inflamation_of_urinary_bladder"] == "no")
            & (df["nephritis_of_renal_pelvis_origin"] == "yes"),
        ],
        [
            "both_diseases",
            "inflamation_of_urinary_bladder",
            "nephritis_of_renal_pelvis_origin",
        ],
        default=np.nan,
    )
    df.drop(
        ["inflamation_of_urinary_bladder", "nephritis_of_renal_pelvis_origin"],
        axis=1,
        inplace=True,
    )
    return df, "label", [i for i in df.columns if i != "label"]


def read_breast_cancer_data() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Breast Cancer Data Set`.

    **Dataset**: https://archive.ics.uci.edu/ml/datasets/breast+cancer

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    filepath = _download_file(
        filepath="breast_cancer/breast-cancer.data",
        url="https://github.com/hypper-team/hypper/raw/main/data/breast_cancer.zip",
    )
    return (
        pd.read_csv(
            filepath,
            names=[
                "label",
                "age",
                "menopause",
                "tumor_size",
                "inv_nodes",
                "node_caps",
                "deg_maling",
                "which_breast",
                "breast_quad",
                "irradiant",
            ],
        ),
        "label",
        [
            "age",
            "menopause",
            "tumor_size",
            "inv_nodes",
            "node_caps",
            "deg_maling",
            "which_breast",
            "breast_quad",
            "irradiant",
        ],
    )


def _read_car_evaluation_data() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Car Evaluation Data Set`.

    **Dataset**: https://archive.ics.uci.edu/ml/datasets/car+evaluation

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    df = pd.read_csv(
        "data/car_evaluation/car.data",
        names=[
            "buying_price",
            "maintenance_cost",
            "num_of_doors",
            "people_capacity",
            "lug_boot_size",
            "safety",
            "label",
        ],
    )
    return df, "label", [i for i in df.columns if i != "label"]


def read_spect_heart() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `SPECT Heart Data Set`.

    **Dataset**: https://archive.ics.uci.edu/ml/datasets/spect+heart

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    f_cols = [f"F{i}" for i in range(1, 23)]
    names = ["overall_diangnosis"] + f_cols
    filepath = _download_file(
        filepath="spect_heart",
        url="https://github.com/hypper-team/hypper/raw/main/data/spect_heart.zip",
    )
    df_train = pd.read_csv(filepath / "SPECT.train", names=names)
    df_test = pd.read_csv(filepath / "SPECT.test", names=names)
    return (
        pd.concat([df_train, df_test], ignore_index=True),
        "overall_diangnosis",
        f_cols,
    )


def read_congressional_voting_records() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Congressional Voting Records Data Set`.

    **Dataset**: https://archive.ics.uci.edu/ml/datasets/congressional+voting+records

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    filepath = _download_file(
        filepath="congressional_voting_records/house-votes-84.data",
        url="https://github.com/hypper-team/hypper/raw/main/data/congressional_voting_records.zip",
    )
    df = pd.read_csv(
        filepath,
        names=[
            "class_name",
            "handicapped_infants",
            "water_project_cost_sharing",
            "adoption_of_the_budget_resolution",
            "physician_fee_freeze",
            "el_salvador_aid",
            "religious_groups_in_schools",
            "anti_satellite_test_ban",
            "aid_to_nicaraguan_contras",
            "mx_missle",
            "immigration",
            "synfuels_corporation_cutback",
            "education_spending",
            "superfund_right_to_sue",
            "crime",
            "duty_free_exports",
            "export_administration_act_south_africa",
        ],
    )
    # Replace '?' with NaNs
    return (
        df.replace("?", np.nan),
        "class_name",
        [i for i in df.columns if i != "class_name"],
    )


def read_criteo(
    nrows: Optional[int] = 1000000, size: Optional[float] = 0.1
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Criteo Sponsored Search Conversion Log Dataset`.

    **Dataset**: https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/

    Args:
        nrows (Optional[int], optional): Read n first rows from file. Defaults to 1000000.
        size (Optional[float], optional): Select percentage of rows with respect to classes distribution. Defaults to 0.1.

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """

    filepath = Path(DATAFILES_PREFIX / "criteo")
    url = "http://go.criteo.net/criteo-research-search-conversion.tar.gz"
    if not filepath.exists():
        LOGGER.warning(
            "Compressed Criteo dataset size is ~2GB. When uncompressed, Criteo will take ~6.4GB of yor disk space."
        )
        if _accept_download("Do you want to continue downloading?"):
            LOGGER.warning("Downloading Criteo dataset. It may take a few minutes ...")
            filepath.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(urlopen(url).read())) as tar:
                tar.extractall(path=filepath)
        else:
            return

    df = pd.read_csv(
        filepath / "Criteo_Conversion_Search" / "CriteoSearchData",
        delimiter="\t",
        names=[
            "Sale",
            "SalesAmountInEuro",
            "time_delay_for_conversion",
            "click_timestamp",
            "nb_clicks_1week",
            "product_price",
            "product_age_group",
            "device_type",
            "audience_id",
            "product_gender",
            "product_brand",
            "product_category_1",
            "product_category_2",
            "product_category_3",
            "product_category_4",
            "product_category_5",
            "product_category_6",
            "product_category_7",
            "product_country",
            "product_id",
            "product_title",
            "partner_id",
            "user_id",
        ],
        nrows=nrows,
        # usecols=['Sale', 'click_timestamp',\
        #     'nb_clicks_1week','product_age_group' ,'device_type',\
        #     'audience_id' ,'product_gender' ,'product_brand' ,'product_category_1',\
        #     'product_category_2', 'product_category_3', 'product_category_4', \
        #     'product_category_5', 'product_category_6', 'product_category_7',
        #     'product_country', 'product_id' ,'product_title' ,'partner_id' ,'user_id'],
        usecols=[
            "Sale",
            "nb_clicks_1week",
            "product_age_group",
            "device_type",
            "audience_id",
            "product_gender",
            "product_brand",
            "product_category_1",
            "product_category_2",
            "product_category_3",
            "product_category_4",
            "product_category_5",
            "product_category_6",
            "product_category_7",
            "product_country",
            "product_id",
            "partner_id",
        ],
        dtype={
            "Sale": np.int8,
            "click_timestamp": np.int64,
            "nb_clicks_1week": np.int64,
            "product_age_group": "object",
            "device_type": "object",
            "audience_id": "object",
            "product_gender": "object",
            "product_brand": "object",
            "product_category_1": "object",
            "product_category_2": "object",
            "product_category_3": "object",
            "product_category_4": "object",
            "product_category_5": "object",
            "product_category_6": "object",
            "product_category_7": "object",
            "product_country": "object",
            "product_id": "object",
            "product_title": "object",
            "partner_id": "object",
            "user_id": "object",
        },
    )
    # df.drop(['SalesAmountInEuro' ,'time_delay_for_conversion', 'product_price'], axis=1, inplace=True)# removing other labels
    # return df, 'Sale'
    _, X, _, y = train_test_split(
        df.drop("Sale", axis=1),
        df["Sale"],
        test_size=size,
        random_state=42,
        stratify=df["Sale"],
    )
    X["Sale"] = y
    return (
        X,
        "Sale",
        [
            "product_age_group",
            "device_type",
            "audience_id",
            "product_gender",
            "product_brand",
            "product_category_1",
            "product_category_2",
            "product_category_3",
            "product_category_4",
            "product_category_5",
            "product_category_6",
            "product_category_7",
            "product_country",
            "product_id",
            "partner_id",
        ],
    )


def read_banking() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Banking Dataset - Marketing Targets`.

    **Dataset**: https://www.kaggle.com/prakharrathi25/banking-dataset-marketing-targets

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    filepath = _download_file(
        filepath="banking_dataset",
        url="https://github.com/hypper-team/hypper/raw/main/data/banking_dataset.zip",
    )
    df_train = pd.read_csv(filepath / "train.csv", delimiter=";")
    # df_test = pd.read_csv(filepath / 'test.csv', delimiter=';')
    # pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    return (
        df_train,
        "y",
        [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "poutcome",
        ],
    )


def read_phishing() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Phishing Dataset`.

    **Dataset**: https://www.kaggle.com/shashwatwork/phishing-dataset-for-machine-learning

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    filepath = _download_file(
        filepath="phishing_websites/phishing.csv",
        url="https://github.com/hypper-team/hypper/raw/main/data/phishing_websites.zip",
    )
    df = pd.read_csv(filepath, index_col=0)
    return df, "class_name", [i for i in df.columns if i != "class_name"]


def read_churn() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `Churn Modelling`.

    **Dataset**: https://www.kaggle.com/shrutimechlearn/churn-modelling

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    filepath = _download_file(
        filepath="churn_modelling/Churn_Modelling.csv",
        url="https://github.com/hypper-team/hypper/raw/main/data/churn_modelling.zip",
    )
    df = pd.read_csv(filepath, index_col=0)
    df.drop(["CustomerId", "Surname"], axis=1, inplace=True)
    return df, "Exited", ["Geography", "Gender", "HasCrCard", "IsActiveMember"]


def read_hr() -> Tuple[pd.DataFrame, str, List[str]]:
    """Loads `HR Analytics: Job Change of Data Scientists`.

    **Dataset**: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

    Returns:
        Tuple[pd.DataFrame, str, list]: Tuple with DataFrame object, label column name and list of categorical columns.
    """
    filepath = _download_file(
        filepath="hr_analytics/aug_train.csv",
        url="https://github.com/hypper-team/hypper/raw/main/data/hr_analytics.zip",
    )
    df_train = pd.read_csv(filepath)
    df_train.drop(["enrollee_id"], axis=1, inplace=True)
    return df_train, "target", [i for i in df_train.columns if i != "target"]
