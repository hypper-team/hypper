import pandas as pd

from hypper.data_reader import read_banking, read_breast_cancer_data, read_churn, \
    read_congressional_voting_records, read_german_data, read_hr, read_phishing, read_spect_heart

def test_reading_data():
    for read_fun in [read_banking, read_breast_cancer_data, read_churn, read_congressional_voting_records, read_german_data, read_hr, read_phishing, read_spect_heart]:
        df, label, cat_cols = read_fun()
        assert type(df) == pd.DataFrame
        assert type(label) == str
        assert type(cat_cols) == list
    return True