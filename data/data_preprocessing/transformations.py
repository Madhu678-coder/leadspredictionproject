import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def replace_unknowns_with_nan(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].replace(r'(?i)unknown', np.nan, regex=True)
    return df

def handle_skewness(df, threshold=0.5):
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if abs(df[col].skew()) > threshold:
            df[col] = df[col].apply(lambda x: np.log1p(x) if pd.notnull(x) and x >= 0 else x)
    return df

replace_unknowns_transformer = FunctionTransformer(replace_unknowns_with_nan, validate=False)
skewness_transformer = FunctionTransformer(handle_skewness, validate=False)
