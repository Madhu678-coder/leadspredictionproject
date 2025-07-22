import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
def replace_unknowns_with_nan(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'category']).columns:
        df[col] = df[col].replace(r'(?i)unknown', np.nan, regex=True)
    return df

def handle_skewness_yeojohnson(df, threshold=0.5):
    df = df.copy()
    num_cols = df.select_dtypes(include='number').columns
    # Initialize transformer for Yeo-Johnson
    pt = PowerTransformer(method='yeo-johnson')
    for col in num_cols:
        if abs(df[col].skew()) > threshold:
            # Reshape and transform the column
            reshaped = df[col].values.reshape(-1, 1)
            try:
                df[col] = pt.fit_transform(reshaped)
            except Exception as e:
                print(f"Skipping column {col}: {e}")
    return df


replace_unknowns_transformer = FunctionTransformer(replace_unknowns_with_nan, validate=False)
skewness_transformer = FunctionTransformer(handle_skewness_yeojohnson, validate=False)

