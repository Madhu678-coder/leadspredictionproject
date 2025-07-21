from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from data.data_preprocessing.mapping import map_categorical_columns, mapping_transformer
from data.data_preprocessing.transformations import replace_unknowns_transformer, skewness_transformer

def build_full_pipeline(df):
    target_col = "Converted"

    label_cols = ["Lead Quality", "Asymmetrique Activity Index", "Asymmetrique Profile Index"]

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if target_col in label_cols:
        label_cols.remove(target_col)

    cat_ohe_cols = [col for col in categorical_cols if col not in label_cols]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    cat_ohe_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    cat_label_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('label', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat_ohe', cat_ohe_pipeline, cat_ohe_cols),
        ('cat_label', cat_label_pipeline, label_cols)
    ], remainder='passthrough')

    full_pipeline = Pipeline([
        ('mapping', mapping_transformer),
        ('replace_unknowns', replace_unknowns_transformer),
        ('skewness', skewness_transformer),
        ('preprocessor', preprocessor)
    ])

    full_pipeline.set_output(transform='default')
    return full_pipeline
