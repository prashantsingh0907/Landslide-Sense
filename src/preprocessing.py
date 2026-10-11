"""
Preprocessing Pipeline for LandslideSense
Constructs leak-free ColumnTransformer pipelines with numerical median imputation,
standard scaling, and categorical one-hot encoding.
"""

from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer configured for numerical and categorical predictors.
    """
    transformers = []

    if numerical_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, numerical_features))

    if categorical_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"  # Exclude any unintended columns
    )
    return preprocessor
