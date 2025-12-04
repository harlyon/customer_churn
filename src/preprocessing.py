from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

def create_preprocessor():
    """Creates the ColumnTransformer for the pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OrdinalEncoder(), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    return preprocessor