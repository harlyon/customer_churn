import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, ID_COLS, TARGET_COL

def load_data(path=DATA_PATH):
    """Loads CSV and drops unnecessary ID columns."""
    try:
        df = pd.read_csv(path)
        print(f"✅ Data loaded. Shape: {df.shape}")

        # Drop IDs if they exist
        df = df.drop(columns=[col for col in ID_COLS if col in df.columns], errors='ignore')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"❌ File not found at {path}. Please check the path.")

def get_data_split(df, test_size=0.2):
    """Splits data into X and y, then train and test sets."""
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Stratify is important for churn data to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test