import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath)
    
    # Drop column 'Date' if exists
    if 'Date' in df.columns:
        df.drop(columns=['Date'], inplace=True)
    
    # Remove duplicated data
    df.drop_duplicates(inplace=True)
    
    # Handle missing values with median
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Identify numeric columns after encoding
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Standardize numeric features
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
