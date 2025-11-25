import polars as pl
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
import numpy as np
df_phishing = pl.read_parquet("emails_phishing_jose.parquet").drop("Date", strict=False)
df_csv = pl.read_parquet("emails_csv.parquet")
df_gmail = pl.read_parquet("emails_gmail.parquet").drop("Date", strict=False)
df_nigerian = pl.read_parquet("phishing_nigerian.parquet").drop("Date", strict=False)

combined_df = pl.concat([df_phishing, df_csv, df_gmail, df_nigerian], how="diagonal_relaxed")
print(combined_df.shape)

def build_preprocessing_pipeline():
    log_features = [
        "num_exclamation_marks", "num_malicious_links"
    ]
    
    numeric_features = [
        "sender_domain_entropy",
        "spf_flag_missing", "dkim_flag_missing", "return_path_domain_missing",
        "num_links", "subject_length", "body_length", "keyword_count", "num_received_headers", 
    ]
    
    log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", MinMaxScaler())
    ])
    
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", MinMaxScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ("log_numeric", log_pipeline, log_features),
        ("numeric", numeric_pipeline, numeric_features)
    ], remainder="drop")
    
    all_features = log_features + numeric_features
    return preprocessor, all_features

df_pd = combined_df.to_pandas()
preprocessor, feature_cols = build_preprocessing_pipeline()
X_numeric = preprocessor.fit_transform(df_pd)
X_processed = pl.DataFrame(X_numeric, schema=feature_cols)
df = pl.concat([combined_df.drop(feature_cols), X_processed], how="horizontal")
    
df.write_parquet("data/emails_v5.parquet")