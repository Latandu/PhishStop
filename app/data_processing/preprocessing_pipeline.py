
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import polars as pl
import numpy as np
class PreprocessingPipeline:
    """Class to build preprocessing pipeline for numeric features"""

    def __init__(self):
        pass

    def build_preprocessing_pipeline(self):
        """Build preprocessing pipeline for numeric features"""

        log_features = [
            "num_exclamation_marks", "num_uppercase_words", "num_malicious_links"
        ]
        
        numeric_features = [
            "sender_domain_entropy",
            "spf_flag_missing", "dkim_flag_missing",
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
    
    def process_pipeline(self, combined_df):
        df_pd = combined_df.to_pandas()
        preprocessor, feature_cols = self.build_preprocessing_pipeline()
        X_numeric = preprocessor.fit_transform(df_pd)
        X_processed = pl.DataFrame(X_numeric, schema=feature_cols)
        df = pl.concat([combined_df.drop(feature_cols), X_processed], how="horizontal")
        return df