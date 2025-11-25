import polars as pl
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from preprocessing_pipeline import PreprocessingPipeline
import numpy as np
from pathlib import Path

# Get the script's directory and construct absolute paths
script_dir = Path(__file__).parent
data_dir = script_dir.parent.parent / "data"
parquet_dir = data_dir / "parquet"

df_phishing = pl.read_parquet(parquet_dir / "emails_phishing_jose.parquet").drop("Date", strict=False).drop("return_path_domain_missing", strict=False)
df_csv = pl.read_parquet(parquet_dir / "emails_csv.parquet").drop("return_path_domain_missing", strict=False)
df_gmail = pl.read_parquet(parquet_dir / "emails_gmail.parquet").drop("Date", strict=False).drop("return_path_domain_missing", strict=False)
df_nigerian = pl.read_parquet(parquet_dir / "phishing_nigerian.parquet").drop("Date", strict=False).drop("return_path_domain_missing", strict=False)   

combined_df = pl.concat([df_phishing, df_csv, df_gmail, df_nigerian], how="diagonal_relaxed")
print(combined_df.shape)
df = PreprocessingPipeline().process_pipeline(combined_df)
df.write_parquet(data_dir / "emails_v5.parquet")