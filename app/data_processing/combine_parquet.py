import polars as pl
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from preprocessing_pipeline import PreprocessingPipeline
import numpy as np
from pathlib import Path

df_phishing = pl.read_parquet( "emails_phishing_jose.parquet").drop("Date", strict=False)
df_csv = pl.read_parquet("emails_csv.parquet")
df_gmail = pl.read_parquet("emails_gmail.parquet").drop("Date", strict=False)
df_nigerian = pl.read_parquet("phishing_nigerian.parquet").drop("Date", strict=False)

combined_df = pl.concat([df_phishing, df_csv, df_gmail, df_nigerian], how="diagonal_relaxed")
print(combined_df.shape)
combined_df.write_parquet("emails_v6.parquet")
